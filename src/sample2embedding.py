import torch
import torch.nn.functional as F
import copy
import json

import sys
import os
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
)
from megatron import mpu
from megatron.text_generation_utils import forward_model, pad_batch, get_batch, filter_logits, stop_tokens_in_completion, switch
from megatron.utils import setup_for_inference_or_eval
from src.util import dir2samples



def input2tokens(lines: iter, tokenizer):
    result = []
    for line_id, line in lines:
        line = line.strip()
        if line:
            tokens = tokenizer.tokenize(line)
            assert len(tokens) < 1024, f"Sample to big. Max tokens should be 1023 but got {len(tokens)}"
            result.append((line_id, [tokenizer.tokenize(line)]))
    return result


def samples2inputs(samples: iter):
    questions = []
    answers = []
    for sample_id, _, sample_q, sample_a1, sample_a2, sample_a3 in samples:
        if not sample_q:
            continue
        questions.append(sample_id, sample_q)
        answers.append(sample_id, sample_a1)
        if sample_a2:
            answers.append(sample_id, sample_a2)
        if sample_a3:
            answers.append(sample_id, sample_a3)
    
    return questions, answers

# input_file = neox_args.sample_input_file,
# output_file = neox_args.sample_output_file


def tokens2embeddings(context_tokens, model, neox_args, stop_tokens=None):
    model.module.clear_cache()

    model.eval()

    # pad batch in order to allow conversion to tensor
    
    context_tokens, context_lengths = pad_batch(
        copy.deepcopy(context_tokens), # TODO not sure i need to deep copy in this case
        pad_id=neox_args.tokenizer.eod,
        pad_len=neox_args.seq_length,
    )
    

    # convert to tensor and broadcast
    context_tokens = torch.cuda.LongTensor(context_tokens)
    if stop_tokens:
        stop_tokens = torch.cuda.LongTensor(stop_tokens)
        if stop_tokens.ndim == 1:
            stop_tokens = stop_tokens.unsqueeze(0)



    # Make sure context tokens + start tokens are the same across all ranks
    token_generation_start_index = torch.cuda.LongTensor(context_lengths)
    torch.distributed.broadcast(
        context_tokens,
        mpu.get_model_parallel_src_rank(),
        group=mpu.get_model_parallel_group()
    )
    torch.distributed.broadcast(
        token_generation_start_index,
        mpu.get_model_parallel_src_rank(),
        group=mpu.get_model_parallel_group()
    )


    # get attention mask / position ids
    context_tokens, attention_mask, position_ids = get_batch(neox_args, context_tokens)

    
    
    batch_size = context_tokens.size(0)
    
    # get the context_index at which generation is to start
    # we start generation at the position where the smallest context ends
    first_token_index_to_generate = token_generation_start_index.min().item()
    maximum_tokens = neox_args.maximum_tokens or (neox_args.seq_length - token_generation_start_index.max().item() - 1)
    last_token_index_to_generate = min(
        neox_args.seq_length
        - 1,  # never generate more than the model's sequence length
        first_token_index_to_generate + maximum_tokens - 1,
    )


    with torch.no_grad():
        # initialize generation variables
        state_is_done = torch.zeros([batch_size]).byte().cuda()

        
        for i in range(first_token_index_to_generate, last_token_index_to_generate + 1):
            if neox_args.recompute:
                model_inputs = (context_tokens, position_ids, attention_mask)
            else:  # use kv cache
                if i == first_token_index_to_generate:
                    tokens_to_use = context_tokens[:, :i]
                    positions_to_use = position_ids[:, :i]
                else:
                    tokens_to_use = context_tokens[:, i - 1].view(batch_size, -1)
                    positions_to_use = position_ids[:, i - 1].view(batch_size, -1)

                model_inputs = (
                    tokens_to_use,  # input_ids
                    positions_to_use,  # position_ids
                    attention_mask,  # attention_mask
                )


            logits = forward_model(model, model_inputs, neox_args.is_pipe_parallel)
            
            if logits is not None:  # if pipe parallel, not all ranks return logits
                generated_token_logits = logits[:, -1].view(batch_size, -1).contiguous() # [bs, seq, vocab_size] -> [bs, vocab_size]
                if neox_args.temperature == 0.0 and neox_args.top_k == 0 and neox_args.top_p == 0.0:
                    generated_tokens = torch.argmax(generated_token_logits, dim=-1).view(-1)
                else:
                    generated_token_logits = generated_token_logits.float()
                    
                    if neox_args.temperature > 0.0:
                        generated_token_logits /= neox_args.temperature
                    
                    generated_token_logits = filter_logits(generated_token_logits, top_k=neox_args.top_k, top_p=neox_args.top_p)
                    next_token_log_probs = F.softmax(generated_token_logits, dim=-1)
                    generated_tokens = torch.multinomial(next_token_log_probs, num_samples=1).view(-1)
            
            else:
                generated_tokens = torch.zeros(batch_size, dtype=torch.long).cuda()

            
            if neox_args.is_pipe_parallel:
                # broadcast generated tokens to pipe parallel group
                src_rank = model.grid.stage_to_global(model.num_stages - 1)
                torch.distributed.broadcast(
                    tensor=generated_tokens,
                    src=src_rank,
                    group=mpu.get_pipe_parallel_group(),
                )
            
    

            # determine if state has started for each batch item
            state_started = token_generation_start_index <= i # check which batch items have been started

            # switch out padding tokens for generated tokens
            context_tokens[:, i] = switch(
                context_tokens[:, i].view(-1),
                generated_tokens,
                state_started,
            )


            # determine if state has finished for each batch item
            state_done = (
                generated_tokens == neox_args.tokenizer.eod
            ).byte() & state_started.byte()  # check which batch items produce an eos_token in the current iteration
            state_is_done = state_is_done | state_done
            stop_tokens_produced = torch.zeros_like(state_is_done)
            for batch_idx in range(len(context_tokens)):
                stop_tokens_produced[batch_idx] = stop_tokens_in_completion(
                    stop_tokens, context_tokens, batch_idx, i
                )
            state_is_done = state_is_done | stop_tokens_produced
            
            
            
            if generated_token_logits.device == torch.device(0):
                gt = generated_tokens.tolist()
                # print("logits:", logits, logits.size())
                # print("generated_logits:", generated_token_logits, generated_token_logits.size())
                # print("generated_tokens:", gt)
                print(neox_args.tokenizer.detokenize(gt), end="")
                
            
            if torch.all(state_is_done):
                break

            yield logits# , generated_token_logits, generated_tokens     


def main():

    model, neox_args = setup_for_inference_or_eval(use_cache=True)
    if neox_args.recompute:
        model.module.inference_mode(use_cache=False)  # don't use kv cache if recomputing


    samples = list(dir2samples("../data"))[:10]
    questions, answers = samples2inputs(samples)

    # TODO maybe only compare the generated embeddings from the question 
    #  to the not generated embedding from the answers?

    question_tokens = input2tokens(questions, neox_args.tokenizer)
    with open("question_embeddings.jsonl", "wt") as file:
        for sample_id, context_tokens in question_tokens:
            for embedding in tokens2embeddings(context_tokens, model, neox_args):
                # we only want the last embedding
                pass 
            json_line = json.dumps({"id": sample_id, "embedding": embedding}) + "\n"
            file.write(json_line)

    answer_tokens = input2tokens(answers, neox_args.tokenizer)
    with open("answer_embedding.jsonl", "wt") as file:
        for sample_id, context_tokens in answer_tokens:
            for embedding in tokens2embeddings(context_tokens, model, neox_args):
                # we only want the first embedding
                break
            json_line = json.dumps({"id": sample_id, "embedding": embedding}) + "\n"
            file.write(json_line)
    
if __name__ == "__main__":
    main()