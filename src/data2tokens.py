from csv import reader


import sys
import os
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
)
from megatron.tokenizer.tokenizer import HFTokenizer

# TODO encoding erros can be fixed with import ftfy text = ftfy.fix_text(text)
path = "../weights/slim/20B_checkpoints/20B_tokenizer.json"

tokenizer = HFTokenizer(path)


sample_path = "data/3.csv"
out_file = "token.jsonl"

with open(sample_path, "rt", newline="") as in_file:
    for i, line in enumerate(reader(in_file)):
        if i == 10:
            break
        question_code = line[2]
        if not question_code:
            continue
        # TODO: i think the text gets stripped etc. before look it up
        print(line)
        print("as tokens:")
        print(tokenizer.tokenize(question_code))





















