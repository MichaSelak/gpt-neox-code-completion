import json

def main():
    
    with open("questions_embeddings.jsonl", "rt") as question_file, \
         open("answers_embeddings.jsonl", "rt") as answer_file:
         for question_line in question_file.readlines():
             question_dic = json.loads(question_line)
