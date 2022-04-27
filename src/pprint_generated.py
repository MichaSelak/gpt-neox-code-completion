import json
import re

path = "result.jsonl"
result = "result.txt"


def unescape_newline(string:str):
    return re.sub("\\\\n", "\n", string)    

def unescape_tab(string:str):
    return re.sub("\\\\t", "\t", string)

with open(path, "rt", newline="") as in_file, \
        open(result, "wt") as out_file:
    for i, dic in enumerate((json.loads(json_line) for json_line in in_file)):
        out_file.write(f"\n=========================  Doc {i + 1}  =========================\n")
        out_file.write(unescape_newline(dic["context"]))
        out_file.write("\n=========================generated:=========================\n")
        out_file.write(unescape_tab(unescape_newline(dic["text"])))
        # out_file.write(dic["text"])
