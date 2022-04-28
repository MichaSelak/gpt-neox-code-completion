from ast import Yield
import csv
from pathlib import Path

def dir2samples(path:str):
    p = Path(path)
    for file in p.glob("?.csv"):
        with file.open("rt", newline="") as fd:
            reader = csv.reader(fd)
            yield from reader

# print(next(dir2samples("../data")))
