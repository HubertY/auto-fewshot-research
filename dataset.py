from copyreg import constructor
import json


class Dataset:
    def append(self, x):
        self.data.append(x)

    def __init__(self):
        self.data = []


def load_boolq(path="boolq.jsonl"):
    ret = Dataset()
    with open(path, mode="r") as file:
        for i, line in enumerate(file):
            try:
                d = json.loads(line)
                ret.append(d)
            except:
                pass
    return ret

def load_commonsense(path="commonsenseqa.jsonl"):
    ret = []
    with open(path, mode="r") as file:
        for i, line in enumerate(file):
            try:
                d = json.loads(line)
                ret.append(d)
            except:
                pass
    file.close()
    return ret

def load_dataset(path="commonsenseqa.jsonl"):
    ret = []
    with open(path, mode="r") as file:
        for i, line in enumerate(file):
            try:
                d = json.loads(line)
                ret.append(d)
            except:
                pass
    file.close()
    return ret
