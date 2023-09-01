import os
import json
import dataset

commonsense = dataset.load_commonsense()

def make_query(q):
    ret = "Problem:\n"
    ret += q["question"]["stem"]
    ret += "\n"
    for choice in q["question"]["choices"]:
        ret += f'{choice["label"]}. {choice["text"]}\n'
    return ret

for f in os.listdir("csqaout"):
    prefix = f.split("_")[0]
    s = ""
    with open(f"csqaout/{f}", "r") as file:
        for i, line in enumerate(file):
            try:
                d = json.loads(line)
                s += d["picked"]
            except:
                pass
    with open(f"csqatext/{prefix}.txt", "w") as file:
        file.write(make_query(commonsense[int(prefix)]) + "\n" + s)
        
        