import dataset
import bloomclient
import asyncio
import json
import torch
import time
import os

commonsense = dataset.load_commonsense()

def is_stop_token(token):
    return token is not None and token[0].item() == 2

def make_query(q):
    ret = ""
    ret += q["question"]["stem"]
    ret += "\n"
    for choice in q["question"]["choices"]:
        ret += f'{choice["label"]}. {choice["text"]}\n'
    return (ret, q["answerKey"])

def sample_token(tokens, probs, temp = 1):
    index = torch.multinomial(probs ** (1/temp), num_samples=1).squeeze(1)[0].item()
    return tokens[:, index]

def extend_input(input, tokens):
    return torch.cat([input] + [torch.tensor([token], dtype=torch.int).unsqueeze(0) for token in tokens if token is not None], dim=-1)

def make_token_entry(_tokens, _probs, selection, attn, N=20):
    attn = attn.to_list() if attn is not None else []
    token_ids = []
    probs = []
    for i in range(N):
        probs.append(_probs[:,i][0].item())
        token_ids.append(_tokens[:,i][0].item())

    return {"text": bloomclient.detokenize(selection), "picked": selection[0].item(), "ids": token_ids, "probs": probs, "attentions": attn}


FOX_QUERY = """Question:
The fox walked from the city into the forest, what was it looking for?
A. pretty flowers
B. hen house
C. natural habitat
D. storybook
E. dense forest
Answer:
"""



PREFIX = "I am an intelligent problem solving AI. When I encounter a problem I first explain my thought process, then I give a final answer.\n"
SUFFIX = "Thought Process:\n"


async def validate(cli, s):
    await cli.send("ADD", bloomclient.tokenize(s))
    await cli.send("FORWARD")
    tokens = await cli.send("TOKENS")
    probs = await cli.send("PROBS")
    tokenranks = []
    aprobs = {"A": 0, "B": 0, "C": 0, "D": 0, "E": 0}
    for j in range(20):
        s = bloomclient.detokenize(tokens[:, j])
        tokenranks.append(s)
        s = s.strip().upper()
        if s == "A" or s == "B" or s == "C" or s == "D" or s == "E":
            aprobs[s] += probs[0][j].item()

    ret = {"probs": aprobs, "tokens": tokenranks[:10]}
    print(ret)
    return ret

import math
async def validate_job(cli, inpath, outpath):
    c = Completion.load(inpath)
    probs = [token["probs"][token["ids"].index(token["picked"])] for token in c.tokens]
    logprobs = [math.log(p) for p in probs]
    modified = [token["picked"] == token["ids"][0] for token in c.tokens]
    probs2 = [p for (p, b) in zip(probs, modified) if b]
    logprobs2 = [math.log(p) for p in probs2]
    totallog = sum(logprobs)
    totallog2 = sum(logprobs2)
    ppl = math.exp(totallog/len(logprobs))
    ppl2 = math.exp(totallog2/len(logprobs2))
    s = (c.s + c.export()).strip() + "\nFinal Answer:\n"
    ansprobs = (await validate(cli, s))["probs"]
    correct = 1
    for choice in ansprobs:
        if choice != c.answer and ansprobs[choice] >= ansprobs[c.answer]:
            correct = 0
    pcorr = ansprobs[c.answer]
    pcorrnorm = pcorr / sum(ansprobs.values())
    r = {
        "output": s, "answer": c.answer, "probs": ansprobs,
        "score": pcorr, "scorenorm": pcorrnorm, "correct": correct,
        "totallog": totallog, "totallog2": totallog2, "ppl": ppl, "ppl2": ppl2
        }
    with open_file(outpath, "w") as f:
        f.write(json.dumps(r))

async def metavalidate(cli, exp, s, PREFIX, SUFFIX, meta = False):
    s = s.split("Problem:")[0].strip()
    s = s.split("Final Answer:")[0].strip()
    items = {}
    if meta:
        for index in range(len(s)):
            if (s[index] == "."):
                items[index] = await validate(cli, PREFIX + s[:index+1] + SUFFIX)
    items[len(s)] = await validate(cli, PREFIX + s + SUFFIX)
    bestscore = -1
    bestindex = -1
    for index in items.keys():
        score = items[index]["probs"][exp]
        if score > bestscore:
            bestscore = score
            bestindex = index
    return {"output": PREFIX + s[:bestindex+1] + SUFFIX, "answer": exp, "score": bestscore, "probs": items[bestindex]}

CODE = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"

class CompletionNode:
    def unwalk(self):
        node = self
        while node is not None:
            yield node
            node = node.parent
    def get_ancestors(self):
        ret = list(self.unwalk())
        ret.reverse()
        return ret
    def write(self, path):
        s = self.master.s
        for node in self.get_ancestors():
            s += node.text
        with open_file(path, "w") as f:
            f.write(s)
    async def greed(self, cli, N=50):
        if not self in self.master.frontier:
            raise Error("not in master frontier, run reindex?")
        self.master.frontier.discard(self)
        await cli.send("ADD", extend_input(self.master.input, [token.picked for token in self.get_ancestors() if token.picked is not None]))
        node = self        
        k = 0

        while not node.terminal and k < N:
            k += 1
            if node.picked is not None:
                await cli.send("APPEND", torch.tensor(node.picked).unsqueeze(-1))
                print(bloomclient.detokenize(torch.tensor(node.picked).unsqueeze(-1)))
            if not node.computed:
                await cli.send("FORWARD")
                node.next_tokens = (await cli.send("TOKENS")).squeeze(-1)[0].tolist()
                node.next_probs = (await cli.send("PROBS")).squeeze(-1)[0].tolist()
                node.computed = True
            for i, (token, prob) in enumerate(zip(node.next_tokens, node.next_probs)):
                next = CompletionNode(self.master, node,token, prob)
                next.code = node.code + CODE[i]
                if next.text.strip().endswith("."):
                    next.terminal = True
                    self.master.terminals.add(next)
                else:
                    self.master.frontier.add(next)
                node.children[token] = next
            node = node.children[node.next_tokens[0]]
            self.master.frontier.discard(node)
        return node, k
    def __init__(self, master, parent, picked, prob):
        self.master = master
        self.parent = parent
        self.picked = picked
        self.text = bloomclient.detokenize(self.picked) if self.picked else "" 
        self.prob = prob
        self.cumlogprob = math.log(self.prob) + (self.parent.cumlogprob if self.parent else 0)
        self.computed = False
        self.terminal = False
        self.next_tokens = []
        self.next_probs = []
        self.data = {}
        self.children = {}
        self.code = ""

class CompletionTree:
    def __init__(self, id, prefix, suffix):
        self.id = id
        self.prefix = prefix
        self.suffix = suffix
        q, a = make_query(commonsense[self.id])
        self.s = prefix + q + suffix
        self.input = bloomclient.tokenize(self.s)
        self.answer = a
        self.head = CompletionNode(self, None, None, 1)
        self.terminals = set()
        self.frontier = set()
        self.frontier.add(self.head)

    def jobs(self, n, decay = 50):
        front = list(self.frontier)
        front.sort(reverse=True, key=lambda node: node.cumlogprob )
        # front.sort(reverse=True, key=lambda node: node.prob * (0.5 ** (len(node.code)/decay)) )
        def jobbify(node):
            return lambda cli: node.greed(cli, 50)
        return [jobbify(node) for node in front[:n]]


async def validate_treenode(self, cli, outpath):
    ancest = self.get_ancestors()
    ans = self.master.answer

    s = self.master.s
    for node in ancest:
        s += node.text
    s = s.strip()
    s += "\nFinal Answer:\n"
    
    ansprobs = (await validate(cli, s))["probs"]
    correct = 1
    for choice in ansprobs:
        if choice != ans and ansprobs[choice] >= ansprobs[ans]:
            correct = 0
    pcorr = ansprobs[ans]
    pcorrnorm = pcorr / sum(ansprobs.values())

    probs = [node.prob for node in ancest if node.picked is not None]
    probs2 = [node.prob for node in ancest if node.picked is not None and node.code[-1]=="0"]
    probs.append(ansprobs[ans])
    probs2.append(ansprobs[ans])
    logprobs = [math.log(p) for p in probs]
    logprobs2 = [math.log(p) for p in probs2]
    totallog = sum(logprobs)
    totallog2 = sum(logprobs2)
    ppl = math.exp(totallog/len(logprobs))
    ppl2 = math.exp(totallog2/len(logprobs2))

    r = {
        "output": s, "answer": ans, "probs": ansprobs,
        "score": pcorr, "scorenorm": pcorrnorm, "correct": correct,
        "totallog": totallog, "totallog2": totallog2, "ppl": ppl, "ppl2": ppl2
        }
    with open_file(outpath, "w") as f:
        f.write(json.dumps(r))

def validate_tree_jobs(tree, outdir):
    def jobbify(node):
        return lambda cli: validate_treenode(node, cli, f"{outdir}/{node.code}.jsonl")
    return [jobbify(node) for node in tree.terminals if node.code[-1] == "0"]


class Completion:
    id: int
    prefix: str
    suffix: str
    tokens: list
    def __init__(self, id, prefix: str, suffix: str):
        self.id = id
        self.prefix = prefix
        self.suffix = suffix
        self.tokens = []
        q, a = make_query(commonsense[self.id])
        self.s = prefix + q + suffix
        self.input = bloomclient.tokenize(self.s)
        self.answer = a
    
    def get_ans(self):
        ret = self.input[0].tolist()
        rettext = ["." for r in ret]
        s = ""
        attentions = None
        for token in self.tokens:
            ret.append(token)
            s = s + token["text"]
            rettext.append(token["text"])
            if s.endswith("Final Answer:\n"):
                attentions = token["attentions"]
                break
        return ret, rettext, attentions



    def write(self, path):
        with open_file(path, "w") as f:
            f.write(f'{json.dumps({"id": self.id, "prefix": self.prefix, "suffix": self.suffix, "s": self.s})}\n')
            for token in self.tokens:
                f.write(f'{json.dumps(token)}\n')

    @staticmethod
    def load(path):
        meta = None
        tokens = []
        with open_file(path, "r") as f:
            for i, line in enumerate(f):
                if i == 0:
                    meta = json.loads(line)
                else:
                    try:
                        token = json.loads(line)
                        if token:
                            tokens.append(token)
                    except:
                        pass
        ret = Completion(meta["id"], meta["prefix"], meta["suffix"])
        ret.tokens = tokens
        return ret

    def branch(self, n, sel):
        if len(self.tokens) <= n:
            raise Error("not enough tokens") 
        ret = Completion(self.id, self.prefix, self.suffix)
        for i in range(n+1):
            ret.tokens.append(self.tokens[i].copy())
        ret.tokens[-1]["picked"] = ret.tokens[-1]["ids"][sel]
        ret.tokens[-1]["text"] = bloomclient.detokenize(ret.tokens[-1]["picked"])
        return ret

    def export(self):
        s = ""
        for token in self.tokens:
            k = bloomclient.detokenize(torch.tensor(token["picked"], dtype = torch.int).unsqueeze(0))
            if k == "</s>":
                break
            s += k
        return s

    async def infer(self, cli, path, temp, N):
        self.write(path)
        await cli.send("ADD", extend_input(self.input, [token["picked"] for token in self.tokens]))
        selection = None        
        k = 0
        while not is_stop_token(selection) and k < N:
            k += 1
            if selection is not None:
                await cli.send("APPEND", selection)
                print(bloomclient.detokenize(selection))
            await cli.send("FORWARD")
            tokens = await cli.send("TOKENS")
            probs = await cli.send("PROBS")
            selection = sample_token(tokens, probs, temp)
            attn = await cli.send("ATTENTIONS")
            entry = make_token_entry(tokens, probs, selection, None)
            tokens.append(entry)
            with open_file(path, "a") as f:
                f.write(f'{json.dumps(entry)}\n')

    async def greed(self, cli, path, N):
        self.write(path)
        await cli.send("ADD", extend_input(self.input, [token["picked"] for token in self.tokens]))
        selection = None        
        k = 0
        while not is_stop_token(selection) and k < N:
            k += 1
            if selection is not None:
                await cli.send("APPEND", selection)
                print(bloomclient.detokenize(selection))
            await cli.send("FORWARD")
            tokens = await cli.send("TOKENS")
            probs = await cli.send("PROBS")
            selection = tokens[:, 0]
            attn = None
            entry = make_token_entry(tokens, probs, selection, None)
            self.tokens.append(entry)
            with open_file(path, "a") as f:
                f.write(f'{json.dumps(entry)}\n')
            if entry["text"].strip().endswith("."):
                break

    async def interactive(self, cli, path, temp):
        self.write(path)
        await cli.send("ADD", extend_input(self.input, [token["picked"] for token in self.tokens]))
        selection = None        
        k = 0
        print(self.input)
        while True:
            k += 1
            if selection is not None:
                await cli.send("APPEND", selection)
            await cli.send("FORWARD")
            tokens = await cli.send("TOKENS")
            probs = await cli.send("PROBS")
            #print(self.export())
            for token, prob in zip(tokens[0].tolist(), probs[0].tolist()):
                print(bloomclient.detokenize(token), token, prob)
            selection = int(input(">"))
            if selection == -1:
                break
            selection = torch.tensor(selection).unsqueeze(-1)
            attn = await cli.send("ATTENTIONS")
            entry = make_token_entry(tokens, probs, selection, None)
            with open_file(path, "a") as f:
                f.write(f'{json.dumps(entry)}\n')

def extract_last_shot(s):
    idx = s.rfind("Problem:\n")
    if idx != -1:
        return s[idx:]
    else:
        return ""

async def val(cli, args):
    path, writepath, meta = args
    comp = Completion.load(path)
    q, a = make_query(commonsense[comp.id])
    s = comp.export()
    res = await metavalidate(cli, a, s, comp.prefix + q + comp.suffix, "\nFinal Answer:\n", meta)
    with open(writepath, "w") as file:
        file.write(json.dumps(res))


async def inferenceworker(items: list, port):
    cli = bloomclient.BloomClient()
    await cli.connect(port)
    while len(items):
        comp, path, temp, n = items.pop()
        x = str(len(items))
        await comp.infer(cli, path, temp, n)
        with open("num.txt", "a") as f:
            f.write(f"{x}\n")
    await cli.send("CLOSE")
    cli.close()


async def validateworker(items: list, port):
    cli = bloomclient.BloomClient()
    await cli.connect(port)
    while len(items):
        item = items.pop()
        await val(cli, item)
    await cli.send("CLOSE")
    cli.close()


# async def main2(folders):
#     for folder in folders:
#         os.mkdir(f"{folder}validate")
#     all = [[(folder, path) for path in os.listdir(folder)] for folder in folders]
#     items = []
#     for item in all:
#         items.extend(item)
#     workers = []
#     for i in range(8):
#         for j in range(40):
#             workers.append(validateworker(items, 1330+i))
#     await asyncio.gather(*workers)

def readjson(path):
    with open(path, "r") as f:
        return json.load(f)



# async def main4(folders):
#     items = []
#     for folder in folders:
#         num = folder.split("/")[1].split("validate")[0]
#         os.mkdir(f"csqaiter3/{num}0")
#         os.mkdir(f"csqaiter3/{num}1")
#         doclist = [readjson(f"{folder}/{path}") for path in os.listdir(folder)]
#         doclist.sort(key=lambda item: item["score"], reverse=True)
#         top2 = doclist[:2]

#         for i in range(2):
#             for j in range(0,80):
#                 pre = top2[i]["output"] + top2[i]["answer"] + "\n"
#                 items.append((2500+j, f"csqaiter3/{num}{i}", pre, "Thought Process:\n", 0.7))
                
#     workers = []
#     for j in range(40):
#         for i in range(8):
#             workers.append(inferenceworker(items,1330+i))
#     await asyncio.gather(*workers)

# async def main5():
#     with open("humanexample2.txt") as file:
#         pre = file.read()
#     workers = []
#     for i in range(8):
#         for j in range(0,10):
#             workers.append(inferenceworker([(2500+i*10+j, f"human2", pre, "Thought Process:\n", 0.7)], 1330+i))
#     await asyncio.gather(*workers)
    
async def check_accuracy(folders):
    for folder in folders:
        doclist = [readjson(f"{folder}/{path}") for path in os.listdir(folder)]
        doclist.sort(key=lambda item: item["score"], reverse=True)
        p = 0
        phat = 0
        c = 0
        for doc in doclist:
            probsum = 0
            bestprob = -1
            bestkey = None
            for key in doc["probs"]["probs"]:
                prob = doc["probs"]["probs"][key]
                probsum += prob
                if prob > bestprob:
                    bestprob = prob
                    bestkey = key
            p += doc["score"]
            if doc["score"] > 0:
                phat += doc["score"] / probsum
            c += 1 if bestkey == doc["answer"] else 0
        n = len(doclist)
        print(folder, (p/n, phat/n, c/n))
        #print(doclist[0]["output"])

def check_accuracy2(folders, i = None):
    print(folders)
    ret = {}
    for folder in folders:
        doclist = [readjson(f"{folder}/{path}") for path in os.listdir(folder)]
        if i is not None:
            doclist = doclist[:i]
        doclist.sort(key=lambda item: item["score"], reverse=True)
        p = 0
        phat = 0
        c = 0
        for doc in doclist:
            p += doc["score"]
            phat += doc["scorenorm"]
            c += doc["correct"]
        n = len(doclist)
        print(folder, (p/n, phat/n, c/n))
        ret[folder.split("/", 1)[-1]] = (p/n, phat/n, c/n)
    return ret

def check_correctness(folders, i = None):
    print(folders)
    ret = {}
    for folder in folders:
        doclist = [readjson(f"{folder}/{path}") for path in os.listdir(folder)]
        if i is not None:
            doclist = doclist[:i]
        doclist.sort(key=lambda item: item["score"], reverse=True)
        p = 0
        phat = 0
        c = 0
        for doc in doclist:
            p += doc["score"]
            phat += doc["scorenorm"]
            c += doc["correct"]
        n = len(doclist)
        print(f"{folder.split('/', 1)[-1]},{c}")
        ret[folder.split("/", 1)[-1]] = (p/n, phat/n, c/n)
    return ret

import pandas as pd
def check_accuracy3(folders):
    print(folders)
    data = {"name": [], "problem": [], "scorenorm": [], "ppl": [], "correct": []}
    for folder in folders:
        for path in os.listdir(folder):
            doc = readjson(f"{folder}/{path}")
            data["name"].append(folder.split("/")[-1])
            data["problem"].append(path.split(".")[0])
            data["scorenorm"].append(doc["scorenorm"])
            data["ppl"].append(doc["ppl"])
            data["correct"].append(doc["correct"])
    df = pd.DataFrame(data)
    df.to_csv("stats.csv")
    summ = df.groupby("name")[["scorenorm", "correct"]].mean().sort_values(["correct", "scorenorm"]).tail(8).reset_index()
    summ.to_csv("summary.csv")
    names = summ["name"]
    print(names)
    smol = df[df["name"].isin(names)]
    smol.sort_values(["name","ppl"]).to_csv("perplex.csv")
    smol.sort_values(["problem","name"]).to_csv("perf.csv")

def megaprompt(folders, orig):
    print(folders)
    data = {"name": [], "problem": [], "scorenorm": [], "ppl": [], "correct": []}
    for folder in folders:
        for path in os.listdir(folder):
            doc = readjson(f"{folder}/{path}")
            data["name"].append(folder.split("exhaust2iterval/")[1])
            data["problem"].append(path.split(".")[0])
            data["scorenorm"].append(doc["scorenorm"])
            data["ppl"].append(doc["ppl"])
            data["correct"].append(doc["correct"])
    df = pd.DataFrame(data)
    summ = df.groupby("name")[["scorenorm", "correct"]].mean().sort_values(["correct", "scorenorm"]).tail(8).reset_index()
    names = summ["name"]
    s = ""
    for name in names:
        doc = readjson(f"{orig}/{name}.jsonl")
        s += doc["output"]
        s += doc["answer"]
        s += "\n"
    return s


async def fox_test(path):
    comp = Completion(4, "Problem:\n", SUFFIX)
    await inferenceworker([(comp, f"{path}/master.jsonl", 1, 1)],1330)
    workers = []
    for i in range(8):
        comp = Completion.load(f"{path}/master.jsonl")
        comp.tokens[0]["picked"] = comp.tokens[0]["ids"][i]
        for j in range(0,50):
            workers.append(inferenceworker([(comp, f"{path}/{i}/{j}.jsonl", 0.7, 100)], 1330+i))
    await asyncio.gather(*workers)

async def infer_one():
    workers = []
    id = 0
    for i in range(8):
        for j in range(0,50):
            jobs = []
            for k in range(20):
                comp = Completion(id, "Problem:\n", "Thought Process:\n")
                jobs.append((comp, f"init/{id}.jsonl", 1, 1))
                id += 1
            workers.append(inferenceworker(jobs, 1330+i))
    await asyncio.gather(*workers)

async def validaterino(jobs):
    workers = []
    for j in range(0,50):
        for i in range(8):
            workers.append(validateworker(jobs, 1330+i))
    await asyncio.gather(*workers)

def list_tokens(folder):
    tokens = [Completion.load(f"{folder}/{path}").tokens[0] for path in os.listdir(folder)]
    probs = {}
    ids = []
    for token in tokens:
        for i, tokenid in enumerate(token["ids"]):
            if not tokenid in probs:
                probs[tokenid] = 0
                ids.append(tokenid)
            probs[tokenid] += token["probs"][i]
    ids.sort(key=lambda id: probs[id], reverse=True)
    return ids[:8]
    # for i in range(20):
    #     print(ids[i], bloomclient.detokenize(torch.tensor(ids[i], dtype = torch.int).unsqueeze(0)), probs[ids[i]])

async def itera(folders):
    indir, outdir = folders
    top8 = list_tokens(indir)
    workers = []
    for i in range(8):
        for j in range(0,100):
            jobs = []
            comp1 = Completion(2*j, "Problem:\n", "Thought Process:\n")
            comp1.tokens.append({"picked": top8[i]})
            comp2 = Completion(2*j+1, "Problem:\n", "Thought Process:\n")
            comp2.tokens.append({"picked": top8[i]})
            jobs.append((comp1, f"{outdir}/{i}/{comp1.id}.jsonl", 0.7, 150))
            jobs.append((comp2, f"{outdir}/{i}/{comp2.id}.jsonl", 0.7, 150))
            workers.append(inferenceworker(jobs,1330+i))
    await asyncio.gather(*workers)


def print_best(indir, metric = "score"):
    doclist = [readjson(f"{indir}/{path}") for path in os.listdir(indir)]
    doclist = [doc for doc in doclist if doc["correct"] == 1]
    if(len(doclist) != 0):
        doclist.sort(key=lambda item: item[metric], reverse=True)
        top8 = doclist
        print(top8[0]["output"], top8[0][metric])
    # for item in top8:
    #     print (item["output"])

def print_all(indir, f=None):
    doclist = [readjson(f"{indir}/{path}") for path in os.listdir(indir)]
    doclist.sort(key=lambda item: item["score"], reverse=True)
    if f is not None:
        with open(f, "w") as ff:
            for item in doclist:
                ff.write(f"{item['output']}\n\n")
    else:
        for item in doclist:
            print (item["output"])

async def attentest(path):
    workers = []
    for i in range(8):
        comp = Completion(i, "Problem:\n", SUFFIX)
        workers.append(inferenceworker([(comp, f"{path}/{i}/master.jsonl", 1, 1)],1330+i))
    await asyncio.gather(*workers)
    workers = []
    for i in range(8):
        for j in range(0,10):
            comp = Completion.load(f"{path}/{i}/master.jsonl")
            comp.tokens[0]["picked"] = comp.tokens[0]["ids"][j]
            workers.append(inferenceworker([(comp, f"{path}/{i}/{j}.jsonl", 0.7, 100)], 1330+i))
    await asyncio.gather(*workers)


async def inferrino(path1, path2):
    workers = []
    for i in range(8):
        jobs = []
        for id in range(0,200):
            item = readjson(f"{path1}/{i}/{id}.jsonl")
            prefix = item["output"] + item["answer"] + "\nProblem:\n"
            comp = Completion(id+200, prefix, "Thought Process:\n")
            # print(prefix)
            jobs.append((comp, f"{path2}/{i}/{id}.jsonl", 0.7, 50))

        for j in range(0, 200, 4):
            workers.append(inferenceworker(jobs[j:j+4], 1330+i))
    await asyncio.gather(*workers)

# jobs = []
# for i in range(8):
#     indir = f"prime/{i}"
#     outdir = f"primevalidate/{i}"
#     for path in os.listdir(indir):
#         jobs.append((f"{indir}/{path}", f"{outdir}/{path}", False))
# for i in range(8):
#     indir = f"prime/{i}"
#     outdir = f"primemetavalidate/{i}"
#     for path in os.listdir(indir):
#         jobs.append((f"{indir}/{path}", f"{outdir}/{path}", True))

#asyncio.run(validaterino(jobs))
# asyncio.run(check_accuracy([f"{folder}/{i}" for i in range(8) for folder in ["primemetavalidate"]]))
# for i in range(8):
#     print_best(f"primemetavalidate/{i}")

# asyncio.run(check_accuracy([f"{folder}/{i}" for i in range(8) for folder in ["primeitervalidate"]]))
# for i in range(8):
#     print_best(f"primeitervalidate/{i}")

# print_all(f"primeitervalidate/7", "hallo.txt")

def iteration_jobs(folders, qstart=400, top=50, rep=2, n=50):
    jobs = []
    indir, outdir = folders
    doclist = [(readjson(f"{indir}/{path}"), path) for path in os.listdir(indir)]
    doclist.sort(key=lambda item: item[0]["score"], reverse=True)
    doclist = doclist[:top]
    for doc in doclist:
        prefix, pid = doc
        pid=int(pid.split(".")[0])
        for r in range(rep):
            comp = Completion(qstart, extract_last_shot(prefix["output"]) + prefix["answer"] + "\nProblem:\n", "Thought Process:\n")
            jobs.append((comp, f"{outdir}/{str(pid+200)}-{str(qstart)}.jsonl", 0.7, n))
            qstart+=1
    return jobs

def extend_jobs(folders):
    jobs = []
    for indir in folders:
        for path in os.listdir(indir):
            comp = Completion.load(f"{indir}/{path}")
            s = comp.export()
            if not "Final Answer:" in s and not "Question:" in s:
                jobs.append((comp, f"{indir}/{path}", 0.7, 50))
    return jobs

async def execute():
    await asyncio.gather(*workers)

def collectstats(path):
    comp = Completion.load(path)
    ret, rettext, attentions = comp.get_ans()
    for text, att in zip(rettext, attentions):
        print(text, att)
    print(len(rettext), len(attentions), sum(attentions))
    print(comp.tokens[0]["attentions"])

def collectstats_folder(path):
    s = f"{path},"
    for f in os.listdir(path):
        json = readjson(f"{path}/{f}")
        

# collectstats("interactive.txt")

async def foxx(path):
    cli = bloomclient.BloomClient()
    await cli.connect(1330)
    comp = Completion(113, "Problem:\n", SUFFIX)
    # comp = Completion.load("interactive2.txt")
    await comp.interactive(cli, path, 5)

def new_future():
    return asyncio.get_event_loop().create_future()

async def worker(jobs, port):
    cli = bloomclient.BloomClient()
    await cli.connect(port)
    while len(jobs):
        await jobs.pop()(cli)
    await cli.send("CLOSE")
    cli.close()


def recurse(comp, path, jobs):
    s = ""
    for i in range(len(comp.tokens)-1):
        agg = 0
        for j in range(10):
            ss = f"{s}{j}.jsonl"
            if j > 0 and not os.path.exists(f"{path}/{ss}.jsonl"):
                print(f"{path}/{ss}.jsonl")
                def lmao(s, ss, c):
                    return lambda cli: c.greed(cli, f"{s}/{ss}.jsonl", 50)
                jobs.append(lmao(path, ss, comp.branch(i,j)))
            agg += comp.tokens[i]["probs"][j]
            if agg > 0.85:
                break
        s = s + "0"


async def exh():
    jobs = []
    # for i in range(8):
    #     comp = Completion(i, "Problem:\n", "Thought Process:\nSince")
    #     jobs.append(lambda cli: comp.greed(cli, f"exhaust2/{i}/0.jsonl", 50))

    for i in range(8):
        comp = Completion.load(f"exhaust2/{i}/0.jsonl")
        recurse(comp, f"exhaust2/{i}", jobs)

    
    workers = []
    for j in range(50):
        for i in range(8):
            workers.append(worker(jobs, 1330+i))
    await asyncio.gather(*workers)

from util import open_file

def file_exists(filename):
    return os.path.isfile(filename)

def log(s, fname = "log.log"):
    with open_file(fname, "a") as f:
        f.write(f"{str(s)}\n")

def val_jobs(indir, outdir, jobs=None):
    if jobs is None:
        jobs = []
    for path in os.listdir(indir):
        if path.endswith(".jsonl"):
            if not file_exists(f"{outdir}/{path}"):
                def f(path):
                    jobs.append(lambda cli: validate_job(cli, f"{indir}/{path}", f"{outdir}/{path}"))
                f(path)
        else:
            val_jobs(f"{indir}/{path}", f"{outdir}/{path}", jobs)
    log(f"val_jobs: {len(jobs)} jobs")
    return jobs

def optimal_frontier(indir):
    
    doclist = [(readjson(f"{indir}/{path}"), path) for path in os.listdir(indir)]
    doclist = [doc for doc in doclist if doc[0]["correct"] == 1]
    frontier = []
    for (doc, path) in doclist:
        ok = True
        for (front, p) in frontier:
            if front["scorenorm"] > doc["scorenorm"] and front["ppl2"] > doc["ppl2"]:
                ok = False
                break
        if ok:
            frontier.append((doc, path))
    return frontier

def correctness(indir):
    doclist = [(readjson(f"{indir}/{path}"), path) for path in os.listdir(indir)]
    doclist = [doc for doc in doclist if doc[0]["correct"] == 1]
    return doclist

def write_stats(indir,path):
    doclist = optimal_frontier(indir)
    with open(path, "w") as f:
        for (doc, p) in doclist:
            o = doc['output'].split("Thought Process:\n")[1].split("\nFinal Answer")[0].replace('\"','')
            f.write(f"\"{o}\",{doc['scorenorm']},{doc['totallog2']},{doc['ppl2']}\n")

def superiter(indir, outdir, items = range(100,108)):
    jobs = []
    for i in range(8):
        docs = optimal_frontier(f"{indir}/{i}")
        for (doc, path) in docs:
            folder = f"{outdir}/{i}/{path.split('.')[0]}"
            for j in items:
                comp = Completion(j, doc['output']+doc['answer']+"\nProblem:\n", "Thought Process:\nSince")
                def lmao(p, c):
                    return lambda cli: c.greed(cli, p, 50)
                jobs.append(lmao(f"{folder}/{j}.jsonl", comp))
    return jobs

async def go(jobs, nworkers = 40, nservers = 8):
    print(len(jobs))
    workers = []
    for j in range(nworkers):
        for i in range(nservers):
            workers.append(worker(jobs, 1330+i))
    await asyncio.gather(*workers)

def recurse_find_folders(path, ret = None):
    if ret is None:
        ret = []
    sub = [p for p in os.listdir(path) if not p.endswith(".jsonl")]
    if len(sub):
        for f in sub:
            recurse_find_folders(f"{path}/{f}", ret)
    else:
        ret.append(path)
    return ret

def inference_jobs(ids, prefix, suffix, path):
    jobs = []
    for i in ids:
        comp = Completion(i, prefix, suffix)
        def fun(c, path):
            return lambda cli: c.greed(cli, path, 50)
        jobs.append(fun(comp, f"{path}/{i}.jsonl"))
    return jobs

threeshot = """Problem:
The forgotten leftovers had gotten quite old, he found it covered in mold in the back of his what?
A. carpet
B. refrigerator
C. breadbox
D. fridge
E. coach
Thought Process:
Since the leftovers were quite old, it was covered in mold, and it was in the back of his refrigerator, I think the answer is B.
Final Answer:
B
Problem:
To locate a choker not located in a jewelry box or boutique where would you go?
A. jewelry store
B. neck
C. jewlery box
D. jewelry box
E. boutique
Thought Process:
Since the choker is not located in a jewelry box or boutique. I would go to a jewelry store.
Final Answer:
A
Problem:
What home entertainment equipment requires cable?
A. radio shack
B. substation
C. cabinet
D. television
E. desk
Thought Process:
Since the question is asking for the equipment that requires cable, I thought that the answer would be the television.
Final Answer:
D
"""

threeshot2 = """Problem:
To locate a choker not located in a jewelry box or boutique where would you go?
A. jewelry store
B. neck
C. jewlery box
D. jewelry box
E. boutique
Thought Process:
Since it clearly states that the choker is not located in a jewelry box or boutique, I would go to a jewelry store.
Final Answer:
A
Problem:
Sammy wanted to go to where the people were.  Where might he go?
A. race track
B. populated areas
C. the desert
D. apartment
E. roadblock
Thought Process:
Since Sammy wanted to see the people, he would have to go to a populated area.
Final Answer:
B
Problem:
What home entertainment equipment requires cable?
A. radio shack
B. substation
C. cabinet
D. television
E. desk
Thought Process:
Since the only equipment that requires cable is a television, I chose D.
Final Answer:
D
"""

threeshothuman = """Problem:
The forgotten leftovers had gotten quite old, he found it covered in mold in the back of his what?
A. carpet
B. refrigerator
C. breadbox
D. fridge
E. coach
Thought Process:
Since the leftovers had gotten quite old, he found it covered in mold, the answer is in the back of his refrigerator.
Final Answer:
B
Problem:
To locate a choker not located in a jewelry box or boutique where would you go?
A. jewelry store
B. neck
C. jewlery box
D. jewelry box
E. boutique
Thought Process:
Since the choker is not located in a jewelry box or boutique, the answer is I would go to a jewelry store.
Final Answer:
A
Problem:
What home entertainment equipment requires cable?
A. radio shack
B. substation
C. cabinet
D. television
E. desk
Thought Process:
Since the home entertainment system requires cable, the answer is television.
Final Answer:
D
"""

        

# print(megaprompt(recurse_find_folders("exhaust2iterval"), "exhaust2val"))
# for i in range(8):
#     write(f"exhaust2val/{i}", f"stats{i}.csv")

#asyncio.run(go(superiter("exhaust2val", "exhaust2iter")))


import pickle
async def treetest(server, problem, path, suffix, N=10, token_quota=1000,):
    clis = []
    for k in range(1):
        cli = bloomclient.BloomClient()
        await cli.connect(1330+server)
        clis.append(cli)

    tree = CompletionTree(problem, "Problem:\n", suffix)
    log(f"{server} starting initial greed")
    leaf, consumed = await tree.head.greed(clis[0], 50)
    leaf.write(f"{path}/{leaf.code}.txt")
    token_quota -= consumed
    log(f"{server} finished initial greed {leaf.code}, {token_quota} tokens remaining")

    for k in range(N-1):
        cli = bloomclient.BloomClient()
        await cli.connect(1330+server)
        clis.append(cli)

    jobs = tree.jobs(N)

    async def crunchjob(cli, jobs):
        nonlocal token_quota
        while token_quota > 0:
            leaf, consumed = await jobs.pop(0)(cli)
            token_quota -= consumed
            leaf.write(f"{path}/{leaf.code}.txt")
            log(f"{server} finished job {leaf.code}, {token_quota} tokens remaining")
            jobs.clear()
            for job in tree.jobs(N):
                jobs.append(job)
        await cli.send("CLOSE")
        cli.close()

    
    await asyncio.gather(*[crunchjob(cli, jobs) for cli in clis])

    with open_file(f"{path}/tree.bin", "wb") as f:
        pickle.dump(tree, f)

async def supertreetest(path, suffix):
    #await asyncio.gather(*[treetest(i, i, f"tree2/{i}", 5, 50) for i in range(1)])
    await asyncio.gather(*[treetest(i, i, f"{path}/{i}", suffix, 40, 10000) for i in range(8)])

def supertreevaljobs(dir):
    jobs = []
    for i in range(8):
        with open(f"{dir}/{i}/tree.bin", "rb") as f:
            tree = pickle.load(f)
        jobs = jobs + validate_tree_jobs(tree, f"{dir}val/{i}")
    return jobs

#asyncio.run(go(supertreevaljobs("tree2")))
#asyncio.run(foxx("interactive.txt"))



def iter_one(indir, path, outdir, items):
    jobs = []
    doc = readjson(f"{indir}/{path}")
    folder = f"{outdir}/{path.split('.')[0]}"
    for j in items:
        comp = Completion(j, doc['output']+doc['answer']+"\nProblem:\n", "Thought Process:\nSince")
        def lmao(p, c):
            return lambda cli: c.greed(cli, p, 50)
        jobs.append(lmao(f"{folder}/{j}.jsonl", comp))
    return jobs

async def gogogo():
    # await go(iter_one("tree2val/5", "08100000000000.jsonl", "tree2iter/5", range(116,200)))

    # # await go(superiter("tree2val", "tree2iter", range(108,116)))
    # log("finished tree2iter")
    # await go(val_jobs("tree2iter", "tree2iterval"))
    # log("finished tree2iterval")
    # await supertreetest("tree3", "Thought Process:\n")
    # log("finished tree3")
    # await go(supertreevaljobs("tree3"))
    # log("finished tree3val")
    #await go(inference_jobs(range(300, 350), threeshot2, "Thought Process:\n", "threeshot2"),20)
    #log("finished inference")
    await go(val_jobs("threeshot2", "threeshot2val"),20)
    log("finished validation")


# asyncio.run(gogogo())
# check_accuracy2(["tree2iterval/5/08100000000000"])
#check_correctness(recurse_find_folders("exhaust2iterval"))
# check_accuracy2(recurse_find_folders("tree3val"))
# for i in range(8):
#     write_stats(f"tree3val/{i}", f"opt_tree3/stats{i}.csv")
# print(sum([len(correctness(f"exhaust2val/{i}")) for i in range(8)]))


# asyncio.run(gogogo())

check_accuracy2(["threeshotval","threeshot2val","threeshothumanval"])

