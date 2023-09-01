
from util import listdir
from sandbox import test_correctness
import sys
from util import readjson, readfile
import gptjclient
import asyncio
import json
import torch
import time
import os
import pickle

import math

import dataset
humaneval = dataset.load_dataset("./humaneval.jsonl")

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def new_future():
    return asyncio.get_event_loop().create_future()


def is_token_code(token, i):
    if isinstance(token, int):
        return token == i
    if isinstance(token, list):
        return False
    else:
        return token is not None and token[0].item() == i


def is_space_token(token):
    return is_token_code(token, 220)


def is_stop_token(token):
    return is_token_code(token, gptjclient.tokenizer.eos_token_id)  # 50256


def make_query(q):
    # TODO
    pass


def sample_p(probs, topk=64, temp=1):
    if temp == 0:
        return 0
    return torch.multinomial(torch.tensor(probs)[:topk] ** (1/temp), num_samples=1).item()

# def sample_token(tokens, probs, temp = 1):
#     index = torch.multinomial(probs ** (1/temp), num_samples=1).squeeze(1)[0].item()
#     return tokens[:, index]


def extend_input(input, tokens):
    return torch.cat([input] + [torch.tensor([token], dtype=torch.int).unsqueeze(0) for token in tokens if token is not None], dim=-1)


def make_token_entry(_tokens, _probs, selection, attn, N=20):
    attn = attn.to_list() if attn is not None else []
    token_ids = []
    probs = []
    for i in range(N):
        probs.append(_probs[:, i][0].item())
        token_ids.append(_tokens[:, i][0].item())

    return {"text": gptjclient.detokenize(selection), "picked": selection[0].item(), "ids": token_ids, "probs": probs}


def open_file(file_path, opt):
    directory = os.path.dirname(file_path)
    if len(directory) > 0 and not os.path.exists(directory):
        os.makedirs(directory)
    return open(file_path, opt)


def file_exists(filename):
    return os.path.isfile(filename)


def log(s, fname="log.log"):
    with open_file(fname, "a") as f:
        f.write(f"{str(s)}\n")


def adjust_temp(l, t):
    if t == 1:
        return l
    else:
        l = [p**(1/t) for p in l]
        s = sum(l)
        return [p/s for p in l]


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

    def get_input_ids(self):
        items = []
        for node in self.get_ancestors():
            if node.picked is not None:
                if isinstance(node.picked, list):
                    items.extend(node.picked)
                else:
                    items.append(node.picked)
        return extend_input(self.master.input, items)

    def get_text(self, include_prompt=True):
        s = self.master.s if include_prompt else ""
        for node in self.get_ancestors():
            s += node.text
        return s

    def get_code(self):
        return [node.sel for node in self.get_ancestors() if node.sel is not None]

    def write(self, path):
        with open_file(path, "w") as f:
            f.write(self.get_text())

    async def compute(self, cli, temp=1):
        if not self.computed:
            await cli.send("ADD", self.get_input_ids())
            await cli.send("FORWARD")
            self.next_tokens = (await cli.send("TOKENS")).squeeze(-1)[0].tolist()
            self.next_probs = adjust_temp((await cli.send("PROBS")).squeeze(-1)[0].tolist(), temp)
            self.computed = True

    def sel_child(self, n):
        if self.computed:
            picked = self.next_tokens[n]
            if picked not in self.children:
                self.children[picked] = CompletionNode(
                    self.master, self, picked, self.next_probs[n], n)
            return self.children[picked]
        else:
            raise Error("node is not computed")

    def __init__(self, master, parent, picked, prob, sel):
        self.master = master
        self.parent = parent
        self.picked = picked
        if is_space_token(self.picked):
            if self.parent is None or self.parent.text.endswith("\n"):
                self.picked = [220, 220, 220]
                self.text = "   "
            elif self.parent.text.endswith("   "):
                self.picked = [220, 220, 220, 220]
                self.text = "    "
            else:
                self.text = gptjclient.detokenize(
                    self.picked) if self.picked else ""
        else:
            self.text = gptjclient.detokenize(
                self.picked) if self.picked else ""
        #print(">", self.picked, self.text)
        self.prob = prob
        self.sel = sel

        self.cumlogprob = math.log(
            self.prob) + (self.parent.cumlogprob if self.parent else 0)

        self.computed = False
        self.next_tokens = []
        self.next_probs = []
        self.children = {}  # dict token id to Node

    def to_dict(self):
        return {
            "code": self.get_code(),
            "picked": 220 if isinstance(self.picked, list) else self.picked,
            "prob": self.prob,
            "sel": self.sel,
            "computed": self.computed,
            "next_tokens": self.next_tokens,
            "next_probs": self.next_probs
        }

    @staticmethod
    def from_dict(master, parent, data):
        ret = CompletionNode(
            master, parent, data["picked"], data["prob"], data["sel"])
        ret.computed = data["computed"]
        ret.next_tokens = data["next_tokens"]
        ret.next_probs = data["next_probs"]
        return ret


def new_future():
    return asyncio.get_event_loop().create_future()


class Worker:
    def __init__(self, port):
        self.port = port
        self.cli = None

    async def connect(self):
        self.cli = gptjclient.Client()
        await self.cli.connect(self.port)

    async def work(self, job):
        await job(self.cli)

    async def close(self):
        await self.cli.send("CLOSE")
        self.cli.close()


class WorkerPool:
    def __init__(self, servers=range(8), max=500):
        self.workers = [[Worker(1330+p) for i in range(max)] for p in servers]
        self.available = [[] for p in servers]
        self.future = [new_future() for p in servers]

    async def conn(self):
        for pool in self.workers:
            await asyncio.gather(*[w.connect() for w in pool])
        self.available = [[w for w in p] for p in self.workers]
        print("all ready")

    async def close(self):
        for pool in self.workers:
            await asyncio.gather(*[w.close() for w in pool])

    async def req(self, server, job):
        while len(self.available[server]) == 0:
            await self.future[server]
            if self.future[server].done():
                self.future[server] = new_future()
        cli = self.available[server].pop()
        await cli.work(job)
        self.available[server].append(cli)
        if not self.future[server].done():
            self.future[server].set_result(True)


class WorkerPool2:
    def __init__(self, servers=range(8)):
        self.workers = [[] for p in servers]
        self.available = [[] for p in servers]

    def close(self):
        for pool in self.workers:
            for w in pool:
                w.close()

    async def req(self, server, job):
        cli = None
        if len(self.available[server]) > 0:
            cli = self.available[server].pop()
        else:
            cli = gptjclient.Client()
            await cli.connect(1330+server)
            self.workers[server].append(cli)
        await job(cli)
        self.available[server].append(cli)


worker_pool = WorkerPool2()

# async def jobbify(proc, server=0):
#     cli = gptjclient.Client()
#     await cli.connect(1330+server)
#     await proc(cli)
#     cli.close()


async def jobbify(proc, server=0):
    await worker_pool.req(server, proc)


class CompletionTree:
    def write(self, path):
        lines = []
        lines.append(json.dumps({"id": self.id, "s": self.s}))

        nodes = [self.head]
        while len(nodes):
            node = nodes.pop()
            lines.append(json.dumps(node.to_dict()))
            for key in node.children:
                nodes.append(node.children[key])

        with open_file(path, "w") as f:
            f.write("\n".join(lines))

    @staticmethod
    def read(path):
        ret = None
        nodes = {}
        with open_file(path, "r") as file:
            for i, line in enumerate(file):
                data = json.loads(line)
                if i == 0:
                    ret = CompletionTree(data["id"], data["s"])
                else:
                    if len(data["code"]) == 0:
                        data["picked"] = None
                        node = CompletionNode.from_dict(ret, None, data)
                        nodes[str([])] = node
                        ret.head = node
                    else:
                        parent = nodes[str(data["code"][:-1])]
                        node = CompletionNode.from_dict(ret, parent, data)
                        nodes[str(data["code"])] = node
                        parent.children[data["picked"]] = node
        return ret

    def __init__(self, id, s):
        self.id = id
        self.s = s
        self.input = gptjclient.tokenize(self.s)
        self.head = CompletionNode(self, None, None, 1, None)

    async def sample(self, server, node, N, stoppingcondition=lambda t, s: False, sel=lambda n: 0):
        # sys.stdout.write(self.s)
        for i in range(N):
            if not node.computed:
                await jobbify(node.compute, server)
            node = node.sel_child(sel(node))
            # sys.stdout.write(node.text)

            if stoppingcondition(node):
                return node.parent
        return node

    async def beamburst(self, server, node, N, depth=1, ret=None):
        if ret is None:
            ret = []
        if not node.computed:
            await jobbify(node.compute, server)
        if(depth > 1):
            await asyncio.gather(*[self.beamburst(server, node.sel_child(i), N, depth-1, ret) for i in range(N)])
        else:
            ret.append(node)
    # def jobs(self, n, decay = 50):
    #     front = list(self.frontier)
    #     front.sort(reverse=True, key=lambda node: node.cumlogprob )
    #     # front.sort(reverse=True, key=lambda node: node.prob * (0.5 ** (len(node.code)/decay)) )
    #     def jobbify(node):
    #         return lambda cli: node.greed(cli, 50)
    #     return [jobbify(node) for node in front[:n]]


def find_highest_mass_node(tree, exclude):
    bfs_queue = [tree.head]
    bestscore = -float("inf")
    bestnode = None
    while len(bfs_queue):
        q = bfs_queue
        bfs_queue = []
        for node in q:
            if not node.computed:
                raise Exception("node must be computed")
            for i in range(len(node.next_tokens)):
                child = node.sel_child(i)
                if child in exclude:
                    continue
                if child.cumlogprob > bestscore:
                    if child.computed:
                        bfs_queue.append(child)
                    else:
                        bestscore = child.cumlogprob
                        bestnode = child
                        break
                else:
                    break
    return bestnode


def sampling_selector(node):
    return sample_p(node.next_probs)


def count_leading_spaces(s):
    return next((i for i, c in enumerate(s) if c != " "), len(s))


def stopping_condition(node):
    if is_stop_token(node.picked) or node.get_input_ids().shape[1] >= 2000:
        return True
    else:
        s = node.get_text()
        lines = [l for l in s.split("\n") if len(l)]
        if len(lines) >= 2 and count_leading_spaces(lines[-1]) > count_leading_spaces(lines[-2]) + 4:
            return True
        if len(lines) > 0:
            last_line = lines[-1]
            return len(last_line) > 0 and not last_line[:4].isspace()
        return False


def save_obj(obj, path):
    with open_file(path, "wb") as f:
        pickle.dump(obj, f)


def load_obj(path):
    with open_file(path, "rb") as f:
        return pickle.load(f)


DONE = 0


async def naive_sampling(prob, path, prefix="", N=200, L=500, serv=None, dataset=humaneval):
    if serv == None:
        serv = prob % 8

    tree = CompletionTree(prob, prefix + dataset[prob]["prompt"])

    await tree.beamburst(serv, tree.head, 5, 4)

    async def job(i):
        print(f"starting {prob}-{i}")
        s1 = (await tree.sample(serv, tree.head, L, stopping_condition, sampling_selector)).get_text()
        with open_file(f"{path}/{prob}/{i}.py", "w") as f:
            global DONE
            DONE += 1
            print(f"{path}/{prob}/{i}.py", DONE)
            f.write(s1)

    await asyncio.gather(*[job(i) for i in range(N)])
    print("saving", prob)
    tree.write(f"{path}/{prob}/tree.jsonl")


async def repair_tree(path="codegen"):
    async def job(i, prob, path, s, ol):
        tree = CompletionTree(prob, s)
        L = 500
        print(f"fixing {path}", L)
        s1 = (await tree.sample(i % 8, tree.head, L, stopping_condition, sampling_selector)).get_text()
        with open_file(path, "w") as f:
            print(f"fixed {path}")
            f.write(s1)

    jobs = []
    i = 0
    for folder in os.listdir(path):
        pl = CompletionTree(
            int(folder), humaneval[int(folder)]["prompt"]).input.shape[1]
        for item in listdir(f"{path}/{folder}"):
            with open_file(item, "r") as f:
                s = f.read()
                if not stopping_condition(s):
                    jobs.append(job(i, int(folder), item, s, pl))
                    i += 1
    print(len(jobs))
    await asyncio.gather(*jobs)


def correctness(paths=["codegen"], dataset=humaneval, n=None):
    for path in paths:
        for folder in os.listdir(path):
            if isinstance(n, list) and int(folder) not in n:
                continue
            data = {}
            for item in os.listdir(f"{path}/{folder}"):
                if item.endswith(".py"):
                    i = int(item.split(".py")[0])
                    with open_file(f"{path}/{folder}/{item}", "r") as f:
                        s = f.read()
                        res = test_correctness(dataset[int(folder)], s)
                        if res:
                            # print("bad", res)
                            data[i] = 0
                        else:
                            # print("good", folder, i)
                            data[i] = 1
            data = [data[i] for i in range(len(data))]
            #print(folder, data, sum(data)/len(data))
            with open_file(f"{path}/{folder}/correctness.json", "w") as f:
                f.write(json.dumps(data))


def count(path="codegen", expect=200):
    lack = 0
    for folder in listdir(path):
        count = 0
        for item in listdir(folder):
            count += 1
        if(count < expect):
            print(folder, count)
            lack += count
        else:
            lack += expect
    print(162*200-lack)


def lenstat(path="codegen"):
    counts = []
    i = 0
    for folder in os.listdir(path):
        pl = CompletionTree(int(folder), humaneval[int(
            folder)]["prompt"].replace("    ", " ")).input.shape[1]
        z = []
        for item in listdir(f"{path}/{folder}"):
            with open_file(item, "r") as f:
                s = f.read()
                z.append(CompletionTree(0, s.replace(
                    "    ", " ")).input.shape[1] - pl)
        print(z)
        counts.append(z)
    print(counts)
    print(sum([sum(item) for item in counts]))


async def serialtest():
    tree = CompletionTree(0, "Answer Key:\n\n"+humaneval[0]["prompt"])
    await tree.beamburst(0, tree.head, 5, 4)
    tree.write("treeserial1.jsonl")
    tree2 = CompletionTree.read("treeserial1.jsonl")
    tree2.write("treeserial2.jsonl")
    tree3 = CompletionTree.read("treeserial2.jsonl")
    tree3.write("treeserial3.jsonl")


async def high_probability_branching(prob, path, prefix="", T=10000, L=50, serv=None, dataset=humaneval):
    if serv == None:
        serv = prob % 8

    tree = CompletionTree(prob, prefix + dataset[prob]["prompt"])
    terminals = set()
    tasks = set()
    answers = set()

    DONE = new_future()

    def donezo(task):
        tasks.discard(task)
        log(str(T), "logiter.log")
        if len(tasks) == 0:
            if T <= 0:
                DONE.set_result(True)
            else:
                candidate = find_highest_mass_node(tree, terminals)
                handling.add(candidate)
                deploy(go(candidate))

    def deploy(t):
        task = asyncio.create_task(t)
        tasks.add(task)
        task.add_done_callback(donezo)

    handling = set()

    async def go(node):
        nonlocal T
        #print("go", node.get_code(), node.prob, node.cumlogprob, T)
        if node not in handling:
            handling.add(node)
        for i in range(L):
            if stopping_condition(node):
                # print("terminal", node.get_code(), node.get_text())
                terminals.add(node)
                answers.add(node.parent)
                break

            with open_file(f"{path}/{prob}/log.jsonl", "a") as f:
                f.write(json.dumps(node.get_code()))
                f.write("\n")
            await jobbify(node.compute, serv)

            T -= 1

            handling.discard(node)
            node = (None if i == L-1 else node.sel_child(0))
            if node is not None:
                handling.add(node)

            if T > (len(tasks)-1) * L:
                candidate = find_highest_mass_node(tree, terminals)
                if candidate not in handling:
                    handling.add(candidate)
                    deploy(go(candidate))

    deploy(go(tree.head))

    await DONE
    for i, ansnode in enumerate(answers):
        with open_file(f"{path}/{prob}/{i}.py", "w") as f:
            f.write(ansnode.get_text())
    tree.write(f"{path}/{prob}/tree.jsonl")


async def promptify(problemset, indir, outdir):
    step = len(goods) // 8 + 1
    use = goods[::step]
    print(len(use))
    n = 0
    jobs = []
    for problem in use:
        #probs = readjson(f"{indir}/{problem}/probs.json")
        correct = readjson(f"{indir}/{problem}/correctness.json")
        correct = [i for i in range(len(correct)) if correct[i]][:4]

        for corr in correct:
            s = readfile(f"{indir}/{problem}/{corr}.py").strip()
            print(s)
            for p in use:
                if p != problem:
                    jobs.append(high_probability_branching(
                        p, f"{outdir}/{problem}/{corr}", prefix=f"{s}\n\n", serv=n % 8))
                    n += 1
    await asyncio.gather(*jobs)
    dirs = []
    for originpath in listdir(outdir):
        for promptselpath in listdir(originpath):
            dirs.append(promptselpath)
    correctness(dirs)


def dump_probs(dirs):
    for dir in dirs:
        probs = {}
        tree = CompletionTree.read(f"{dir}/tree.jsonl")
        for file in listdir(dir):
            if file.endswith(".py"):
                s = readfile(file).strip()
                nodes = [tree.head]
                n = 0
                while len(nodes):
                    node = nodes.pop()
                    n += 1
                    if node.get_text().strip() == s:
                        probs[file] = [node.cumlogprob,
                                       len(node.get_ancestors())]
                        break
                    nodes.extend(node.children.values())
                if file not in probs:
                    raise Exception(f"completion not found {file} {n}")
        with open_file(f"{dir}/probs.json", "w") as f:
            f.write(json.dumps(probs))


async def main1():
    jobs = []
    for p in range(len(humaneval)):
        jobs.append(naive_sampling(p, "codegen_answerkey",
                    prefix="Answer Key:\n", serv=p % 8))
        jobs.append(naive_sampling(p, "codegen_vanilla", serv=p % 8))
        jobs.append(high_probability_branching(p, "codegen_pmass",
                    prefix="Answer Key:\n", serv=p % 8))
    await asyncio.gather(*jobs)
    worker_pool.close()
    correctness(["codegen_pmass", "codegen_answerkey", "codegen_vanilla"])
    print("graceful completion!")

goods = [0, 2, 3, 4, 7, 8, 12, 13, 15, 16, 18, 23, 24, 26, 27, 28, 29, 30, 31,
         34, 35, 42, 45, 48, 49, 51, 52, 53, 54, 55, 58, 60, 63, 66, 101, 152, 162]
kernel = [0, 8, 18, 28, 35, 51, 58, 152]


async def main2():
    await promptify(goods, "codegen_pmass", "codegen_pmass_iter")


def canonical_prompt(prob, prefix=""):
    s = prefix + humaneval[prob]["prompt"] + \
        humaneval[prob]["canonical_solution"]
    return s


async def promptiter(prompt, outdir, sampler=naive_sampling, include=None, exclude=None):
    jobs = []
    if prompt is None:
        s1 = "Answer Key:\n\n"
    elif isinstance(prompt, str):
        s1 = readfile(prompt).strip() + "\n\n"
    elif isinstance(prompt, int):
        s1 = canonical_prompt(prompt, prefix="Answer Key:\n").strip() + "\n\n"
    else:
        raise Exception("prompt must be None, path or int")
    i = 0
    for p in range(len(humaneval)):
        if include is not None and p not in include:
            continue
        if exclude is not None and p in exclude:
            continue
        jobs.append(sampler(p, outdir, prefix=s1, serv=i % 8))
        i += 1
    await asyncio.gather(*jobs)
    correctness([outdir])


async def main3(include=None, exclude=None):
    jobs = []
    s1 = readfile("codegen_pmass/28/1.py").strip() + "\n\n"
    s2 = readfile("codegen_pmass/58/13.py")[12:].strip() + "\n\n"
    s = s1+s2
    i = 0
    for p in range(len(humaneval)):
        if include is not None and p not in include:
            continue
        if exclude is not None and p in exclude:
            continue
        jobs.append(high_probability_branching(
            p, "codegen_pmiterval/28_1-58_13", prefix=s, serv=i % 8))
        i += 1
    await asyncio.gather(*jobs)
    worker_pool.close()

    correctness(["codegen_pmiterval/28_1-58_13"])


async def main4():
    await promptiter("codegen_pmass/58/13.py", "codegen_lowt_iter", include=goods, exclude=kernel)
    await promptiter(None, "codegen_pm_lowt", sampler=high_probability_branching, include=goods, exclude=kernel)
    await promptiter("codegen_pmass/58/13.py", "codegen_pm_lowt_iter", sampler=supersearch, include=goods, exclude=kernel)

# async def main7():
#     await promptiter(58,"codegen_pmass_iter/58/canonical", sampler=supersearch, include=kernel, exclude = [58])


# x = [path for path in listdir("codegen_canonval/58_7") if not "correctness.json" in list(os.listdir(path))]
# print(x)
# print(len(x))

# print(count("codegen_lowt"))

async def main():
    await main1()
    await main2()
    await main3()
    await main4()

asyncio.run(main())
