from util import correctness_stats, measure, pass_at_k, listdir
from util import plotterino


def permutify(list, n):
    if n == 0:
        yield []
    elif len(list) < n:
        raise ValueError("yikes")
    elif len(list) == n:
        yield list
    else:
        for i in range(n-1, len(list)):
            for l in permutify(list[0:i], n-1):
                yield l + [list[i]]


def dirs(ks, lim=8):
    ret = []
    for d in ks:
        ret += sorted([i for i in listdir(
            f"data/codegen_pmass_iter/{d}") if not i.endswith("canonical")],
            key=lambda s: int(s.split("/")[-1]))[:lim]
    return ret


metric = {"pass@1": lambda c, n, cst: pass_at_k(n, c, 1)}

kernel = [0, 8, 18, 28, 35, 51, 58, 152]

_data, _ = correctness_stats(dirs(kernel), union=True, include=kernel)
_stats = measure(_data, metric)
_statss = [(k[24:], _stats[k]["pass@1"]) for k in _stats]
_statss.sort(key=lambda a: a[1], reverse=True)


def new_counter(n=16):
    ret = {s: 0 for (s, _) in _statss[:n]}
    ret["other"] = 0
    return ret


def increment_counter(ctr, key):
    ctr[key if key in ctr else "other"] += 1


def main():
    chartdata = []
    avgs = {}
    Ns = [4, 5, 6, 7, 8]
    for n in Ns:
        print(n)
        avgs[n] = 0
        counter = new_counter(10)
        for subkernel in permutify(kernel, n):
            data, probs = correctness_stats(
                dirs(subkernel), union=True, include=subkernel)
            stats = measure(data, metric)
            stats = [(k, stats[k]["pass@1"])for k in stats]
            stats.sort(key=lambda a: a[1], reverse=True)
            increment_counter(counter, stats[0][0][24:])
            avgs[n] += _stats[stats[0][0]]["pass@1"]
        print(counter)
        vals = counter.values()
        valsum = sum(vals)
        chartdata.append([val/valsum for val in vals])
        avgs[n] /= valsum
    plotterino(chartdata, "prompt", "top-rank rate", "ablation.png", "Top-ranked Prompt Distribution under Input Ablation",
               [k.replace("/", "-") for k in counter.keys()], legend=[f"N={n}" for n in Ns], ticksize=9)
    print(avgs)


main()
