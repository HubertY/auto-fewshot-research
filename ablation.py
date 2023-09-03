import matplotlib.pyplot as plt
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
    scores = {}
    Ns = [1, 2, 3, 4, 5, 6, 7]
    for n in Ns:
        print(n)
        scores[n] = []
        counter = new_counter(10)
        for subkernel in permutify(kernel, n):
            data, probs = correctness_stats(
                dirs(subkernel), union=True, include=subkernel)
            stats = measure(data, metric)
            stats = [(k, stats[k]["pass@1"])for k in stats]
            stats.sort(key=lambda a: a[1], reverse=True)
            increment_counter(counter, stats[0][0][24:])
            scores[n].append(_stats[stats[0][0]]["pass@1"])
        print(counter)
        vals = counter.values()
        valsum = sum(vals)
        chartdata.append([val/valsum for val in vals])
    plotterino(chartdata, "prompt", "top-rank rate", "ablation.png", "Top-ranked Prompt Distribution under Input Ablation",
               [k.replace("/", "-") for k in counter.keys()], legend=[f"N={n}" for n in Ns], ticksize=9)
    scores[0] = [0.3079]
    scores[8] = [0.3575744743842625]
    plotterino(scores, "Input Size (N)", "pass@1 score",
               "ablation2.png", "Selection Phase Score Distribution of Selected Prompt under Input Ablation", scores.keys())


partialstat = {0: 0.3079,
               2: 0.3205017692244429,
               3: 0.3323557535734997,
               4: 0.34015109622821427,
               5: 0.34419946828908765,
               6: 0.34963226382931323,
               7: 0.35447447532494314,
               8: 0.3575744743842625}


def scatter(data, xlabel, ylabel, path, title, xvals=None, legend=None, ticksize=None):
    # Plotting the scores
    for datum in data:
        plt.scatter([], [], marker='o')

    # Labeling the axes
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    ax = plt.gca()
    if xvals is None:
        ax.axes.xaxis.set_ticklabels([])
    if ticksize:
        ax.axes.xaxis.set_tick_params(labelsize=ticksize)
    plt.title(title)

    # Show the plot
    if legend:
        plt.legend(legend)
    plt.savefig(path)
    plt.clf()


main()
