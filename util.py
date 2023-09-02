import numpy as np
import os
import json
import matplotlib
import matplotlib.pyplot as plt


def listdir(path):
    for item in os.listdir(path):
        yield f"{path}/{item}"


def readjson(path):
    with open(path, "r") as f:
        return json.load(f)


def readfile(path):
    with open(path, "r") as f:
        return f.read()


def count_lines(fpath, exclude=None):
    try:
        with open_file(fpath, "r") as f:
            c = 0
            for i, l in enumerate(f):
                if exclude is None or not exclude in l:
                    c += 1
    except:
        c = -1
    return c


def pass_at_k(n, c, k):
    if n == -1:
        return -1
    if c == 0:
        return 0
    if n - c < k:
        return 1.0
    return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))


def open_file(file_path, opt):
    directory = os.path.dirname(file_path)
    if len(directory) > 0 and not os.path.exists(directory):
        os.makedirs(directory)
    return open(file_path, opt)


def correctness_stats(folders=[], include=None, exclude=None, union=False):
    probs = []
    data = {folder: {} for folder in folders}
    fn = set.intersection if not union else set.union
    for prob in fn(*[set(os.listdir(folder)) for folder in folders]):
        prob = int(prob)
        if include is not None and prob not in include:
            continue
        if exclude is not None and prob in exclude:
            continue
        probs.append(prob)

        for folder in folders:
            if str(prob) in os.listdir(folder):
                corrness = readjson(f"{folder}/{prob}/correctness.json")
                cost = count_lines(
                    f"{folder}/{prob}/log.jsonl", exclude="false")
                data[folder][prob] = [sum(corrness), len(corrness), cost]
            else:
                data[folder][prob] = None
    probs.sort()
    return data, probs


def mean(l):
    return sum(l)/len(l)


def measure(data, metrics={}):
    data = {k: {kk: {metric: mf(*vv) if vv is not None else None for metric,
                     mf in metrics.items()} for kk, vv in v.items()} for k, v in data.items()}
    probs = []
    for v in data.values():
        probs = v.keys()
        break
    for prob in probs:
        for metric in metrics.keys():
            vals = []
            for v in data.values():
                val = v[prob][metric]
                if val is not None:
                    vals.append(val)
            avg = sum(vals)/len(vals)
            for v in data.values():
                val = v[prob][metric]
                if val is None:
                    v[prob][metric] = avg

    ret = {k: {metric: mean([d[metric] for d in v.values()])
               for metric in metrics.keys()} for k, v in data.items()}

    return ret


def plotterino(data, xlabel, ylabel, path, title, xvals=None, legend=None, ticksize=None):
    # Plotting the scores
    for datum in data:
        plt.plot(xvals if xvals else [
                 i for i in range(len(datum))], datum, marker='o')

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
