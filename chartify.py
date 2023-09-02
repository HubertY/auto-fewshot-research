from util import correctness_stats, measure, pass_at_k, listdir, plotterino


from util import open_file


def plot_scores(students_scores, path, datapoints=None, legend=None, xlabel="", ylabel="", title=""):
    num_questions = len(students_scores[0])  # Number of questions
    num_students = len(students_scores)  # Number of students

    # Sort the question indexes
    sorted_indexes = sorted(range(num_questions),
                            key=lambda i: students_scores[0][i])

    labels = datapoints if datapoints is not None else list(
        range(num_questions))
    labels = [labels[i] for i in sorted_indexes]
    # Create a list of question labels based on the sorted indexes
    question_labels = [f"{i}" for i in labels]

    # Create a list of scores for each student based on the sorted indexes
    sorted_scores = [[student[i] for i in sorted_indexes]
                     for student in students_scores]

    plotterino(sorted_scores, xlabel, ylabel, path, title, None, legend)


def fmt(i):
    if isinstance(i, float):
        return f"{i:.4f}"
    else:
        return str(i)


def chartify(data, title="chart", first_label="x", in_labels=None, out_labels=None, path=None):
    if in_labels is None:
        in_labels = data.keys()
    if out_labels is None:
        out_labels = []
        for k, v in data.items():
            out_labels = v.keys()
            break

    s = title + "\n"
    s += first_label
    s += "".join([f" {ol}" for ol in out_labels]) + "\n"
    for inl in in_labels:
        s += str(inl)
        s += "".join([f" & {fmt(data[inl][ol])}" for ol in out_labels]) + "\\\\\n"
    if path is not None:
        with open_file(path, "w") as f:
            f.write(s)
    return s


goods = [0, 2, 3, 4, 7, 8, 12, 13, 15, 16, 18, 23, 24, 26, 27, 28, 29, 30, 31,
         34, 35, 42, 45, 48, 49, 51, 52, 53, 54, 55, 58, 60, 63, 66, 101, 152, 162]
kernel = [0, 8, 18, 28, 35, 51, 58, 152]


def main():
    def _pass_at(k):
        return lambda c, n, cst: pass_at_k(n, c, k)

    def _pass_at_t(k):
        return lambda c, n, cst: pass_at_k(cst, c, k)

    metrics = {
        "pass@any": lambda c, n, cst: c > 0,
        "pass@1": _pass_at(1),
        "pass@10": _pass_at(10),
        "pass@100": _pass_at(100),
        "costn": lambda c, n, cst: cst/n,
        "cost": lambda c, n, cst: cst,
    }
    pmetrics = {
        "pass@any": lambda c, n, cst: c > 0,
        "pass@1": _pass_at(1),
        "pass@10": _pass_at(10),
        "pass@100": _pass_at(100),
        "pass@100t": _pass_at_t(100),
        "pass@1000t": _pass_at_t(1000),
        "costn": lambda c, n, cst: cst/n,
        "cost": lambda c, n, cst: cst,
    }

    data, probs = correctness_stats(
        ["data/codegen_vanilla", "data/codegen_lowt"])
    _data = []
    _data.append([v[0]/200 for v in data["data/codegen_vanilla"].values()])
    _data.append([v[0]/200 for v in data["data/codegen_lowt"].values()])
    plot_scores(_data, "charts/graph1.png", legend=["T=1.0", "T=0.2"], xlabel="problem",
                ylabel="pass@1 accuracy", title=f"GPT-J Perf., HumanEval (N = {len(_data[0])}, 200 samples), Sorted")

    _data = []
    _data.append([data["data/codegen_vanilla"][k][0]/200 for k in goods])
    _data.append([data["data/codegen_lowt"][k][0]/200 for k in goods])
    plot_scores(_data, "charts/graph2.png", datapoints=goods, legend=["T=1.0", "T=0.2"], xlabel="problem",
                ylabel="pass@1 accuracy", title=f"GPT-J Perf., Abridged HumanEval (N = {len(_data[0])}, 200 samples), Sorted")

    data, probs = correctness_stats(
        ["data/codegen_pmass", "data/codegen_answerkey", "data/codegen_vanilla", "data/codegen_lowt"])
    stats = measure(data, metrics)
    chartify(
        stats, title=f"Sampling Strategy Performance, Full Dataset (N = {len(probs)})", path="charts/chart3.txt")

    data, probs = correctness_stats(["data/codegen_pmass", "data/codegen_answerkey",
                                    "data/codegen_vanilla", "data/codegen_lowt"], include=goods, exclude=kernel)
    stats = measure(data, metrics)
    chartify(
        stats, title=f"Sampling Strategy Performance, Abridged Dataset (N = {len(probs)})", path="charts/chart4.txt")

    tmetrics = {
        "pass@1": _pass_at(1),
        "pass@100t": _pass_at_t(100),
        "pass@1000t": _pass_at_t(1000),
        "found": lambda c, n, cst: c,
        "cost": lambda c, n, cst: cst,
    }

    data, probs = correctness_stats(
        ["data/codegen_pmass", "data/codegen_answerkey"], include=goods, exclude=kernel)
    stats = measure(data, tmetrics)
    chartify(
        stats, title=f"Token Efficiency, Abridged Dataset (N = {len(probs)})", path="charts/chart5.txt")

    data, probs = correctness_stats(["data/codegen_pmass"], include=kernel)
    data = data["data/codegen_pmass"]
    stats = {p: {"cost": v[2], "n": v[1], "c": v[0],
                 "selected": 8 if v[0] > 8 else v[0]} for (p, v) in data.items()}
    chartify(stats, title=f"Prompt Proposal Phase",
             first_label="problem", in_labels=kernel, path="charts/chart6.txt")

    dirs = [item for d in listdir("data/codegen_pmass_iter")
            for item in listdir(d)]
    dirs.append("data/codegen_pmass")
    data, probs = correctness_stats(dirs, union=True, include=kernel)
    print(probs)
    stats = measure(data, pmetrics)
    keys = list(stats.keys())
    keys.sort(key=lambda k: stats[k]["pass@1"], reverse=True)
    print(chartify(stats, title=f"Prompt Evaluation Phase",
          first_label="prompt", in_labels=keys, path="charts/chart7.txt"))

    dirs = [
        "data/codegen_pmass",
        "data/codegen_pmiterval/28_1-58_13",
        "data/codegen_pmiterval/58_13",
        "data/codegen_pmiterval/58_15",
        "data/codegen_pmcanonval/58",
        "data/codegen_pmiterval/152_151",
    ]
    data, probs = correctness_stats(dirs, include=goods, exclude=kernel)
    stats = measure(data, pmetrics)
    chartify(
        stats, title=f"Prompt Performance Validation, Abridged Dataset (N = {len(probs)})", path="charts/chart8.txt")


main()
