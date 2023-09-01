components = [
    "-remove all instances of the lowercase letter s from the string",
    "-replace all spaces with exclamation points in the string",
    "-convert the string to lowercase",
    "-remove the first and last two characters of the string",
    "-remove all vowels from the string",
    "-remove every third character from the string",
    "-drop the last half of the string, as computed by characters",
    "-replace spaces with triple spaces",
    "-reverse the order of words in the string",
    "-add the word apples after every word in the string",
    "-make every other character in the string uppercase",
    "-delete all exclamation points, question marks, and periods from the string"
]

solutions = [
    lambda s : s.replace("s", ""),
    lambda s : s.replace(" ", "!"),
    lambda s : s.lower(),
    lambda s : s[2:-2],
    lambda s : "".join(char for char in s if char not in "aeiouAEIOU"),
    lambda s : "".join(char for i, char in enumerate(s) if i % 3 != 0),
    lambda s : s[: len(s) // 2],
    lambda s : s.replace(" ", " "),
    lambda s : " ".join(s.split()[::-1]),
    lambda s : " ".join(word + " apples" for word in s.split()),
    lambda s : "".join(char.upper() if i % 2 == 0 else char for i, char in enumerate(s)),
    lambda s : "".join([x for x in s if x not in ".!?"])
]

header = """def string_manipulation(s: str):
    \"\"\"
    This function takes a string as input, then returns the result of performing
    the following sequence of 
    manipulations on that string:
"""

test_string = "the TEST quick Brown test fox JUMPS test over the tesT lazy test dog"

def checkify(sol):
    return f"""def check(candidate):
    assert candidate("{test_string}") === "{sol}"\n
"""

import json
import random
def gen_dataset(outfile, max = 7, n = 1):
    for k in range(n):
        items = list(range(max))
        s = header
        sol = test_string

        random.shuffle(items)

        for i in items:
            s += "    "
            s += components[i]
            s += "\n"

            sol = solutions[i](sol)


            data = {"task_id": f"Synthetic/{k}-{i}",
                    "prompt": s + "    \"\"\"\n",
                    "entry_point": "string_manipulation",
                    "test": checkify(sol)
                    }
            with open(outfile, "a") as f:
                f.write(json.dumps(data))
                f.write("\n")

gen_dataset("synthetic.jsonl", n=5)