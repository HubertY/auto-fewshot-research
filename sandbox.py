# https://bitbucket.org/fdominec/experimental_sandbox_in_cpython38/src/master/sandbox_experiment.py

#!/usr/bin/python3.8
# -*- coding: utf-8 -*-

# QUEST FOR A PYTHON SANDBOX:
# 1. You design an script where an exec() command is ran.
# 2. The code ran by exec() is supplied by your opponent, trying e.g. to
#    write something to the './forbidden.txt' file or access
#    forbidden_object. Your opponent has
#    full knowledge of the outer script (no security through obscurity).
#    There should be no trick to break the sandbox whatsoever. Crashing
#    the subprocess Cpython does not count.
# 3. The exec()uted code should however allow as large subset of Python
#    capabilities as possible, including interaction with variables/
#    functions/objects in the outer script, if and only if they are
#    explicitly listed as accessible.
# 4. Elegant, fast, dependency-frugal solution is also appreciated.
# 5. The outer script writes something to the './allowed.txt' file
#    in the end.


from typing import List, Tuple, Optional, Any
import dataset
safe_command = """
def hello():
    e = 5 + 5
    e = e*e
    print("hello world", e)
hello()
"""


def false_print(*arg):
    pass


accessible_locals = {"List": List, "Tuple": Tuple,
                     "Any": Any, "print": false_print, "Optional": Optional}
event_whitelist = ["compile", "exec"]


def sandbox2020(code, timeout=1):
    def ran_in_subprocess(conn):
        from sys import addaudithook

        def block_mischief(event, arg):
            if type(event) != str:
                raise

            if event not in event_whitelist:
                raise IOError(f"halt: illegal event {event}")
            # Security note: Well-designed objects can be passed to this function that could expose the top-level namespace
            # (through catching error and reading sys.exc_info()[2].tb_frame.f_back.f_globals). This should not enable modifying
            # variables outside the multiprocessing sandbox, but could give access to some internal sandbox variables. This function
            # thus should not refer to any 'lock variables'. It is safer to check the results in the main thread.
            if event == 'open' and type(arg[1]) == str and arg[1] != 'r':
                #print('\taudit:', event, arg)
                raise IOError('file write forbidden')
            if event.split('.')[0] in ['subprocess', 'shutil', 'winreg']:
                #print('\taudit:', event, arg)
                raise IOError(
                    'potentially dangerous, filesystem-accessing functions forbidden')

        addaudithook(block_mischief)
        # No way to remove or circumwent audit hooks from python. No access to this function.
        del(block_mischief)

        #sandbox_locals['conn'] = con
        try:
            exec(code, accessible_locals)
            conn.send(False)
        except AssertionError:
            conn.send(f"test failed")
        except Exception as e:
            conn.send(f"aborted {e}")
        conn.close()

    from multiprocessing import Process, Pipe
    parent_conn, child_conn = Pipe()
    p = Process(target=ran_in_subprocess, args=(child_conn,))
    p.start()

    # NOTE: The sandbox is probably safe against file writing, as well as against access into the main process.
    # Yet the objects returned from it as results could have been manipulated. Asserting the output objects to be
    # of expected data types is an extra safety measure. But be careful whenever your main program flow is
    # controlled by the returned objects' attributes, e.g. file paths could change.

    if parent_conn.poll(timeout):
        p.terminate()
        return parent_conn.recv()
    else:
        p.terminate()
        return "timeout"


humaneval = dataset.load_dataset("./humaneval.jsonl")


def preprocess(s):
    s = s.split("<|endoftext|>")[0]
    lines = s.split("\n")
    if len(lines) > 0 and not lines[-1].startswith("    "):
        lines.pop()
    for i, line in enumerate(lines):
        if line.startswith("def"):
            return "\n".join(lines[i:])


def construct(problem, completion):
    s = preprocess(completion)

    #s += item["canonical_solution"]
    s += problem["test"] + "\n"
    s += f"check({problem['entry_point']})"
    return s


def test_correctness(problemdata, completion):
    s = construct(problemdata, completion)
    # print(s)
    ret = sandbox2020(s, 0.5)
    # print(ret)
    return ret
