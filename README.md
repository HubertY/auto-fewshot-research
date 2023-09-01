# auto-fewshot-research

[paper here](https://hubertyuan.com/static/files/thesis_en.pdf)

Use `gptserver.py` to manage instances of GPT-J, one per GPU. Each instance will listen for requests on a local port.

`testrig2.py` are the experiments for GPT-J and HumanEval.

Old code for the BLOOM server and HumanEval dataset is under `/old` (unorganized).

To replicate data summarization, download and extract [`data.tar.gz`](https://drive.google.com/file/d/1NwDAIsR52x5ZZwGfa2ZIWUjgbDNV8Ekd/view?usp=sharing) and run `chartify.py`. To replicate GPT-J/HumanEval data generation, all necessary code is contained in `testrig2.py` (see the main functions).

Unfortunately the archived data excludes the full tree logs as they are too large, so the token cost figures for naive sampling can't be easily reconstructed. The fields that require this data (relating to naive sampling efficiency) are output as placeholder -1 by `chartify.py`.