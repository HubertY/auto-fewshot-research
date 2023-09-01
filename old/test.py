from tokenize import String
from transformers import BatchEncoding, BloomForCausalLM, AutoTokenizer, PreTrainedTokenizer

import torch
import torch.distributed as dist
from torch import Tensor, FloatTensor, nn

from transformers.generation_logits_process import (LogitsProcessorList)


class Searcher:
    def tokens(self):
        return torch.argsort(self.probs, -1, True)[0][:100]

    def sample(self):
        return torch.multinomial(self.probs, num_samples=1).squeeze(1)

    def append(self, next_tokens: Tensor):
        next_tokens = next_tokens * self.unfinished_sequences  # tensor([1])
        print("append", repr(self.tokenizer.decode(next_tokens)))
        # update generated ids, model inputs, and length for next step
        self.input_ids = torch.cat(
            [self.input_ids, next_tokens[:, None]], dim=-1)

    def greedyappend(self):
        self.append(torch.argmax(self.scores))

    def sampleappend(self):
        self.append(self.sample())

    def forward(self):
        # forward pass to get next token
        model_inputs = self.model.prepare_inputs_for_generation(self.input_ids)
        outputs = self.model(**model_inputs, return_dict=True)

        cur_len = self.input_ids.shape[-1]

        # pre-process distribution
        next_token_logits = outputs.logits[:, -1, :]
        next_token_scores = self.logits_processor(
            self.input_ids, next_token_logits)
        next_token_scores = self.logits_warper(
            self.input_ids, next_token_scores)

        probs = nn.functional.softmax(next_token_scores, dim=-1)

        self.scores = next_token_scores
        self.probs = probs
        cur_len = cur_len + 1

    def export(self):
        return self.tokenizer.batch_decode(self.input_ids)

    def gen(self, n, fn):
        for i in range(n):
            self.forward()
            fn()

    def tokenize(self, s: str):
        print("tokenizing...")
        ret = self.tokenizer(s, return_tensors="pt").input_ids
        print("tokenizing done")
        return ret

    def set_inputs(self, s: str):
        self.input_ids = self.tokenize(s)
        self.unfinished_sequences = self.input_ids.new(
            self.input_ids.shape[0]).fill_(1)

    def append_inputs(self, s: str):
        self.append(self.tokenize(s))

    def __init__(
        self,
        model: BloomForCausalLM,
        tokenizer: PreTrainedTokenizer,
        # logits_processor: LogitsProcessorList,
        # logits_warper: LogitsProcessorList,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.logits_processor = LogitsProcessorList()
        self.logits_warper = LogitsProcessorList()
        self.input_ids = BatchEncoding()
        self.unfinished_sequences = None
        self.scores = FloatTensor()
        self.probs = Tensor()
        self.set_inputs("")

model_path = "./bloom"

import time

start_time = time.perf_counter()
def recordTime():
    print(time.perf_counter() - start_time)

# Generate
print("loading model")
model = BloomForCausalLM.from_pretrained(model_path)
recordTime()
print("loading tokenizer")
tokenizer = AutoTokenizer.from_pretrained(model_path)
recordTime()

prompt = "Hey, are you conscious? Can you talk to me?"
print("initializing searcher")
gen = Searcher(model, tokenizer)
gen.set_inputs(prompt)
recordTime()
gen.gen(10, gen.sampleappend)
recordTime()
print(gen.export())
recordTime()
