from code import interact
import time
from transformers import BloomConfig
from transformers.models.bloom.modeling_bloom import BloomBlock, build_alibi_tensor, BloomScaledSoftmax
import os
import torch
import torch.nn as nn
from collections import OrderedDict
import math


class BlockLoader:
    def __init__(self):
        self.blocks = dict()


model_path = "./bloom"  # replace with your local folder path


shards = dict()
def torchload(shard_num: int):
    if shard_num not in shards:
        shards[shard_num] = torch.load(os.path.join(model_path, f"pytorch_model_{shard_num:05d}-of-00072.bin"))
    return shards[shard_num]

def get_state_dict(shard_num, prefix=None):
    d = torchload(shard_num)
    return d if prefix is None else OrderedDict((k.replace(prefix, ''), v) for k, v in d.items())


start_time = time.perf_counter()


def preprint(s):
    print(s, end='', flush=True)


def recordTime(s):
    print(s, time.perf_counter() - start_time)


config = BloomConfig.from_pretrained(model_path)
print(config)
recordTime("loaded config")

def load_embeddings(device: torch.device):
    state_dict = get_state_dict(
        shard_num=1, prefix="word_embeddings_layernorm.")
    embeddings = nn.Embedding.from_pretrained(
        state_dict.pop('word_embeddings.weight')).to(device)
    lnorm = nn.LayerNorm(config.hidden_size,
                         eps=config.layer_norm_epsilon, dtype=torch.bfloat16).to(device)
    lnorm.load_state_dict(state_dict)
    return embeddings, lnorm


def load_causal_lm_head(device: torch.device):
    linear = nn.utils.skip_init(
        nn.Linear, config.hidden_size, config.vocab_size, bias=False, dtype=torch.bfloat16).to(device)
    linear.load_state_dict(get_state_dict(
        shard_num=1, prefix="word_embeddings."), strict=False)
    return linear

def bsd(block_num: int):
    return get_state_dict(shard_num=block_num + 2, prefix=f"h.{block_num}.")

def load_block(block_obj: BloomBlock, block_num: int):
    block_obj.load_state_dict(bsd(block_num))

def prefetch(n=70):
    with open("prefetch.txt", "w") as f:
        for i in range(n):
            print(f"prefetching block data {i+1}/{n}")
            bsd(i)
            f.write(f"{i+1}\n")
        f.write("done")

class BloomManager:
    @torch.no_grad()
    def forward(self):
        inputs = self.input_ids_list
        attens = [torch.zeros(input.shape[1]).bfloat16().to(self.device) if input is not None else None for input in inputs]

        print(f"loading attention mask ", end='')
        
        # 1. Create attention mask and position encodings
        # attention_mask = [torch.ones(
        #     len(ids), device=self.device).unsqueeze(0) if ids is not None else None for ids in inputs]
        # alibi = [build_alibi_tensor(mask, config.num_attention_heads,
        #                            torch.bfloat16).to(self.device) if mask is not None else None for mask in attention_mask]

        attention_mask = [torch.ones(
            len(ids), device=self.device).unsqueeze(0).bfloat16() if ids is not None else None for ids in inputs]
        alibi = [build_alibi_tensor(ids.shape[1], config.num_attention_heads,
                                   torch.bfloat16).to(self.device) if ids is not None else None for ids in inputs]

        recordTime("done")

        print(f"initializing hidden states... ", end='')
        # 2. Load and use word embeddings
        
        embeddings, lnorm = load_embeddings(self.device)

        hidden_states = [lnorm(embeddings(ids)) if ids is not None else None for ids in inputs]
        del embeddings
        del lnorm
        recordTime("done")

        # 3. Load and use the BLOOM blocks sequentially
        for block_num in range(0, self.blocks):
            load_block(self.block1, block_num)

            recordTime(f"loaded block {block_num}")
            for i in range(len(hidden_states)):
                if hidden_states[i] is not None:
                    out = self.block1(
                        hidden_states[i], attention_mask=attention_mask[i], alibi=alibi[i])
                    # out = self.block1(
                    #     hidden_states[i], attention_mask=attention_mask[i], alibi=alibi[i], output_attentions = True)
                    hidden_states[i] = out[0]
                    # self_attentions = out[1]
                    # self_attentions = torch.mean(self_attentions,0)
                    # self_attentions = torch.mean(self_attentions,0)
                    # self_attentions = self_attentions[-1]
                    # print(self_attentions)
                    # attens[i] = self_attentions + attens[i]
                    # (attens[i])
            recordTime(f"applied block {block_num}")

        del attention_mask
        del alibi

        final_lnorm = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_epsilon, dtype=torch.bfloat16).to(self.device)
        final_lnorm.load_state_dict(
            get_state_dict(shard_num=72, prefix="ln_f."))

        for i in range(len(hidden_states)):
            if hidden_states[i] is not None:
                hidden_states[i] = final_lnorm(hidden_states[i])

        recordTime(f"applied final lnorm")
        del final_lnorm

        lm_head = load_causal_lm_head(self.device)
        for i in range(len(hidden_states)):
            if hidden_states[i] is not None:
                hidden_states[i] = lm_head(hidden_states[i])
        del lm_head
        recordTime(f"applied lm head")

        # process or warp here
        self.attens = [att/len(hidden_states) if att is not None else None for att in attens]
        self.probs = [nn.functional.softmax(logits, dim=-1)[:,-1,:].cpu().sort(-1, True) if logits is not None else None for logits in hidden_states]

    @torch.no_grad()
    def add_input(self, tokens: str):
        seq = tokens.to(self.device)
        if len(self.empty_inputs) > 0:
            index = self.empty_inputs.pop()
            self.input_ids_list[index] = seq
            return index
        else:
            self.input_ids_list.append(seq)
            return len(self.input_ids_list)-1


    @torch.no_grad()
    def append_input(self, index, new_ids):
        if len(self.input_ids_list) > index and self.input_ids_list[index] is not None:
            self.input_ids_list[index] = torch.cat(
                [self.input_ids_list[index], new_ids.unsqueeze(0).to(self.device)], dim=-1)

    def delete_input(self, index):
        if len(self.input_ids_list) > index and self.input_ids_list[index] is not None:
            self.input_ids_list[index] = None
            self.empty_inputs.append(index)

    def get_probs(self, i: int, n: int = None):
        if n is None:
            n = 20
        if len(self.input_ids_list) > i and self.probs[i] is not None:
            return self.probs[i].values[:,:n]
    
    def get_tokens(self, i: int, n: int = None):
        if n is None:
            n = 20
        if len(self.input_ids_list) > i and self.probs[i] is not None:
            return self.probs[i].indices[:,:n]

    def get_attentions(self, i: int):
        if len(self.attens) > i and self.attens[i] is not None:
            return self.attens[i]


    @torch.no_grad()
    def export(self, index):
        if len(self.input_ids_list) > index and self.input_ids_list[index] is not None:
            return self.input_ids_list[index][0]

    @torch.no_grad()
    def __init__(self, gpu: int, blocks: int = 70):
        recordTime("initializing manager")
        self.blocks = blocks
        self.device = torch.device("cuda", gpu)
        self.input_ids_list = []
        self.empty_inputs = []
        self.attens = []

        self.block1 = BloomBlock(
            config, layer_number=1).bfloat16().to(self.device)

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("./bloom")
print("loaded tokenizer")

def tokenize(s: str):
    x = tokenizer.encode(s, return_tensors='pt')
    for i in x[0]:
        print(i, tokenizer.decode(i.item()))
    return x
    
# torch.set_printoptions(profile="full")
# bloom = BloomManager(1,2)
# s = "What number comes after five I think the answer is"
# bloom.add_input(tokenize(s))
# bloom.forward()