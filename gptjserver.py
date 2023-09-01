from array import array
import asyncio
import pickle
import sys
import torch
from torch import nn

from anyio import to_thread

terminator = bytes([ord(p) for p in '\xde\xad\xbe\xef\xde\xad\xbe\xef'])


def _idgen():
    i = 0
    while True:
        yield i
        i += 1
idgen = _idgen()

def new_future():
    return asyncio.get_event_loop().create_future()

async def r(reader: asyncio.StreamReader):
    return pickle.loads(await reader.readuntil(terminator))
async def w(writer: asyncio.StreamWriter, data):
    writer.write(pickle.dumps(data) + terminator)
    #print(len(pickle.dumps(data) + terminator))
    return await writer.drain()


from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")

print("tokenizer ready")

sys.setrecursionlimit(10000)

class Manager:
    def __init__(self, gpu: int):
        self.device = torch.device("cuda", gpu)
        self.model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-j-6B", torch_dtype=torch.float16).to(self.device)
        print(f"Manager {gpu} ready!")
    def forward(self, input_ids):
        input_ids = input_ids.to(self.device)
        # hidden_states = self.model.transformer(input_ids)[0]
        # logits = self.model.lm_head(hidden_states).to(torch.float32)
        # next_token_logits = logits[:, -1, :].cpu()
        # return next_token_logits
        return self.model.generate(
            input_ids,
            do_sample=False,
            max_new_tokens=1,
            pad_token_id=tokenizer.eos_token_id,
            return_dict_in_generate=True,
            output_scores=True
        ).scores[-1].cpu()

class DummyManager:
    def __init__(self, gpu: int):
        pass
    def forward(self, input_ids):
        return torch.rand((input_ids.shape[0], 50400))

import os
def open_file(file_path, opt):
    directory = os.path.dirname(file_path)
    if len(directory) > 0 and not os.path.exists(directory):
        os.makedirs(directory)
    return open(file_path, opt)

def log(s, fname = "log.log"):
    with open_file(fname, "a") as f:
        f.write(f"{str(s)}\n")

class GPTJServer:

    async def tick(self):
        TOP_K = 64

        self.ready.sort(reverse=True, key=lambda id: len(self.inputs[id]) if id in self.inputs else -1)
        
        while len(self.ready) and self.ready[-1] not in self.inputs:
            self.inputs.pop()
        
        if len(self.ready) == 0:
            return

        #batch: 64 - 128 -> 4 2048
        ids = []

        n = len(self.inputs[self.ready[-1]])
        if n >= 2048:
            self.ready = []
            return
        batch_max = 4 * 2048 // n

        while len(self.ready) and len(ids) < batch_max and len(self.inputs[self.ready[-1]]) == n:
            ids.append(self.ready.pop())
        
        # print(f"clearing {len(ids)} items, L = {n}, capacity = {batch_max}")
        if len(ids) == 0:
            return

        input_ids = torch.stack([self.inputs[id] for id in ids], 0)
        
        logits = await to_thread.run_sync(self.model.forward, input_ids)
        values, indices = nn.functional.softmax(logits, dim=-1).cpu().sort(-1, True)

        for i, id in enumerate(ids):
            self.futures[id].set_result((values[i][:TOP_K].detach().clone(), indices[i][:TOP_K].detach().clone()))


    async def handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        id = next(idgen)
        self.futures[id] = new_future()
        try:
            while True:
                command, payload = await r(reader)
                #print(command, payload)
                if command == "FORWARD":
                    if self.inputs[id] is not None and not id in self.ready:
                        self.futures[id] = new_future()
                        self.ready.append(id)
                        await self.futures[id]
                    await w(writer, True)
                elif command == "ADD":
                    payload = payload[0]
                    self.inputs[id] = payload
                    await w(writer, id)
                elif command == "APPEND":
                    if self.inputs[id] is not None:
                        self.inputs[id] = torch.cat([self.inputs[id], payload], dim=-1)
                    else:
                        self.inputs[id] = payload
                    await w(writer, True)
                elif command == "PROBS":
                    v, i = await self.futures[id]
                    await w(writer, v.unsqueeze(0))
                elif command == "TOKENS":
                    v, i = await self.futures[id]
                    await w(writer, i.unsqueeze(0))
                elif command == "EXPORT":
                    await w(writer, self.inputs[id].unsqueeze(0))
                elif command == "CLOSE":
                    await w(writer, True)
                    break
                elif command == "EXIT":
                    sys.exit(0)
        except Exception as e:
            # print(f"client {id} is over")
            # log(f"client {id} is over: {e}")
            pass
        finally:
            writer.close()
            if id in self.inputs:
                del self.inputs[id]
            if id in self.futures:
                del self.futures[id]


    async def listen(self, port):
        server = await asyncio.start_server(lambda reader, writer: self.handle(reader, writer), "localhost", port, limit=1024*1024)
        addrs = ", ".join(str(sock.getsockname()) for sock in server.sockets)
        print(f"serving on {addrs}")
        async with server:
            await server.serve_forever()

    async def tickloop(self):
        while True:
            await self.tick()
            await asyncio.sleep(0.1)

    def __init__(self, model: Manager):
        self.model = model
        self.idgen = _idgen()
        self.ready = []

        self.inputs = dict()
        self.futures = dict()


async def servify(model, port):
    serv = GPTJServer(model)
    await asyncio.gather(serv.listen(port), serv.tickloop())

async def main(man=Manager):
    gpus = [int(i) for i in sys.argv[1:]]
    models = [man(gpu) for gpu in gpus]

    print("listening")
    await asyncio.gather(*[servify(model, 1330+i) for i, model in enumerate(models)])

asyncio.run(main(Manager))