import asyncio
import pickle
import torch

from transformers import AutoTokenizer


terminator = bytes([ord(p) for p in '\xde\xad\xbe\xef\xde\xad\xbe\xef'])


model_path = "EleutherAI/gpt-j-6B"
tokenizer = AutoTokenizer.from_pretrained(model_path)
print("loaded tokenizer")

def tokenize(s: str):
    return tokenizer.encode(s, return_tensors='pt')

def detokenize(data) -> str:
    return tokenizer.decode(data)

class Client:
    async def send(self, command: str, payload = None):
        #print(command, payload)
        dump = pickle.dumps((command, payload))
        self.writer.write(dump + terminator)
        await self.writer.drain()

        data = await self.reader.readuntil(terminator)
        return pickle.loads(data)

    def close(self):
        self.writer.close()
    async def connect(self,port):
        self.reader, self.writer = await asyncio.open_connection(
        "localhost", port, limit=1024*1024)
    def __init__(self):
        self.reader = None
        self.writer = None

sentence = """Question: Good Samaritan laws offer legal protection to people who give reasonable assistance to those who are, or who they believe to be, injured, ill, in peril, or otherwise incapacitated. The protection is intended to reduce bystanders' hesitation to assist, for fear of being sued or prosecuted for unintentional injury or wrongful death. An example of such a law in common-law areas of Canada: a good Samaritan doctrine is a legal principle that prevents a rescuer who has voluntarily helped a victim in distress from being successfully sued for wrongdoing. Its purpose is to keep people from being reluctant to help a stranger in need for fear of legal repercussions should they make some mistake in treatment. By contrast, a duty to rescue law requires people to offer assistance and holds those who fail to do so liable.
According to the passage, do good samaritan laws protect those who help at an accident?
A: True
B: False
Answer:
"""

"""
WARNING: the server will block until all connected clients have called FORWARD.
you must send CLOSE or otherwise disconnect when done or else you will stall the server forever.
server should be robust to everything, probably.

Most commands will block until the server has finished its current forward pass, so the first ADD might take a minute.
"""
async def clitest(n = 100):
    s = " test" * n
    cli = Client()
    await cli.connect(1330)
    await cli.send("ADD", tokenize(s))
    for i in range(10):
        await cli.send("FORWARD")
        tokens = await cli.send("TOKENS")
        print(f'token id {tokens[:,0]} = "{detokenize(tokens[:,0])}"')
        await cli.send("APPEND", tokens[:,0])
    tokens = await cli.send("EXPORT")

    s = detokenize(tokens[0])
    print(tokens)    
    print(s)
    await cli.send("CLOSE")
    cli.close()


import random
async def main():
    await asyncio.gather(*[clitest(random.randint(50,80)) for i in range(1000)])

#run this to kill the server
async def terminate(p=1330):
    cli = Client()
    await cli.connect(p)
    await cli.send("EXIT")
    cli.close()

# asyncio.run(main())



