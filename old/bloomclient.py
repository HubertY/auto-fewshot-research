import asyncio
import pickle
import torch

from transformers import AutoTokenizer


terminator = bytes([ord(p) for p in '\xde\xad\xbe\xef\xde\xad\xbe\xef'])


model_path = "./bloom"  # replace with your local folder path
tokenizer = AutoTokenizer.from_pretrained(model_path)
print("loaded tokenizer")

def tokenize(s: str):
    return tokenizer.encode(s, return_tensors='pt')

def detokenize(data) -> str:
    return tokenizer.decode(data)

class BloomClient:
    async def send(self, command: str, payload = None):
        print(command, payload)
        dump = pickle.dumps((command, payload))
        self.writer.write(dump + terminator)
        await self.writer.drain()

        data = await self.reader.readuntil(terminator)
        return pickle.loads(data)

    def close(self):
        self.writer.close()
    async def connect(self,port):
        self.reader, self.writer = await asyncio.open_connection(
        "localhost", port, limit=9999999)
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
async def main():
    cli = BloomClient()
    await cli.connect(1330)
    await cli.send("ADD", tokenize(sentence))
    for i in range(100):
        await cli.send("FORWARD")
        tokens = await cli.send("TOKENS")
        print(f'token id {tokens[:,0]} = "{detokenize(tokens[:,0])}"')
        await cli.send("APPEND", tokens[:,0])
    tokens = await cli.send("EXPORT")
    with open("ids.out", "w") as f:
        f.write(pickle.dumps(tokens))

    s = detokenize(tokens)
    with open("str.out", "w") as f:
        f.write(s)
    
    print(s)
    await cli.send("CLOSE")
    cli.close()


#run this to kill the server
async def terminate(p):
    cli = BloomClient()
    await cli.connect(1330)
    await cli.send("EXIT")
    cli.close()

#asyncio.run(terminate(1337))



