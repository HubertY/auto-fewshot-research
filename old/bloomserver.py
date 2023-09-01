from array import array
import asyncio
import pickle
import sys
import torch
from anyio import to_thread


terminator = bytes([ord(p) for p in '\xde\xad\xbe\xef\xde\xad\xbe\xef'])

print("loading bigbloom module")
import bigbloom

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
    return await writer.drain()

class BloomServer:
    model: bigbloom.BloomManager
    readyup: dict
    async def tick(self):
        await asyncio.gather(*self.readyup.values())
        self.readyup = {k: new_future() for k in self.readyup}
        self.forward_run = new_future()
        old_fstart = self.forward_start
        self.forward_start = new_future()
        old_fstart.set_result(True)
        await self.FORWARD()
        self.forward_run.set_result(True)

    # async def ADD_INPUT(self, s: str):
    #     return await to_thread.run_sync(self.model.add_input, s)
    async def FORWARD(self):
        return await to_thread.run_sync(self.model.forward)
    # async def DELETE(self, location):
    #     index = location
    #     return await to_thread.run_sync(self.model.delete_input(index))

    async def handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        id = next(idgen)
        self.readyup[id] = new_future()
        try:
            while True:
                command, payload = await r(reader)
                print(command, payload)
                if command == "FORWARD":
                    await self.forward_run #wait for current forward to end
                    self.readyup[id].set_result(True)
                    await self.forward_start #wait for next forward to start
                    await self.forward_run #wait for next forward to end
                    await w(writer, True)
                elif command == "ADD":
                    await self.forward_run
                    if id in self.processmap:
                        self.model.delete_input(self.processmap[id])
                    self.processmap[id] = self.model.add_input(payload)
                    await w(writer, self.processmap[id])
                elif command == "APPEND":
                    if id in self.processmap:
                        index = self.processmap[id]
                        await self.forward_run
                        self.model.append_input(index, payload)
                    await w(writer, True)
                elif command == "PROBS":
                    ret = None
                    if id in self.processmap:
                        index = self.processmap[id]
                        await self.forward_run
                        ret = self.model.get_probs(index, payload)
                    await w(writer, ret)
                elif command == "TOKENS":
                    ret = None
                    if id in self.processmap:
                        index = self.processmap[id]
                        await self.forward_run
                        ret = self.model.get_tokens(index, payload)
                    await w(writer, ret)
                elif command == "ATTENTIONS":
                    ret = None
                    if id in self.processmap:
                        index = self.processmap[id]
                        await self.forward_run
                        ret = self.model.get_attentions(index)
                    await w(writer, ret)
                elif command == "EXPORT":
                    ret = None
                    if id in self.processmap:
                        index = self.processmap[id]
                        await self.forward_run
                        ret = self.model.export(index)
                    await w(writer, ret)
                elif command == "CLOSE":
                    await w(writer, True)
                    break
                elif command == "EXIT":
                    sys.exit(0)
        except Exception as e:
            print(f"client {id} is over: {e}")

        finally:
            writer.close()
            await self.forward_run
            if id in self.processmap:
                self.model.delete_input(self.processmap[id])
                del self.processmap[id]
            ready = self.readyup[id]
            del self.readyup[id]
            ready.set_result(True)
            
    async def listen(self, port):
        server = await asyncio.start_server(lambda reader, writer: self.handle(reader, writer), "localhost", port, limit=9999999)
        addrs = ", ".join(str(sock.getsockname()) for sock in server.sockets)
        print(f"serving on {addrs}")
        async with server:
            await server.serve_forever()

    async def tickloop(self):
        while True:
            if len(self.processmap):
                await self.tick()
            await asyncio.sleep(0.1)

    def __init__(self, model: bigbloom.BloomManager):
        self.model = model
        self.idgen = _idgen()
        self.processmap = dict()
        self.readyup = dict()
        self.commandqueue = []
        self.forward_run = new_future() #resolved when forward has ended
        self.forward_run.set_result(True)
        self.forward_start = new_future() #resolved the next time forward starts

# change this to a lower number for testing
BLOCKS = 70
async def servify(bloom, port):
    serv = BloomServer(bloom)
    await asyncio.gather(serv.listen(port), serv.tickloop())

async def main():
    gpus = [int(i) for i in sys.argv[1:]]
    blooms = [bigbloom.BloomManager(gpu, BLOCKS) for gpu in gpus]
    
    print("preloading bloom blocks to cpu memory (this will take ~20 minutes)")
    bigbloom.prefetch(BLOCKS)
    print("listening")
    await asyncio.gather(*[servify(bloom, 1330+i) for i, bloom in enumerate(blooms)])

asyncio.run(main())