from fastapi import FastAPI
import asyncio
import sys
from config import APIConfig

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
import time
import zmq.asyncio, zmq

app = FastAPI()


async def generate(msg):
    ctx = zmq.asyncio.Context()
    sock = ctx.socket(zmq.DEALER)
    sock.connect(APIConfig.API_CONNECT)
    await sock.send_multipart([msg.encode("utf-8")])
    response = await sock.recv_multipart()
    print(response)
    return response[0].decode("utf-8")


@app.post("/ask")
async def prompt(input: str):
    start_time = time.time()
    # send to controller
    msg = await generate(input)
    end_time = time.time()
    return {
        "result": msg,
        "start_time": start_time,
        "end_time": end_time,
        "total_time": end_time - start_time,
    }
