# shared_queue.py
import asyncio

message_queue = asyncio.Queue()

async def log_execution(message):
    await message_queue.put(message)

async def get_message():
    return await message_queue.get()

def queue_empty():
    return message_queue.empty()
