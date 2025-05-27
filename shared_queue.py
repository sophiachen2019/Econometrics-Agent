# shared_queue.py
import asyncio
from typing import Dict

# 使用字典存储每个用户的消息队列
message_queues: Dict[str, asyncio.Queue] = {}

async def get_or_create_queue(user_id: str) -> asyncio.Queue:
    """获取或创建用户专属的消息队列"""
    if user_id not in message_queues:
        message_queues[user_id] = asyncio.Queue()
    return message_queues[user_id]

async def log_execution(message: str, user_id: str):
    """将消息写入指定用户的队列"""
    queue = await get_or_create_queue(user_id)
    await queue.put(message)

async def get_message(user_id: str):
    """从指定用户的队列获取消息"""
    queue = await get_or_create_queue(user_id)
    return await queue.get()

def queue_empty(user_id: str) -> bool:
    """检查指定用户的队列是否为空"""
    if user_id not in message_queues:
        return True
    return message_queues[user_id].empty()

def cleanup_queue(user_id: str):
    """清理用户的消息队列"""
    if user_id in message_queues:
        del message_queues[user_id]
