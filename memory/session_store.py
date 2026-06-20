"""多会话对话记忆存储"""
from typing import Dict

from memory.rag_memory import ConversationMemory


class SessionStore:
    """按 session_id 隔离对话历史。"""

    def __init__(self):
        self._sessions: Dict[str, ConversationMemory] = {}

    def get(self, session_id: str) -> ConversationMemory:
        if session_id not in self._sessions:
            self._sessions[session_id] = ConversationMemory()
        return self._sessions[session_id]

    def clear(self, session_id: str) -> None:
        if session_id in self._sessions:
            self._sessions[session_id].clear()

    def count(self) -> int:
        return len(self._sessions)
