"""Base channel interface for chat platforms."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from loguru import logger

from rag_agent.bus.events import InboundMessage, OutboundMessage
from rag_agent.bus.queue import MessageBus


class BaseChannel(ABC):
    """Abstract interface that all chat channels implement."""

    name: str = "base"
    display_name: str = "Base"
    transcription_api_key: str = ""

    def __init__(self, config: Any, bus: MessageBus):
        self.config = config
        self.bus = bus
        self._running = False

    @abstractmethod
    async def start(self) -> None:
        """Start channel listener loop."""
        raise NotImplementedError

    @abstractmethod
    async def stop(self) -> None:
        """Stop channel and clean resources."""
        raise NotImplementedError

    @abstractmethod
    async def send(self, msg: OutboundMessage) -> None:
        """Deliver one outbound message via this channel."""
        raise NotImplementedError

    def is_allowed(self, sender_id: str) -> bool:
        """Check sender permissions using allow_from semantics."""
        allow_list = getattr(self.config, "allow_from", [])
        if not allow_list:
            logger.warning("{}: allow_from is empty - all access denied", self.name)
            return False
        if "*" in allow_list:
            return True
        return str(sender_id) in allow_list

    async def _handle_message(
        self,
        sender_id: str,
        chat_id: str,
        content: str,
        media: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        session_key: str | None = None,
    ) -> None:
        """Validate and route one inbound message to the bus."""
        if not self.is_allowed(sender_id):
            logger.warning(
                "Access denied for sender {} on channel {}. Add sender to allow_from to grant access.",
                sender_id,
                self.name,
            )
            return

        msg = InboundMessage(
            channel=self.name,
            sender_id=str(sender_id),
            chat_id=str(chat_id),
            content=content,
            media=media or [],
            metadata=metadata or {},
            session_key_override=session_key,
        )
        await self.bus.publish_inbound(msg)

    @classmethod
    def default_config(cls) -> dict[str, Any]:
        """Return default config payload for this channel."""
        return {"enabled": False}

    @property
    def is_running(self) -> bool:
        """Channel running flag."""
        return self._running
