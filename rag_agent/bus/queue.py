"""Async message queue for decoupled channel-agent communication."""

import asyncio

from rag_agent.bus.events import InboundMessage, OutboundMessage


class MessageBus:
    """Two-queue bus: inbound (channels -> agent), outbound (agent -> channels)."""

    def __init__(self):
        self.inbound: asyncio.Queue[InboundMessage] = asyncio.Queue()
        self.outbound: asyncio.Queue[OutboundMessage] = asyncio.Queue()

    async def publish_inbound(self, msg: InboundMessage) -> None:
        """Publish one inbound message for agent consumption."""
        await self.inbound.put(msg)

    async def consume_inbound(self) -> InboundMessage:
        """Consume the next inbound message (blocks until available)."""
        return await self.inbound.get()

    async def publish_outbound(self, msg: OutboundMessage) -> None:
        """Publish one outbound message for channel delivery."""
        await self.outbound.put(msg)

    async def consume_outbound(self) -> OutboundMessage:
        """Consume the next outbound message (blocks until available)."""
        return await self.outbound.get()

    @property
    def inbound_size(self) -> int:
        """Current inbound queue size."""
        return self.inbound.qsize()

    @property
    def outbound_size(self) -> int:
        """Current outbound queue size."""
        return self.outbound.qsize()
