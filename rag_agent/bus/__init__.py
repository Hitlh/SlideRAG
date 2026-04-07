"""Message bus primitives for channel-agent decoupling."""

from .events import InboundMessage, OutboundMessage
from .queue import MessageBus

__all__ = ["InboundMessage", "OutboundMessage", "MessageBus"]
