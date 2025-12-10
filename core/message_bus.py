
import asyncio
import uuid
from datetime import datetime
from typing import Dict, Callable, Any, Optional, List
from dataclasses import dataclass, field
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class Message:
    """Standard message format for agent communication"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    type: str = "notification"  # request, response, notification, error
    sender: str = ""
    recipient: Optional[str] = None  # None means broadcast
    topic: str = ""
    payload: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    correlation_id: Optional[str] = None  # Links related messages
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary"""
        return {
            "id": self.id,
            "type": self.type,
            "sender": self.sender,
            "recipient": self.recipient,
            "topic": self.topic,
            "payload": self.payload,
            "timestamp": self.timestamp,
            "correlation_id": self.correlation_id
        }


class MessageBus:
    """
    Central message bus for agent communication.
    Implements publish-subscribe pattern with async message delivery.
    """
    
    def __init__(self):
        self._subscriptions: Dict[str, Dict[str, Callable]] = {}
        
        self._agent_queues: Dict[str, asyncio.Queue] = {}
        
        self._agent_tasks: Dict[str, asyncio.Task] = {}
        
        self._message_history: List[Message] = []
        self._max_history = 100
        
        logger.info("MessageBus initialized")
    
    
    def subscribe(self, topic: str, agent_id: str, callback: Callable):
        """
        Subscribe an agent to a topic.
        
        Args:
            topic: Message topic to subscribe to
            agent_id: Unique identifier for the agent
            callback: Async function to call when message arrives
        """
        if topic not in self._subscriptions:
            self._subscriptions[topic] = {}
        
        self._subscriptions[topic][agent_id] = callback
        
        if agent_id not in self._agent_queues:
            self._agent_queues[agent_id] = asyncio.Queue()
            self._agent_tasks[agent_id] = asyncio.create_task(
                self._process_agent_queue(agent_id)
            )
        
        logger.info(f"Agent '{agent_id}' subscribed to topic '{topic}'")
    
    
    def unsubscribe(self, topic: str, agent_id: str):
        """Unsubscribe an agent from a topic"""
        if topic in self._subscriptions and agent_id in self._subscriptions[topic]:
            del self._subscriptions[topic][agent_id]
            logger.info(f"Agent '{agent_id}' unsubscribed from topic '{topic}'")
    
    
    async def publish(self, message: Message):
        """
        Publish a message to all subscribers of its topic.
        
        Args:
            message: Message object to publish
        """
        logger.info(
            f"Publishing message: {message.type} | "
            f"Topic: {message.topic} | "
            f"From: {message.sender} | "
            f"To: {message.recipient or 'broadcast'}"
        )
        
        self._add_to_history(message)
        
        subscribers = self._subscriptions.get(message.topic, {})
        
        if not subscribers:
            logger.warning(f"No subscribers for topic '{message.topic}'")
            return
        
        for agent_id, callback in subscribers.items():
            if message.recipient and message.recipient != agent_id:
                continue
            
            await self._agent_queues[agent_id].put((message, callback))
            logger.debug(f"Message queued for agent '{agent_id}'")
    
    
    async def _process_agent_queue(self, agent_id: str):
        """
        Process messages from an agent's queue.
        Runs continuously in background.
        """
        queue = self._agent_queues[agent_id]
        
        while True:
            try:
                message, callback = await queue.get()
                
                try:
                    await callback(message)
                    logger.debug(f"Agent '{agent_id}' processed message {message.id}")
                except Exception as e:
                    logger.error(
                        f"Error processing message in agent '{agent_id}': {e}",
                        exc_info=True
                    )
                    error_msg = Message(
                        type="error",
                        sender="message_bus",
                        topic="system_error",
                        payload={
                            "error": str(e),
                            "agent_id": agent_id,
                            "original_message_id": message.id
                        }
                    )
                    await self.publish(error_msg)
                
                queue.task_done()
                
            except asyncio.CancelledError:
                logger.info(f"Message processing cancelled for agent '{agent_id}'")
                break
            except Exception as e:
                logger.error(
                    f"Unexpected error in agent queue '{agent_id}': {e}",
                    exc_info=True
                )
    
    
    def _add_to_history(self, message: Message):
        """Add message to history, maintaining max size"""
        self._message_history.append(message)
        if len(self._message_history) > self._max_history:
            self._message_history.pop(0)
    
    
    def get_message_history(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get recent message history"""
        return [msg.to_dict() for msg in self._message_history[-limit:]]
    
    
    async def shutdown(self):
        """Gracefully shutdown the message bus"""
        logger.info("Shutting down MessageBus...")
        
        for agent_id, task in self._agent_tasks.items():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        
        logger.info("MessageBus shutdown complete")