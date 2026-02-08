"""
Conversation memory management
"""
from langchain.memory import (
    ConversationBufferWindowMemory,
    ConversationSummaryMemory
)
import logging

logger = logging.getLogger(__name__)

class MultiLevelMemory:
    """
    Combines short-term and summary memory
    """
    
    def __init__(self, llm, window_size: int = 5):
        """
        Initialize multi-level memory
        
        Args:
            llm: Language model for summarization
            window_size: Number of recent exchanges to remember
        """
        logger.info(f"Initializing memory (window={window_size})...")
        
        # Short-term: Last N exchanges
        self.short_term = ConversationBufferWindowMemory(
            k=window_size,
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )
        
        # Long-term: Summary of entire conversation
        self.summary = ConversationSummaryMemory(
            llm=llm,
            memory_key="conversation_summary"
        )
        
        logger.info("✅ Memory initialized")
    
    def save_context(self, inputs: dict, outputs: dict):
        """Save conversation turn to both memories"""
        self.short_term.save_context(inputs, outputs)
        self.summary.save_context(inputs, outputs)
    
    def load_short_term(self) -> dict:
        """Load recent conversation history"""
        return self.short_term.load_memory_variables({})
    
    def load_summary(self) -> dict:
        """Load conversation summary"""
        return self.summary.load_memory_variables({})
    
    def clear(self):
        """Clear all memory"""
        self.short_term.clear()
        self.summary.clear()
        logger.info("🧹 Memory cleared")