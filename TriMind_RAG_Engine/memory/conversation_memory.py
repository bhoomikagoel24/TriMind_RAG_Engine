from langchain_classic.memory import ConversationBufferWindowMemory


def get_conversation_memory(window_size: int = 5):
    return ConversationBufferWindowMemory(
        k=window_size,
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"
    )