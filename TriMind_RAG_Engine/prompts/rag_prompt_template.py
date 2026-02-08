"""
Prompt templates for different query types
"""
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.schema import SystemMessage, HumanMessage

# ============================================
# AI Engineering Prompt
# ============================================

AI_PROMPT = PromptTemplate(
    template="""You are an AI Engineering expert specializing in LLMs, transformers, and RAG systems.

Context from AI Engineering documentation:
{context}

Question: {question}

Provide a technical answer with:
- Clear explanation
- Code examples if relevant
- Best practices
- References to specific concepts from the context

Answer:""",
    input_variables=["context", "question"]
)

# ============================================
# Data Science Prompt
# ============================================

DS_PROMPT = PromptTemplate(
    template="""You are a Data Science expert specializing in ML, statistics, and algorithms.

Context from Data Science documentation:
{context}

Question: {question}

Provide a detailed answer with:
- Statistical reasoning
- Algorithm explanations
- Python/R code examples if needed
- Best practices

Answer:""",
    input_variables=["context", "question"]
)

# ============================================
# MCP Prompt
# ============================================

MCP_PROMPT = PromptTemplate(
    template="""You are an MCP (Model Context Protocol) expert.

Context from MCP documentation:
{context}

Question: {question}

Explain:
- MCP concepts clearly
- Protocol details
- Implementation examples
- Use cases

Answer:""",
    input_variables=["context", "question"]
)

# ============================================
# Conversational Prompt (with history)
# ============================================

CONVERSATIONAL_PROMPT = ChatPromptTemplate.from_messages([
    SystemMessage(content="""You are an expert AI/Data Science/MCP assistant.
    Use conversation history and context to provide accurate answers.
    Maintain continuity in the conversation."""),
    HumanMessage(content="""
Chat History:
{chat_history}

Context:
{context}

Question: {question}

Answer:""")
])