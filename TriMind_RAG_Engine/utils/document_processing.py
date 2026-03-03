# Retriever → Clean → Deduplicate → Reorder → LLM
import re
from typing import List
from langchain_core.documents import Document
from langchain_community.document_transformers import LongContextReorder


# =========================================================
# 1️⃣ REMOVE DUPLICATE SOURCES
# =========================================================

def deduplicate_sources(docs: List[Document]) -> List[Document]:
    """
    Remove duplicate documents based on (source, page)
    """

    seen = set()
    unique_docs = []

    for doc in docs:
        # key = (
        #     doc.metadata.get("source"),
        #     doc.metadata.get("page")
        # )
        key = hash(doc.page_content.strip())

        if key not in seen:
            seen.add(key)
            unique_docs.append(doc)

    return unique_docs


# =========================================================
# 2️⃣ REMOVE TABLE OF CONTENTS / INDEX PAGES
# =========================================================

def filter_toc_pages(docs: List[Document]) -> List[Document]:
    """
    Remove TOC / index-like pages.
    """

    filtered = []

    for doc in docs:
        content = doc.page_content
        content_lower = content.lower()

        # Heuristic patterns
        has_many_dots = content.count("....") >= 3
        has_many_newlines = content.count("\n") > len(content) / 30
        has_toc_keywords = any(
            k in content_lower for k in [
                "table of contents",
                "index",
                "contents",
                "chapter list"
            ]
        )

        has_toc_pattern = bool(
            re.search(r"\d+\.\d+\).*\.{3,}", content)
        )

        is_toc = (
            len(content.split()) < 200 and
            (has_many_dots or has_many_newlines or has_toc_keywords or has_toc_pattern)
        )

        if not is_toc:
            filtered.append(doc)

    return filtered


# =========================================================
# 3️⃣ REORDER DOCUMENTS FOR BETTER CONTEXT
# =========================================================

def reorder_documents(docs: List[Document]) -> List[Document]:
    """
    Reorder documents so most relevant appear
    at beginning and end (helps LLM attention)
    """

    try:
        reorderer = LongContextReorder()
        return reorderer.transform_documents(docs)
    except Exception:
        return docs


# =========================================================
# 4️⃣ FULL PROCESSING PIPELINE
# =========================================================

def process_documents(query: str, docs: List[Document]) -> List[Document]:
    """
    Full post-retrieval processing:
    1. Remove TOC pages
    2. Remove duplicates
    3. Reorder for optimal LLM attention
    """
    print(f"After cleaning: {len(docs)} documents")
    # Step 1: Remove TOC
    docs = filter_toc_pages(docs)

    # Step 2: Remove duplicates
    docs = deduplicate_sources(docs)

    # Step 3: Reorder
    docs = reorder_documents(docs)

    return docs
