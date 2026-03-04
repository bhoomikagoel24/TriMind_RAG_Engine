from typing import List
from langchain_core.documents import Document


def evaluate_retrieval(query: str, docs: List[Document]):

    print("\n--- Retrieval Evaluation ---")

    print(f"Query: {query}")
    print(f"Documents retrieved: {len(docs)}")

    unique_sources = set()
    for doc in docs:
        unique_sources.add(doc.metadata.get("source"))

    print(f"Unique sources: {len(unique_sources)}")

    for i, doc in enumerate(docs[:3], 1):
        print(f"\nDoc {i} Source: {doc.metadata.get('source')}")
        print("Snippet:", doc.page_content[:200])

# GROUNDING Score
# It checks how much of answer content overlaps with retrieved documents.
# If overlap low -> possible hallucination.

def compute_grounding_score(answer: str, docs):

    if not docs:
        print("Grounding Score: 0 (No documents)")
        return 0

    answer_lower = answer.lower()
    match_count = 0

    for doc in docs:
        snippet = doc.page_content[:300].lower()
        if snippet in answer_lower:
            match_count += 1

    score = match_count / len(docs)

    print(f"Grounding Score: {round(score, 2)}")

    if score < 0.2:
        print("⚠ Possible hallucination detected")

    return score

# Compute Confidence Score 
# it depends on source diversity and answer completeness
def compute_confidence_score(answer: str, docs):

    if not docs:
        return 0.0

    unique_sources = len(set(
        doc.metadata.get("source", "")
        for doc in docs
    ))

    diversity_score = unique_sources / len(docs)

    answer_length_score = min(len(answer.split()) / 50, 1)

    confidence = round((diversity_score * 0.6) + (answer_length_score * 0.4), 2)

    print(f"Confidence Score: {confidence}")

    return confidence