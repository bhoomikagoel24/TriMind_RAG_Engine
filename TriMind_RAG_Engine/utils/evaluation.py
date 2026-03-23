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
        return 0

    answer_words = set(answer.lower().split())
    doc_words = set()

    for doc in docs:
        doc_words.update(doc.page_content.lower().split())

    overlap = answer_words.intersection(doc_words)

    score = len(overlap) / (len(answer_words) + 1)

    print(f"Grounding Score: {round(score, 2)}")

    if score < 0.2:
        print("⚠ Possible hallucination detected")

    return score


# Compute Confidence Score 
# it depends on source diversity and answer completeness
def compute_confidence_score(answer: str, docs, grounding_score=0):

    if not docs:
        return 0.0

    unique_sources = len(set(doc.metadata.get("source", "") for doc in docs))
    diversity_score = unique_sources / len(docs)

    answer_length_score = min(len(answer.split()) / 50, 1)

    # include grounding
    confidence = round(
        (diversity_score * 0.5) +
        (answer_length_score * 0.3) +
        (grounding_score * 0.2),
        2
    )

    print(f"Confidence Score: {confidence}")

    return confidence
