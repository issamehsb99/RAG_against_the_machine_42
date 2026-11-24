from . import indexer, tf_idf
from rank_bm25 import BM25Okapi
from pathlib import Path
from .data_models import (
    RagDataset,
    MinimalSource,
    StudentSearchResults,
    MinimalSearchResults,
)
import numpy as np
from . import retriever


def search_dataset(dataset_path, k=5):
    dataset_path = Path(dataset_path)

    with open(dataset_path, "r") as file:
        data = file.read()

    rag_dataset = RagDataset.model_validate_json(data)

    results = []
    chunks = indexer.Chunker("data/raw/vllm-0.10.1")
    tokens = indexer.Tokenizer(chunks.raw_text)
    corpus = tokens.tokens
    counts = tokens.counts
    bm25 = retriever.BM25(corpus, counts)

    for item in rag_dataset.rag_questions:
        query_tokens = indexer.Tokenizer([item.question]).tokens[0]
        scores = bm25.bm25(query_tokens)
        hits = sorted(zip(chunks.chunks, scores), key=lambda x: x[1], reverse=True)[:k]
        sources = [chunk for chunk, _ in hits]
        msr = MinimalSearchResults(
            question_id=item.question_id,
            question=item.question,
            retrieved_sources=sources,
        )
        results.append(msr)

    student_search = StudentSearchResults(search_results=results, k=k)
    output_file = "results.json"

    with open(output_file, "w") as file:
        file.write(student_search.model_dump_json(indent=2))


search_dataset("private/UnansweredQuestions/dataset_code_private.json")

../vllm-0.10.1/vllm/third_party/pynvml.py