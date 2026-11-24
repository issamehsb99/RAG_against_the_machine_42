from pathlib import Path
from pydantic import BaseModel, Field
from ast import parse, get_source_segment
from json import dump, load
from typing import List
documents_path = {'python': list(), 'text': list()}
chunks = {'python': list(), 'text': list()}
stop_words = {
    "a",
    "an",
    "the",
    "and",
    "or",
    "but",
    "if",
    "then",
    "is",
    "am",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "do",
    "does",
    "did",
    "of",
    "to",
    "in",
    "on",
    "at",
    "by",
    "for",
    "from",
    "with",
    "about",
    "into",
    "through",
    "during",
    "this",
    "that",
    "these",
    "those",
    "it",
    "its",
    "as",
    "not",
    "so",
    "than",
    "too",
    "what",
    "how",
    "where",
    "which",
    "why",
    "who",
    "when",
    "can",
    "could"
}
size = 2000
class Source(BaseModel):
    fp: str
    content: str
    s: int
    e: int
    cid: int
def get_paths():
    repo = Path('vllm-0.10.1')
    for path in repo.rglob('*'):
        if path.is_file() and path.suffix == '.py':
            documents_path['python'].append(path)
        elif path.is_file() and (path.suffix == '.md' or path.suffix == '.txt'):
            documents_path['text'].append(path)
def save_chunk(s, e, fp, content, mode = 'python'):
    global N
    chunks[mode].append(Source(s=s, e=e, fp=str(fp), content=content[s:e], cid = N))
    N += 1
def text_chunking_strategy(paths: list[str]):
    overlap = size - min(600, size-1)
    for path in paths:
        with open(path) as f:
            content = f.read()

        start = 0
        content_len = len(content)
        while start < content_len:
            end = min(start+size,content_len)
            save_chunk(start, end, path, content, mode='text')
            if end >= content_len:
                break
            start += overlap
def python_chunking_strategy(paths: list[str]):
    for path in paths:
        with open(path) as f:
            code = f.read()
        try:
            tree = parse(code)
        except SyntaxError:
            pass
        start = None
        end = 0
        for node in tree.body:
            segment = get_source_segment(code, node)
            node_start = code.find(segment, end)
            node_end = node_start + len(segment)
            if node_end - node_start >= size:
                if start is not None:
                    save_chunk(start, end,path, code)
                    start = None
                for s in range(node_start, node_end, size):
                    save_chunk(s, min(node_end, s+size),path, code)
            elif start is None:
                start = node_start
            elif node_end - start >= size:
                save_chunk(start, end, path, code)
                start = node_start
            end = node_end
        if start is not None:
            save_chunk(start, end, path, code)
# get_paths()
# text_chunking_strategy(documents_path['text'])
# python_chunking_strategy(documents_path['python'])
# all_chunks = [chunk.model_dump() for mod in chunks for chunk in chunks[mod]]

# with open("chunks_database.json", "w") as f:
#     dump(all_chunks, f, indent=4)
def tokenization(chunks: dict[str: list, str:list] = None, query = None):
    from string import punctuation
    to_remove = set(stop_words)
    table = str.maketrans(punctuation, " " * len(punctuation))
    if query:
        return [term.lower() for term in query.translate(table).split() if len(term) > 1 and term not in stop_words]
    results = []
    for mod in chunks:
        is_py = mod == 'python'
        for chunk in chunks[mod]:
            semi_cleaned_text = chunk.content.translate(table)
            medium_cleaned_text = " ".join([word.lower() for word in semi_cleaned_text.split() if len(word) > 1 and (is_py or word not in to_remove)])
            results.append(chunk.model_copy(update={'content': medium_cleaned_text}))
    return results

# to_index = tokenization(chunks)

def index(to_index):
    inverted_index = {}
    for chunk in to_index:
        terms = chunk.content.split()
        for term in terms:
            if term not in inverted_index:
                inverted_index[term] = {}
            if chunk.cid not in inverted_index[term]:
                inverted_index[term][chunk.cid] = 0
            inverted_index[term][chunk.cid] += 1
    return inverted_index

# inverted_index = index(to_index)

# with open("inverted_index.json", "w") as f:
#     dump(inverted_index, f, indent=4)
N = 13070
def vector_chunks(index, query= None):
    from math import log
    from collections import Counter
    def idf(term):
        df = len(index.get(term, {}))
        return log((N - df + 0.5) / (df + 0.5) + 1)
    if query:
        query_tf = Counter(query)
        vector = {}
        for term, tf in query_tf.items():
            vector[term] = (1 + log(tf)) * idf(term)
        return vector
    vectors = {}
    for term, posting_list in index.items():
        for chunk_id, tf in posting_list.items():
            if chunk_id not in vectors:
                vectors[chunk_id] = {}
            vectors[chunk_id][term] = (1 + log(tf)) * idf(term)
    return vectors

def cosine_similarity(chunk_vectors, query_vector):
    from math import sqrt
    mag_query = sqrt(sum(v * v for v in query_vector.values()))
    scores = []
    for chunk_id, vector in chunk_vectors.items():
        dp = sum(v * vector[term] for term, v in query_vector.items() if term in vector)
        chunk_mag = sqrt(sum(v * v for v in vector.values()))
        scores.append((chunk_id, dp / (mag_query * chunk_mag)**0.1))
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:5]

import uuid

class UnansweredQuestion(BaseModel):
    question_id: str = Field(default_factory=lambda:
    str(uuid.uuid4()))
    question: str

class MinimalSource(BaseModel):
    file_path: str
    first_character_index: int
    last_character_index: int

class MinimalSearchResults(BaseModel):
    question_id: str
    question: str
    retrieved_sources: List[MinimalSource]

class StudentSearchResults(BaseModel):
    search_results: List[MinimalSearchResults]
    k: int

def answer_dataset():
    with open("UnansweredQuestions/dataset_docs_public.json") as f:
        questions = [UnansweredQuestion(**q) for q in load(f).get('rag_questions')]
    with open("chunks_database.json") as f:
        chunks_map = {str(chunk['cid']): chunk for chunk in load(f)}
    with open("inverted_index.json") as f:
        inverted_index = load(f)
    search_result = []
    vectors = vector_chunks(inverted_index)
    for question in questions:
        minima_list = []
        query_vector = vector_chunks(inverted_index, tokenization(query=question.question))
        for chunk_id, score in cosine_similarity(vectors, query_vector):
            chunk = chunks_map.get(str(chunk_id))
            source = MinimalSource(file_path="data/raw/"+chunk['fp'], first_character_index=chunk['s'], last_character_index=chunk['e'])
            minima_list.append(source)
        search_result.append(MinimalSearchResults(question_id=question.question_id, question=question.question, retrieved_sources=minima_list))
    with open("search_result.json", "w") as f:
        dump(StudentSearchResults(search_results=search_result, k=5).model_dump(), f, indent=4)
answer_dataset()