from pathlib import Path
from models import (
    RagDataset,
    MinimalSource,
    StudentSearchResults,
    MinimalSearchResults,
    UnansweredQuestion
)
import json 
from BM25 import BM25
from chunking import chunking


def main(k=5):

    import json
    with open("UnansweredQuestions/dataset_code_public.json") as f:
        questions = [UnansweredQuestion(**q) for q in json.load(f).get('rag_questions')]

    test = chunking(2000)
    di = test.find_files()
    print("start")
    test.chunking_py(di["py"])
    test.chunk_docs(di["docs"])
    print("done")
    all_chunks = [
        d.model_dump() for mode in test.dic_chunks for d in test.dic_chunks[mode]
    ]
    to_index = [d for mode in test.dic_chunks for d in test.dic_chunks[mode]]

    with open("issam.json", "w") as f:
        json.dump(all_chunks, f, indent=4)

    c = BM25(to_index)
    c.tokenizer()
    k = c.tf_term()
    with open("index.json", "w") as e:
        json.dump(k, e, indent=4)
    result  = []
    for ques in questions:
        new = c.BM25(ques.question)[-5:]
        # print("results ===",new)
        m= MinimalSearchResults(
            question = ques.question,
            question_id = ques.question_id,
            retrieved_sources = [
                MinimalSource(
                    file_path= all_chunks[chun[0]-1]["file_path"],
                    first_character_index=all_chunks[chun[0]-1]["first_character_index"],
                    last_character_index=all_chunks[chun[0]-1]["last_character_index"]
                    )for chun in new]
        )
        result.append(m)
    search_result = StudentSearchResults(search_results=result, k=5).model_dump()

    with open("dataset_code_public.json",'w') as f:
        json.dump(search_result, f,indent=4)
    
main()
