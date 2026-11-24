from string import punctuation
from nltk.stem import PorterStemmer
import math


class BM25:
    def __init__(self, all_chunks):
        self.all_chunks = all_chunks
        self.inverted = {}
        self.corpus = []
        self.k = 1.2
        self.b = 0.75
        self.avrg_len = 0
    def query_tokenizer(self, query):

        stemmer = PorterStemmer()
        punc = str.maketrans(punctuation, " " * len(punctuation))
        new = query.translate(punc)

        stem = [stemmer.stem(word.lower()) for word in new.split() if len(word) > 1]
        res = " ".join(stem)
        return res

    def tokenizer(self):

        stemmer = PorterStemmer()
        punc = str.maketrans(punctuation, " " * len(punctuation))
        chunks_len = 0
        for chunk in self.all_chunks:
            tokenized = chunk.content.translate(punc)
            stemming = [
                stemmer.stem(word.lower())
                for word in tokenized.split()
                if len(word) > 1
            ]
            chunks_len += len(stemming)
            self.corpus.append(chunk.model_copy(update={"content": " ".join(stemming)}))
        self.avrg_len = chunks_len/len(self.corpus)
        return self.corpus

    def tf_term(self):
        self.inverted = {}
        for chunk in self.corpus:
            content = chunk.content
            for term in content.split():
                if term not in self.inverted:
                    self.inverted[term] = {}
                if chunk.chunk_id not in self.inverted[term]:
                    self.inverted[term][chunk.chunk_id] = 0
                self.inverted[term][chunk.chunk_id] += 1
        return self.inverted

    # def counter(self):
    #     from collections import Counter

    #     res = []
    #     for chunk in self.corpus:
    #         res.append(Counter(chunk.content.split()))
    #     return res

    def idf(self, word):
        N = len(self.corpus)
        if word not in self.inverted:
            n = 0
        else:
            n = len(self.inverted[word])
        return math.log(1 + (N - n + 0.5 )/ (n + 0.5))

    def get_tf_term(self, word, chunk_id, len_d):
        if word not in self.inverted or chunk_id not in self.inverted[word]:
            return 0
        tf = self.inverted[word][chunk_id]
        nume = tf * (self.k+1)
        len_normalisation  = 1-self.b + self.b * (len_d /self.avrg_len)
        doni = tf + self.k * len_normalisation
        return nume/doni

    def BM25(self, query):
        ready_query = self.query_tokenizer(query)
        scores = []
        for i in range(1,len(self.corpus)+1):
            score = 0
            for word in ready_query.split():
                Bm_idf = self.idf(word)
                Bm_tf = self.get_tf_term(word, i, len(self.corpus[i-1].content.split()))
                score+= Bm_idf * Bm_tf
            scores.append((i, score))
        return sorted(scores, key=lambda index: index[1])
                        
                
