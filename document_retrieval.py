import numpy as np
from typing import List

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class Retriever:
    def __init__(self, num_parts=18013, suffix_length=5, dataset='./wikitext-103'):
        self.vectorizer = TfidfVectorizer()
        self.corpus = []
        for split_idx in range(num_parts):
            num_zeros = suffix_length - len(str(split_idx))
            split_idx_str = num_zeros * '0' + str(split_idx)
            split_file = open(dataset + f'/wiki.train.split.{split_idx_str}.tokens')
            split_text = split_file.read()
            self.corpus.append(split_text)

        self.corpus_tfidf = self.vectorizer.fit_transform(self.corpus)

    def retrieve(self, prefix: str, k=5) -> List[str]:
        prefix_tfidf = self.vectorizer.transform([prefix])
        indices = cosine_similarity(self.corpus_tfidf, prefix_tfidf)[:,0].argsort()[-k:].tolist()
        indices.reverse()
        return indices
    
    def print_indices(self, indices: List[int], length_per_example=1000) -> None:
        for index in indices:
            print('index: ', index)
            print('example: ', self.corpus[index][:length_per_example])

if __name__ == '__main__':
    retriever = Retriever()
    while(True):
        prefix = input("prefix: ") 
        indices = retriever.retrieve(prefix)
        retriever.print_indices(indices)
       
