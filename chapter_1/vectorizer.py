import torch
from typing import List, Optional, Tuple
from tqdm.auto import tqdm
from functools import reduce
import matplotlib.pyplot as plt
import seaborn as sns


class PytorchVectorizer:
    """A pytorch not optimize vectorizer, poor implementation!
    """
    def __init__(self, documents=List[str]):
        self.documents = documents
        self.build_vocab()

    def build_vocab(self) -> None:
        vocab = set(
            word.lower() for document in self.documents for word in document.split()
        )
        self.vocab = list(vocab)
        self.vocab.sort()
        self.word2idx = dict((word, idx) for idx, word in enumerate(self.vocab))
        self.idx2word = [word for word in self.vocab]
        self.vocab_size = len(self.vocab)
        self.vocab_matrix = torch.eye(self.vocab_size, dtype=torch.int32)

    def one_hot_reprensentation(self) -> torch.Tensor:
        document_tokens = self.tokenize(documents=self.documents)
        one_hot = []
        for document_token in tqdm(
            document_tokens, desc="Getting One Hot", total=len(document_tokens)
        ):
            one_hot_matrix = self.get_one_hot_matrix(document_token=document_token)
            one_hot.append(reduce(lambda x, y: x|y, one_hot_matrix))
        return torch.stack(one_hot, dim=0)
    
    def get_one_hot_matrix(self, document_token) -> torch.Tensor:
        idx_word = [self.word2idx[token] for token in document_token]
        return self.vocab_matrix[idx_word]
    
    def tf_representation(self) -> torch.Tensor:
        document_tokens = self.tokenize(documents=self.documents)
        tf_rep = []
        for document_token in tqdm(
            document_tokens, desc="Get TF representation", total=len(document_tokens)
        ):
            one_hot_matrix = self.get_one_hot_matrix(document_token=document_token)
            tf_rep.append(one_hot_matrix.sum(dim=0))
        return torch.stack(tf_rep, dim=0)
    
    def tfidf_representation(self) -> torch.Tensor:
        document_tokens = self.tokenize(documents=self.documents)
        tf_rep, one_hot = [], []
        for document_token in tqdm(
            document_tokens, desc="Get TF-IDF representation", total=len(document_tokens)
        ):
            one_hot_matrix = self.get_one_hot_matrix(document_token=document_token)
            tf_rep_vector = one_hot_matrix.sum(dim=0).to(dtype=torch.float32)
            tf_rep_vector /= tf_rep_vector.sum()
            tf_rep.append(tf_rep_vector)
            one_hot.append(reduce(lambda x, y: x|y, one_hot_matrix))
        tf = torch.stack(tf_rep, dim=0)
        nw = torch.stack(one_hot, dim=0).sum(dim=0)
        idf = torch.log10(len(self.documents)/nw)    
        return tf * idf
        
    def tokenize(self, documents: List[str]) -> List[List[str]]:
        return [[word.lower() for word in document.split()] for document in documents]

    def plot(self, representation: torch.Tensor, title: Optional[str] = None):
        sns.heatmap(representation, annot=True,
                    cbar=False, xticklabels=self.vocab, 
                    yticklabels=["Sentence_"+str(i) for i in range(len(self.documents))])
        if title:
            plt.title(title)
        plt.tight_layout()
        plt.show()