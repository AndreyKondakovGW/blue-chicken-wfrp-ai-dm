
import os
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from rank_bm25 import BM25Okapi
import nltk
from nltk.tokenize import word_tokenize
from collections import defaultdict
from typing import List
from src.pdf_reader.pdf_parser import PDFParser
import re
import numpy as np
import nltk
import pickle
from sentence_transformers import CrossEncoder
DATABASE_PATH = './databases'

class DenseRetriver():
    """
    Class that implement dense document retrival useing FAISS vector database and Ollama embedding models
    """
    def __init__(self,
        embedder,
        database_path=DATABASE_PATH
    ):
        self.name = "Dense Retriver"
        self.embedder = embedder
        self.database_path = database_path
        self.database = None

    def create_new_vectorstore(self, vectorstore_name, text_chunks):
        "Create a new vector store from text chunks and save it locally."
        
        db.save_local(os.path.join(self.database_path, vectorstore_name))
        self.database = db
        return self.database

    def load_vectorstore(self, vectorstore_name):
        self.database = FAISS.load_local(os.path.join(self.database_path, vectorstore_name), self.embedder, allow_dangerous_deserialization=True)
        return self.database

    def search(self, query, k=6):
        return self.database.similarity_search_with_score(query.lower(), k=6)

class SparseRetriver():
    """
    Class that implement sparse document retrival using BM25 model
    """
    def __init__(self,
        database_path=DATABASE_PATH
    ):
        self.name = "Sparse Retriver"
        self.database_path = database_path
        self.chuncks= []

    def create_new_vectorstore(self, vectorstore_name, text_chunks):
        tokenized_chunks = [
            word_tokenize(doc.lower())
            for doc in text_chunks
        ]
        self.chuncks = text_chunks
        bm25 = BM25Okapi(tokenized_chunks)
        self.database = bm25
        with open(os.path.join(self.database_path, vectorstore_name+".pkl"), "wb") as f:
            pickle.dump(bm25, f)
        return self.database

    def load_vectorstore(self, vectorstore_name):
        with open(os.path.join(self.database_path, vectorstore_name+".pkl"), "rb") as f:
            self.database = pickle.load(f)
        return self.database

    def search(self, query, k=6):
        scores = self.database.get_scores(word_tokenize(query.lower()))

        top_k = sorted(
            zip(self.chuncks, scores),
            key=lambda x: x[1],
            reverse=True
        )[:k]

        return top_k

class HybridRetrival():
    def __init__(self,
        dense_embedder=None,
        database_path=DATABASE_PATH,
        use_sparse=True
    ):
        self.database_path = database_path
        self.use_sparse=use_sparse
        self.vector_store = None

        if not dense_embedder is None:
            self.embedder = dense_embedder
        else:
            print("Can not create Dense Retrival without Embedding model which is None") 

    def create_new_vectorstore(self, vectorstore_name, text_chunks, save=True):
        self.vector_store = FAISS.from_texts(text_chunks, self.embedder, normalize_L2=True)
        if self.use_sparse:
            tokenized_chunks = [
                word_tokenize(doc.lower())
                for doc in text_chunks
            ]
            self.sparsedatabase = BM25Okapi(tokenized_chunks)
            if save:
                with open(os.path.join(self.database_path, vectorstore_name+".pkl"), "wb") as f:
                    pickle.dump(self.sparsedatabase, f)
        if save:
            self.vector_store.save_local(os.path.join(self.database_path, vectorstore_name))

    def load_vectorstore(self, vectorstore_name):
        self.vector_store = FAISS.load_local(os.path.join(self.database_path, vectorstore_name), self.embedder, allow_dangerous_deserialization=True)
        if self.use_sparse:
            with open(os.path.join(self.database_path, vectorstore_name+".pkl"), "rb") as f:
                self.sparsedatabase = pickle.load(f)

    def search(self, query, k=6):
        dense_res = self.vector_store.similarity_search_with_score(query.lower(), k=k)

        rankings = []
        rankings.append([
                doc.id
                for doc, _ in dense_res
            ])

        if self.use_sparse:
            scores = self.sparsedatabase.get_scores(word_tokenize(query.lower()))
            top_idx = np.argsort(scores)[::-1][:k]
            
            ids = list(self.vector_store.docstore._dict)
            sparse_res = [(self.vector_store.docstore._dict[ids[i]], scores[i]) for i in top_idx]
            rankings.append([
                doc.id
                for doc, _ in sparse_res
            ])

            scores = defaultdict(float)
            rrf_k = 60

            for ranking in rankings:
                for rank, doc_id in enumerate(ranking, start=1):
                    scores[doc_id] += 1 / (rrf_k + rank)

            ranked_ids = sorted(
                scores,
                key=scores.get,
                reverse=True
            )
        
            merged_scroes = [(self.vector_store.docstore._dict[i], scores[i]) for i in ranked_ids[:k]]

            return [doc.page_content for doc, _ in merged_scroes]
        else:
            return [doc.page_content for doc, _ in dense_res]

from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter

class PDFReader():
    def __init__(self, 
        chunk_size=128*8,
        chunk_overlap=64*4
    ,):
        self.text_spliter = RecursiveCharacterTextSplitter(
                                        chunk_size=chunk_size, 
                                        chunk_overlap=chunk_overlap,
                                        length_function=len,
                                        separators=["<p>", "\n\n", "\n", ". ", " ", ""])

    def split_text(self, text):
        #better spliter with re for PDF files
        # Remove multiple consecutive newlines and replace them with a single newline
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r"\n", " ", text)
        text = text.lower()
        chunks = self.text_spliter.split_text(text)
        return chunks

    def cosine_sim(self, vec1, vec2):
        "Calculate cosine similarity between two vectors."
        if np.linalg.norm(vec1) == 0 or np.linalg.norm(vec2) == 0:
            return 0.0
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

    def semantic_merge_chunks(
        self,
        chunks: List[str],
        embedder,
        threshold: float = 0.82,
        min_chunk_len: int = 400,
        ) -> List[str]:
        """
        Merge adjacent chunks if semantically similar.
        """

        if not chunks:
            return []

        # Precompute embeddings
        embeddings = [embedder.embed_query(c) for c in chunks]

        merged_chunks = []
        current_chunk = chunks[0]
        current_emb = embeddings[0]

        for i in range(1, len(chunks)):
            sim = self.cosine_sim(current_emb, embeddings[i])

            should_merge = (
                sim >= threshold or
                len(current_chunk) < min_chunk_len
            )

            if should_merge:
                current_chunk += "\n\n" + chunks[i]
                current_emb = (current_emb + np.array(embeddings[i])) / 2
            else:
                merged_chunks.append(current_chunk)
                current_chunk = chunks[i]
                current_emb = np.array(embeddings[i])

        merged_chunks.append(current_chunk)

        return merged_chunks

    def create_chunks_from_pdf(self, file_path, embedder=None):
        _, file_extension = os.path.splitext(file_path)
        if file_extension == ".pdf":
            parser = PDFParser()
            file_text = parser.parse_file(file_path)
        file_chunks = []
        for page_num, page_text in file_text.items():
            page_chunks = self.split_text(page_text)
            #page_chunks = self.semantic_merge_chunks(page_chunks, embedder=embedder)
            page_chunks = [f"Page {page_num}: {chunk}" for chunk in page_chunks]
            file_chunks.extend(page_chunks)

        return file_chunks

class Rerancker():
    def __init__(self, model_name="BAAI/bge-reranker-v2-m3"):
        self.model = CrossEncoder("BAAI/bge-reranker-v2-m3")

    def rerank_docs(query, docs, k=3):
        pairs = [(query, doc) for doc in docs]
        scores = model.predict(pairs)

        ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)

        return [doc for doc, score in ranked[:k]]

if __name__ == "__main__":
    nltk.download('punkt_tab')
    enbedding_model_name="bge-m3"
    embedder = OllamaEmbeddings(model=enbedding_model_name)

    pdf_reader = PDFReader()
    pdf_path = "./documents/up_in_arms_ru.pdf"
    query = "Помощь при проверке"
    doc_chunks = pdf_reader.create_chunks_from_pdf(pdf_path, embedder)
    print(f"Created {len(doc_chunks)} chunks from the PDF file.")

    retriver = HybridRetrival(dense_embedder=embedder)

    retriver.create_new_vectorstore("up_in_arms_ru", doc_chunks)
    # retriver.load_vectorstore("WFRPG4E_ru")
    # docs = retriver.search(query, k=10)
    # for doc in docs:
    #     print(doc[:30])
    

    # Sort by score
    

    
