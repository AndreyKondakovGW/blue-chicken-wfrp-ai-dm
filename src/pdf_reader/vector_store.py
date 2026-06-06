import os
import re
import argparse
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter
from src.pdf_reader.pdf_parser import PDFParser
import numpy as np
from typing import List
DATABASE_PATH = './databases'

class VectorStore:
    '''
    Class responsible to create FAISS vector store from a file (currently only PDF supported).
    This class use RecursiveCharacterTextSplitter from langchain.
    '''
    def __init__(self, 
        enbedding_model_name="bge-m3",
        chunk_size=128*8,
        chunk_overlap=64*4,
        database_path=DATABASE_PATH
    ):
        try:
            from langchain_ollama import OllamaEmbeddings
        except ImportError:
            print("To create embeddings of the pdf file Ollama pacakge is needed. Please writr instruction how to install Ollama in README file")

        self.text_spliter = RecursiveCharacterTextSplitter(
                                        chunk_size=chunk_size, 
                                        chunk_overlap=chunk_overlap,
                                        length_function=len,
                                        separators=["<p>", "\n\n", "\n", ". ", " ", ""])

        self.embedder = OllamaEmbeddings(model=enbedding_model_name)
        self.database_path = database_path

    def create_new_vectorstore(self, vectorstore_name, text_chunks):
        "Create a new vector store from text chunks and save it locally."
        db = FAISS.from_texts(text_chunks, self.embedder, normalize_L2=True)
        db.save_local(os.path.join(self.database_path, vectorstore_name))
        return db

    def load_vectorstore(self, vectorstore_name):
        return FAISS.load_local(os.path.join(self.database_path, vectorstore_name), self.embedder, allow_dangerous_deserialization=True)

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
        threshold: float = 0.82,
        min_chunk_len: int = 400
        ) -> List[str]:
        """
        Merge adjacent chunks if semantically similar.
        """

        if not chunks:
            return []

        # Precompute embeddings
        embeddings = [self.embedder.embed_query(c) for c in chunks]

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

    def create_from_pdf(self, file_path, filename):
        "Create a vector store from a PDF file."
        _, file_extension = os.path.splitext(file_path)
        if file_extension == ".pdf":
            parser = PDFParser()
            file_text = parser.parse_file(file_path)
        file_chunks = []
        for page_num, page_text in file_text.items():
            page_chunks = self.split_text(page_text)
            page_chunks = self.semantic_merge_chunks(page_chunks)
            page_chunks = [f"Page {page_num}: {chunk}" for chunk in page_chunks]
            file_chunks.extend(page_chunks)
            if page_num == 158:  # Print the chunks for page 158
                print(page_text)

        print(f"Created {len(file_chunks)} chunks from the PDF file.")
        return self.create_new_vectorstore(filename, file_chunks)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a vector database from a PDF file.")
    parser.add_argument("pdf_path", type=str, help="Path to the PDF file.")
    parser.add_argument("vector_store_name", type=str, help="Name for the vector store.")
    args = parser.parse_args()

    vector_store = VectorStore()
    vector_store = vector_store.create_from_pdf(args.pdf_path, args.vector_store_name)