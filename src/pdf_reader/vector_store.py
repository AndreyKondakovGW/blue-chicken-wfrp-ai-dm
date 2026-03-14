import os
import re
import argparse
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter
from src.pdf_reader.pdf_parser import PDFParser


DATABASE_PATH = './databases'

class VectorStore:
    '''
    Class responsible to create FAISS vector store from a file (currently only PDF supported).
    This class use RecursiveCharacterTextSplitter from langchain.
    '''
    def __init__(self, 
        enbedding_model_name="mxbai-embed-large",
        chunk_size=128*4,
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
        db = FAISS.from_texts(text_chunks, self.embedder)
        db.save_local(os.path.join(self.database_path, vectorstore_name))
        return db

    def load_vectorstore(self, vectorstore_name):
        return FAISS.load_local(os.path.join(self.database_path, vectorstore_name), self.embedder, allow_dangerous_deserialization=True)

    def split_text(self, text):
        #better spliter with re for PDF files
        # Remove multiple consecutive newlines and replace them with a single newline
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r"\n", " ", text)
        chunks = self.text_spliter.split_text(text)
        return chunks

    def create_from_pdf(self, file_path, filename):
        "Create a vector store from a PDF file."
        _, file_extension = os.path.splitext(file_path)
        if file_extension == ".pdf":
            parser = PDFParser()
            file_text = parser.parse_file(file_path)
        file_chunks = []
        for page_num, page_text in file_text.items():
            page_chunks = self.split_text(page_text)
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