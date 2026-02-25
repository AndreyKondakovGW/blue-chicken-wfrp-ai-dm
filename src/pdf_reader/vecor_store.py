import os
import argparse
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import CharacterTextSplitter
from src.pdf_reader.pdf_parser import PDFParser

DATABASE_PATH = './databases'

class VectorStore:
    '''
    Class responsible to create FAISS vector store from a file (currently only PDF supported).
    This class use CharacterTextSplitter from langchain.
    '''
    def __init__(self, 
        enbedding_model_name="mxbai-embed-large",
        chunk_size=1024,
        chunk_overlap=256,
        database_path=DATABASE_PATH
    ):
        try:
            from langchain_ollama import OllamaEmbeddings
        except ImportError:
            print("To create embeddings of the pdf file Ollama pacakge is needed. Please writr instruction how to install Ollama in README file")

        self.text_spliter = CharacterTextSplitter(separator='\n',
                                        chunk_size=chunk_size, chunk_overlap=chunk_overlap,
                                        length_function=len)

        self.embedder = OllamaEmbeddings(model=enbedding_model_name)
        self.database_path = database_path

    def create_new_vectorstore(self, vectorstore_name, text_chunks):
        "Create a new vector store from text chunks and save it locally."
        db = FAISS.from_texts(text_chunks, self.embedder)
        db.save_local(os.path.join(self.database_path, vectorstore_name))
        return db

    def load_vectorstore(self, vectorstore_name):
        return FAISS.load_local(os.path.join(self.database_path, vectorstore_name), self.embedder, allow_dangerous_deserialization=True)

    def create_from_pdf(self, file_path, filename):
        "Create a vector store from a PDF file."
        _, file_extension = os.path.splitext(file_path)
        if file_extension == ".pdf":
            parser = PDFParser()
            file_text = parser.parse_file(file_path)
        file_chunks = self.split_text(file_text)

        print(f"{len(file_chunks)} chunks created from the PDF file.")
        return self.create_new_vectorstore(filename, file_chunks)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a vector database from a PDF file.")
    parser.add_argument("pdf_path", type=str, help="Path to the PDF file.")
    parser.add_argument("vector_store_name", type=str, help="Name for the vector store.")
    args = parser.parse_args()

    vector_store = VectorStore()
    vector_store = vector_store.create_from_pdf(args.pdf_path, args.vector_store_name)