# # from langchain.document_loaders import DirectoryLoader
# from langchain_community.document_loaders import DirectoryLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.schema import Document
# # from langchain.embeddings import OpenAIEmbeddings
# from langchain_openai import OpenAIEmbeddings
# from langchain_community.vectorstores import Chroma
# import openai 
# from dotenv import load_dotenv
# import os
# import shutil

# # Load environment variables. Assumes that project contains .env file with API keys
# load_dotenv()
# #---- Set OpenAI API key 
# # Change environment variable name from "OPENAI_API_KEY" to the name given in 
# # your .env file.
# openai.api_key = ""
# CHROMA_PATH = "chroma"
# DATA_PATH = "documents"


# def main():
#     generate_data_store()


# def generate_data_store():
#     documents = load_documents()
#     chunks = split_text(documents)
#     save_to_chroma(chunks)


# def load_documents():
#     loader = DirectoryLoader(DATA_PATH, glob="*.md")
#     documents = loader.load()
#     return documents


# def split_text(documents: list[Document]):
#     text_splitter = RecursiveCharacterTextSplitter(
#         chunk_size=300,
#         chunk_overlap=100,
#         length_function=len,
#         add_start_index=True,
#     )
#     chunks = text_splitter.split_documents(documents)
#     print(f"Split {len(documents)} documents into {len(chunks)} chunks.")

#     document = chunks[10]
#     print(document.page_content)
#     print(document.metadata)

#     return chunks


# def save_to_chroma(chunks: list[Document]):
#     # Clear out the database first.
#     if os.path.exists(CHROMA_PATH):
#         shutil.rmtree(CHROMA_PATH)

#     # Create a new DB from the documents.
#     db = Chroma.from_documents(
#         chunks, OpenAIEmbeddings(), persist_directory=CHROMA_PATH
#     )
#     db.persist()
#     print(f"Saved {len(chunks)} chunks to {CHROMA_PATH}.")


# if __name__ == "__main__":
#     main()

import os
import shutil
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

CHROMA_PATH = "chroma"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
SOURCE_FILE = "documents/book.md"

def load_and_process_document(file_path):
    print(f"Attempting to read file: {file_path}")
    print(f"File exists: {os.path.exists(file_path)}")
    print(f"File size: {os.path.getsize(file_path)} bytes")
    
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    
    print(f"Content length: {len(content)} characters")
    print("First 500 characters of the file:")
    print(content[:500])
    print("..." if len(content) > 500 else "")
    print("\n")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(content)
    
    print(f"Processed document into {len(chunks)} chunks.")
    print(f"First chunk preview: {chunks[0][:200]}...")
    
    return [Document(page_content=chunk, metadata={"source": file_path, "chunk_id": i+1}) for i, chunk in enumerate(chunks)]

def main():
    # Check if Chroma database already exists
    if os.path.exists(CHROMA_PATH):
        user_input = input("Chroma database already exists. Do you want to delete and recreate it? (y/n): ")
        if user_input.lower() == 'y':
            print("Removing existing Chroma database...")
            shutil.rmtree(CHROMA_PATH)
        else:
            print("Exiting without changes.")
            return

    print("Processing document and creating new Chroma database...")
    documents = load_and_process_document(SOURCE_FILE)
    embedding_function = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    db = Chroma.from_documents(documents, embedding_function, persist_directory=CHROMA_PATH)
    db.persist()

    print(f"Chroma database created with {db._collection.count()} documents.")

if __name__ == "__main__":
    main()


