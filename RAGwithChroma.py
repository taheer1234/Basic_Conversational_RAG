import os
from langchain.text_splitter import CharacterTextSplitter, TextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings

current_dir = os.getcwd()
file_path = os.path.join(current_dir, "Books", "A_Room_with_a_View.txt")
datastore_dir = os.path.join(current_dir, "Datastore", "chroma_db")

loader = TextLoader(file_path, encoding="utf-8")
text = loader.load()
print("Loaded Text!\n")
print(text)
# splitter = CharacterTextSplitter(chunk_size = 4000, chunk_overlap = 200) # Default values for chunks
# docs = splitter.split_documents(text)
# print("Split into Documents!\n")

# embedding_model = GPT4AllEmbeddings(model_name="all-MiniLM-L6-v2.gguf2.f16.gguf")
# embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-large-en-v1.5")
# print("Loaded Embedding Model!\n")
# db = Chroma.from_documents(documents=docs, embedding=embedding_model, persist_directory=datastore_dir)
# print("Saved Vector Database!")