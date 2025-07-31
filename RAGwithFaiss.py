import os
import faiss
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader

current_dir = os.getcwd()
datastore_dir = os.path.join(current_dir, "Datastore", "faiss_db")
book_path = os.path.join(current_dir, "Books", "A_Room_with_a_View.txt")

print("Initializing Embedding Model...")
embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-large-en-v1.5")
print("Initialized Embedding Model!")

loader = TextLoader(file_path=book_path, encoding="utf-8")
docs = loader.load()

splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=0)
docs = splitter.split_documents(docs)

print("Initializing Datastore...")
# index = faiss.IndexFlatL2(len(embedding_model.embed_query("hello")))
vector_store = FAISS.from_documents(docs, embedding=embedding_model)
print("Initialized Database!")

# Saving the docs to the vector store with our chosen embedding model.
# vector_store.add_documents(docs)
print("Saving Documents...")
vector_store.save_local(datastore_dir)
print("Saved Documents!")