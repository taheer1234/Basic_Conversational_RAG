import os
# from langchain_community.vectorstores import Chroma, FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

current_dir = os.getcwd()
Datastore_dir = os.path.join(current_dir, "Datastore", "chroma_db")

print("Initializing Embedding Model...")
embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-large-en-v1.5")
print("Initialized Embedding Model!")

print("Initializing Database...")
db = Chroma(embedding_function=embedding_model, persist_directory=Datastore_dir)
print("Initialized Database!")

retriever = db.as_retriever(search_type="mmr",
                            search_kwargs={"k": 3, "lambda_mult":0.9})

while True:
    query = input("What do you want to know about A Room with a View? ")
    if query.lower() == "exit":
        break

    answer = retriever.invoke(query)
    print("Here is the most relevant document:\n",answer[0].page_content,"\n")