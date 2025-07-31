import os
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
# I'm gonna try and use this to create a message history but im unsure how to do it due to the templating.
from langchain_core.messages import HumanMessage, AIMessage

from dotenv import load_dotenv
# There is a future warning that kept arising after asking a question so this is put in place to block it.
import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)

load_dotenv()

model = ChatGroq(model="llama-3.1-8b-instant")

current_dir = os.getcwd()
Datastore_dir = os.path.join(current_dir, "Datastore", "chroma_db")

print("Initializing Embedding Model...")
embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-large-en-v1.5")
print("Initialized Embedding Model!")

print("Initializing Datastore...")
db = Chroma(embedding_function=embedding_model, persist_directory=Datastore_dir)
print("Initializing Datastore!\n")

# Setting up the retriever we want that goes and collects relevant data from the database
retriever = db.as_retriever(search_type="similarity_score_threshold", search_kwargs={"k":3, "score_threshold":0.2})

prompt = [("system", "You are a expert on the book \'A Room With a View by E. M. Forster\'. Answer the user's query using the following information: {content}\nIf you dont find anythinf relevant say I don't know"),
          ("human", "the user wants to know {query}")]

template = ChatPromptTemplate.from_messages(prompt)

while True:
    query = input("What do you want to know about A Room With a View? ")
    if query.lower() == "exit":
        break
    content = retriever.invoke(query)
    response = model.invoke(template.invoke({"content":f"{content}", "query":f"{query}"}))
    print("\nAI:", response.content, "\n")