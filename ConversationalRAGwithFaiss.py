import os
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.runnables import RunnableLambda
from langchain.schema.output_parser import StrOutputParser

import warnings
warnings.simplefilter('ignore', FutureWarning)
from dotenv import load_dotenv 
load_dotenv()

current_dir = os.getcwd()
datastore_dir = os.path.join(current_dir, "Datastore", "faiss_db")

model = ChatGroq(model="qwen/qwen3-32b", temperature=0.5, reasoning_format="hidden")

print("Initializing Embedding Model...")
embedding_model_name = os.environ['EMBEDDING_MODEL_NAME']
# For context the name is "BAAI/bge-large-en-v1.5"
embedding_model = HuggingFaceEmbeddings(model_name=embedding_model_name)
print("Initalized Embedding Model!")

print("Initializing Vector Store...")
vector_store = FAISS.load_local(datastore_dir, embedding_model, allow_dangerous_deserialization=True)
print("Initalized Vector Store!")

retriever = vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 3})
print("Set up Retriever!")

new_question_template = [("system", "Given a chat history and the latest user question "
                                    "which might reference context in the chat history, "
                                    "formulate a standalone question which can be understood "
                                    "without the chat history. Do NOT answer the question, just "
                                    "reformulate it if needed and otherwise return it as is."),
                                    ("system","Here is the chat history: {history}"),
                                    ("human", "{query}")]

new_question_prompt = ChatPromptTemplate.from_messages(new_question_template)

retrieved_docs_template = [("system", "You are a expert on the book \'A Room With a View by E. M. Forster\'. Use "
                                        "the following pieces of retrieved context to answer the "
                                        "question. If you don't know the answer, just say that you "
                                        "don't know. Use three sentences maximum and keep the answer "
                                        "concise."
                                        "\n\n"
                                        "{context}"),
                                        ("system","Here is the chat history: {history}"),
                                        ("human", "{query}")]

retrieved_docs_prompt = ChatPromptTemplate.from_messages(retrieved_docs_template)

chat_history = []
format_retrieved_docs_prompt = RunnableLambda(lambda x: retrieved_docs_prompt.invoke({"context": x, "history": f"{chat_history}", "query": f"{query}"}))
chain = new_question_prompt | model | StrOutputParser() | retriever | format_retrieved_docs_prompt | model | StrOutputParser()

while True:
    query = input("What do you want to know about A_Room_with_a_View? ")
    if query.lower() == "exit":
        break
    response = chain.invoke({"history": f"{chat_history}", "query": f"{query}"})
    print("\nAI:", response, "\n")
    chat_history.append(HumanMessage(query))
    chat_history.append(AIMessage(response))