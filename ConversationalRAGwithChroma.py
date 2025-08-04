import os
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda
from langchain.schema.output_parser import StrOutputParser
from dotenv import load_dotenv
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)
load_dotenv()

current_dir = os.getcwd()
datastore_dir = os.path.join(current_dir, "Datastore", "chroma_db")

model = ChatGroq(model="llama-3.1-8b-instant", temperature=0.9)

print("Initializing Embedding Model...")
embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-large-en-v1.5")
print("Initialized Embedding Model!")

print("Initializing Datastore...")
db = Chroma(embedding_function=embedding_model, persist_directory=datastore_dir)
print("Initializing Datastore!\n")

retriever = db.as_retriever(search_type="similarity", search_kwargs={"k":3})

# We feed the new_queston_promt to go through the model after being formatted, and then use the model output to get our retriever to retrieve.
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

format_retrieved_docs_prompt = RunnableLambda(lambda x: retrieved_docs_prompt.invoke({"context":x, "history":f"{chat_history}", "query":f"{query}"}))

chain = new_question_prompt | model | StrOutputParser() | retriever | format_retrieved_docs_prompt | model | StrOutputParser()

chat_history = []
# This part makes up the Conversational RAG chatbot part of the script.
while True:
    query = input("What would you like to know? ")
    if query.lower() == "exit":
        break
    response = chain.invoke({"history":f"{chat_history}","query":f"{query}"})
    chat_history.append(HumanMessage(query))
    chat_history.append(AIMessage(response))
    print("\nAI:", response, "\n")