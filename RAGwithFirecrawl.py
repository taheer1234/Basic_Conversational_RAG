import os
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import FireCrawlLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from firecrawl import FirecrawlApp
from langchain_core.documents import Document
from dotenv import load_dotenv
load_dotenv()

current_dir = os.getcwd()
datastore_dir = os.path.join(current_dir, "Datastore", "firecrawl_db")

E_M_N = os.environ["EMBEDDING_MODEL_NAME"]
print("Initializing Embedding Model...")
embedding_model = HuggingFaceEmbeddings(model_name=E_M_N)

print("Scraping Website for Information...")
app = FirecrawlApp(os.environ["FIRECRAWL_API_KEY"])
scrape_response = app.scrape_url("https://en.wikipedia.org/wiki/Elon_Musk")
docs = [Document(page_content=scrape_response.markdown, metadata={"Source": (scrape_response.metadata["title"], scrape_response.metadata["sourceURL"])})]

print("Splitting Documents...")
splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
docs = splitter.split_documents(docs)

print("Creating Vector Store...")
vector_store = FAISS.from_documents(documents=docs, embedding=embedding_model)

print("Saving Vector Store...")
vector_store.save_local(datastore_dir)
print("Saved Vector Store Locally!")