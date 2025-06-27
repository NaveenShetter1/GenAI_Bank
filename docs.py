from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import GoogleGenerativeAI
from langchain.chains import RetrievalQA
import os

# Set API key
os.environ["GOOGLE_API_KEY"] = "AIzaSyAscCPbKWrM41aBimhKexldkDt77XaeNk0"

# Load PDF
loader = PyMuPDFLoader("bank_policy.pdf")
docs = loader.load()

# Split into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(docs)

# Use HuggingFace embeddings (free)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Store in FAISS
vectorstore = FAISS.from_documents(chunks, embeddings)

# Create retriever
retriever = vectorstore.as_retriever()

# Gemini LLM
llm = GoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0.3
)

# QA chain
qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

# Ask question
query = "how many days will it take to close the account?"
answer = qa.invoke(query)

print("\n📌 Answer:", answer["result"])
