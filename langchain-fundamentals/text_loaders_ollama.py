from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import pprint
import re

# Data cleaning function
def clean_text(text):
    # Remove unwanted characters (e.g., digits, special characters)
    text = re.sub(r"[^a-zA-Z\s]", "", text)

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    # Convert to lowercase
    text = text.lower()

    return text

documents = TextLoader("./doc/dream.txt").load()

# Split the text into characters
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)

cleaned_text = clean_text(documents[0].page_content)
texts = text_splitter.split_text(cleaned_text)

# Load the Ollama Embeddings to vectorize the text
embeddings = OllamaEmbeddings(
    model="qwen3-embedding:4b",
    base_url="http://localhost:11434"
)

# create the retriever from the loaded embeddings and documents
retriever = FAISS.from_texts(texts, embeddings).as_retriever(search_kwargs={"k": 2})


# Query the retriever
# query = "what did Martin Luther King Jr. dream about?"
query = "Give me a summary of the speech in bullet points"
docs = retriever.invoke(query)

pprint.pprint(f" => DOCS: {docs}:")

# Chat with the model and our docs

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate


# # Create the chat prompt
prompt = ChatPromptTemplate.from_template(
    "Please use the following docs {docs},and answer the following question {query}",
)

# # Create a chat model
model = ChatOpenAI(
    model="llama3.1",
    base_url="http://localhost:11434/v1",
    api_key="ollama",
)

chain = prompt | model | StrOutputParser()

response = chain.invoke({"docs": docs, "query": query})
print(f"Model Response: \n \n{response}")
