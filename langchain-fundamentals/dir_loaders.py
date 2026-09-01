from langchain_community.document_loaders import TextLoader
import os

# Load all .txt files from the data directory manually
documents = []
data_dir = "./data/"

# Iterate through all files in the directory
for filename in os.listdir(data_dir):
    # Only process text files
    if filename.endswith('.txt'):
        file_path = os.path.join(data_dir, filename)
        loader = TextLoader(file_path)
        documents.extend(loader.load())
        print(f"Loaded: {filename}")

# Display summary of loaded documents
print(f"\nTotal documents loaded: {len(documents)}")

# Show details of each document
for i, doc in enumerate(documents):
    print(f"\n--- Document {i+1} ---")
    print(f"Source: {doc.metadata.get('source', 'unknown')}")
    print(f"Content length: {len(doc.page_content)} characters")
    print(f"Preview (first 100 chars): {doc.page_content[:100]}...")



#     
# from langchain_community.document_loaders import (
#     TextLoader,
#     PyPDFLoader,
#     CSVLoader,
#     DirectoryLoader,
# )
#
# # from langchain_community.document_loaders import DirectoryLoader, TextLoader
# dir_loader = DirectoryLoader("./data/", glob="**/*.txt")
# dir_documents = dir_loader.load()
#
# print("Directory Text Documents:", dir_documents)
