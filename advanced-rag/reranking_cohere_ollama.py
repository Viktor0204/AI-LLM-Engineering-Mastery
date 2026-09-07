import os
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_core.prompts.prompt import PromptTemplate
import cohere

# must install this: pip install "unstructured[pdf]"
## Must pip install torch transformers sentence-transformers cohere

load_dotenv()
# Get your cohere API key on: www.cohere.com
co = cohere.ClientV2(api_key=os.environ["COHERE_API_KEY"])

collection_name = "pdf_collection"

# Default Ollama models
DEFAULT_LLM_MODEL = "ollama:qwen3:8b"  # Can be changed to any other model
DEFAULT_EMBEDDING_MODEL = "ollama:nomic-embed-text:latest"  # Lightweight embedding model


class DocumentProcessor:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
        )

    def load_documents(self, data_directory: str) -> List[Document]:
        """Load documents from a directory."""
        try:
            if not os.path.exists(data_directory):
                st.error(f"Directory does not exist: {data_directory}")
                return []

            loader = DirectoryLoader(
                data_directory, glob="**/*.*", show_progress=True
            )

            documents = loader.load()
            st.info(f"Loaded {len(documents)} documents from {data_directory}")
            return documents

        except Exception as e:
            st.error(f"Error loading documents: {str(e)}")
            return []

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into smaller chunks."""
        try:
            splits = self.text_splitter.split_documents(documents)
            st.info(f"Split documents into {len(splits)} chunks")
            return splits
        except Exception as e:
            st.error(f"Error splitting documents: {str(e)}")
            return documents


class ChromaDBManager:
    def __init__(self, persist_directory: str, embedding_model: str = DEFAULT_EMBEDDING_MODEL):
        self.persist_directory = persist_directory
        self.embedding_model = embedding_model
        self.embedding_function = self._get_embedding_function()
        os.makedirs(persist_directory, exist_ok=True)

    def _get_embedding_function(self):
        """Get the appropriate embedding function based on selection."""
        if self.embedding_model == "openai":
            return OpenAIEmbeddings()
        elif self.embedding_model.startswith("ollama:"):
            model_name = self.embedding_model.replace("ollama:", "")
            return OllamaEmbeddings(model=model_name)
        else:
            # Default to Ollama
            return OllamaEmbeddings(model="nomic-embed-text:latest")

    def create_or_load_db(
            self,
            data_directory: str = "data",
            collection_name: str = collection_name,
    ) -> Chroma:
        """Creates a new database or loads existing one."""
        try:
            # First try to load existing database
            if os.path.exists(os.path.join(self.persist_directory, "chroma.sqlite3")):
                st.info("Loading existing ChromaDB...")
                vector_store = Chroma(
                    collection_name=collection_name,
                    embedding_function=self.embedding_function,
                    persist_directory=self.persist_directory,
                )
                st.success("Successfully loaded existing database!")
                return vector_store

            # If no existing DB, create new one from documents
            st.info("No existing database found. Creating new one...")

            processor = DocumentProcessor()
            documents = processor.load_documents(data_directory)
            if not documents:
                st.error("No documents found to process!")
                return None

            splits = processor.split_documents(documents)
            st.info(f"Created {len(splits)} document chunks")

            vector_store = Chroma.from_documents(
                documents=splits,
                embedding=self.embedding_function,
                collection_name=collection_name,
                persist_directory=self.persist_directory,
            )

            st.success("Successfully created and persisted new database!")
            return vector_store

        except Exception as e:
            st.error(f"Error initializing ChromaDB: {str(e)}")
            return None


class CohereReranker:
    def __init__(self):
        self.co = co

    def rerank(
            self, query: str, documents: List[Document], top_k: int = 3
    ) -> List[Dict]:
        try:
            docs = [str(doc.page_content) for doc in documents]

            response = self.co.rerank(
                model="rerank-v3.5", query=query, documents=docs, top_n=top_k
            )

            reranked_results = []

            for result in response.results:
                reranked_results.append(
                    {
                        "document": documents[result.index],
                        "relevance_score": float(result.relevance_score),
                        "index": result.index,
                    }
                )

            return reranked_results

        except Exception as e:
            st.error(f"Reranking error: {str(e)}")
            # Fallback: return original documents with default scoring
            return [
                {"document": doc, "relevance_score": 1.0 - (i * 0.1), "index": i}
                for i, doc in enumerate(documents[:top_k])
            ]


class RAGSystem:
    def __init__(self, persist_directory: str, llm_model: str = DEFAULT_LLM_MODEL,
                 embedding_model: str = DEFAULT_EMBEDDING_MODEL):
        self.llm_model = llm_model
        self.embedding_model = embedding_model
        self.db_manager = ChromaDBManager(persist_directory, embedding_model)
        self.vector_store = self.db_manager.create_or_load_db()
        self.reranker = CohereReranker()
        self.llm = self._get_llm()

    def _get_llm(self):
        """Get the appropriate LLM based on selection."""
        if self.llm_model == "openai":
            return ChatOpenAI(temperature=0, model="gpt-4o-mini")
        elif self.llm_model.startswith("ollama:"):
            model_name = self.llm_model.replace("ollama:", "")
            return ChatOllama(model=model_name, temperature=0)
        else:
            # Default to Ollama
            return ChatOllama(model="qwen3:8b", temperature=0)

    def query(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        try:
            initial_results = self.vector_store.similarity_search(query, k=top_k * 3)

            if not initial_results:
                return {
                    "answer": "No relevant documents found.",
                    "reranked_results": [],
                    "context": "",
                }

            reranked_results = self.reranker.rerank(
                query=query,
                documents=initial_results,
                top_k=min(top_k, len(initial_results)),
            )

            top_reranked = reranked_results[:top_k]

            context_parts = []
            for i, result in enumerate(top_reranked, 1):
                doc = result["document"]
                score = result["relevance_score"]
                content = doc.page_content
                context_parts.append(f"[Document {i} (Score: {score:.3f})]:\n{content}")

            context = "\n\n".join(context_parts)

            prompt = PromptTemplate(
                template="""Based on the provided context, answer the question comprehensively.
                Include relevant quotes and cite the documents using their numbers [Doc X].
                If the information cannot be found in the context, say so.

                Context:
                {context}

                Question: {question}

                Please provide a detailed answer that:
                1. Directly addresses the question
                2. Uses specific citations [Doc X]
                3. Includes relevant quotes when appropriate
                4. Indicates confidence based on document relevance scores

                Answer:""",
                input_variables=["context", "question"],
            )

            response = self.llm.invoke(prompt.format(context=context, question=query))

            return {
                "answer": response.content,
                "reranked_results": top_reranked,
                "context": context,
            }

        except Exception as e:
            st.error(f"Query error: {str(e)}")
            import traceback
            st.error(f"Traceback: {traceback.format_exc()}")
            return {
                "answer": "An error occurred while processing your query.",
                "reranked_results": [],
                "context": "",
            }


def display_results(result):
    if not result["reranked_results"]:
        st.warning("No results found.")
        return

    if result["answer"]:
        st.markdown("### 💡 Answer")
        st.markdown(
            f"""<div style='background-color: #f0f2f6; padding: 20px; 
            border-radius: 10px;'>{result['answer']}</div>""",
            unsafe_allow_html=True,
        )

    st.markdown("### 📊 Reranking Statistics")

    scores = [doc["relevance_score"] for doc in result["reranked_results"]]
    if scores:
        cols = st.columns(3)
        with cols[0]:
            st.metric("Average Score", f"{sum(scores) / len(scores):.3f}")
        with cols[1]:
            st.metric("Max Score", f"{max(scores):.3f}")
        with cols[2]:
            st.metric("Min Score", f"{min(scores):.3f}")

    st.markdown("### 📚 Reranked Sources")

    for i, doc in enumerate(result["reranked_results"], 1):
        score = doc["relevance_score"]
        confidence = (
            "High Relevance 🟢"
            if score >= 0.7
            else "Medium Relevance 🟡" if score >= 0.4 else "Low Relevance 🔴"
        )

        with st.expander(f"Document {i} - {confidence} (Score: {score:.3f})"):
            content = doc['document'].page_content
            st.markdown(f"**Relevance Score:** {score:.3f}")
            st.markdown("**Content:**")
            st.code(content, language="text")


def get_available_ollama_models():
    """Get list of available Ollama models."""
    try:
        import requests
        response = requests.get("http://localhost:11434/api/tags", timeout=2)
        if response.status_code == 200:
            models = response.json().get("models", [])
            return [model["name"] for model in models]
        return []
    except:
        return []


def main():
    st.set_page_config(page_title="RAG with Cohere Reranking", layout="wide")
    st.title("📚 RAG System with Cohere Reranking")

    # Default model list if Ollama is not available
    default_models = ["qwen3:8b", "llama3.1:latest", "qwen3:14b", "llama3.2:latest"]

    # Get available Ollama models
    ollama_models = get_available_ollama_models()
    if not ollama_models:
        ollama_models = default_models
        st.warning("⚠️ Ollama not detected. Using default models. Make sure Ollama is running.")

    # Initialize paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_directory = os.path.join(current_dir, "data")
    persist_directory = os.path.join(current_dir, "chromadb")

    os.makedirs(data_directory, exist_ok=True)
    os.makedirs(persist_directory, exist_ok=True)

    # Sidebar for model configuration
    with st.sidebar:
        st.header("⚙️ Model Configuration")

        # LLM Model options
        llm_options = ["openai"] + [f"ollama:{model}" for model in ollama_models]

        # Default to qwen3:8b if available
        default_llm_index = 0
        for i, opt in enumerate(llm_options):
            if "qwen3:8b" in opt:
                default_llm_index = i
                break

        selected_llm = st.selectbox(
            "Select LLM Model:",
            options=llm_options,
            index=default_llm_index,
            format_func=lambda x: x.replace("ollama:", "") if x.startswith("ollama:") else "OpenAI GPT-4o-mini",
            help="Choose the model for answer generation"
        )

        # Embedding Model options
        embedding_options = ["openai"] + [f"ollama:{model}" for model in ollama_models]

        # Default to nomic-embed-text if available
        default_embedding_index = 0
        for i, opt in enumerate(embedding_options):
            if "nomic-embed-text" in opt:
                default_embedding_index = i
                break

        selected_embedding = st.selectbox(
            "Select Embedding Model:",
            options=embedding_options,
            index=default_embedding_index,
            format_func=lambda x: x.replace("ollama:", "") if x.startswith("ollama:") else "OpenAI Embeddings",
            help="Choose the model for document vectorization"
        )

        # Ollama info
        if selected_llm.startswith("ollama:") or selected_embedding.startswith("ollama:"):
            st.info("ℹ️ Make sure Ollama is running and models are pulled.")
            st.code("ollama serve", language="bash")
            st.caption("To pull a model: ollama pull <model_name>")

        st.divider()

        # Rebuild button
        if st.button("🔄 Rebuild System", type="secondary", use_container_width=True):
            st.session_state.rag_system = None
            st.cache_data.clear()
            st.rerun()

        st.divider()
        st.caption("💡 Tip: For best quality use qwen3:8b or llama3.1")

    # Initialize RAG system with selected models
    if ("rag_system" not in st.session_state or
            st.session_state.get("current_llm") != selected_llm or
            st.session_state.get("current_embedding") != selected_embedding):

        with st.spinner(f"Initializing system with {selected_llm}..."):
            try:
                rag_system = RAGSystem(
                    persist_directory,
                    llm_model=selected_llm,
                    embedding_model=selected_embedding
                )
                st.session_state.rag_system = rag_system
                st.session_state.current_llm = selected_llm
                st.session_state.current_embedding = selected_embedding

                model_display = selected_llm.replace("ollama:", "") if selected_llm.startswith("ollama:") else "OpenAI"
                st.sidebar.success(f"✅ System initialized with {model_display}")

            except Exception as e:
                st.error(f"❌ Initialization error: {str(e)}")
                st.info("Make sure Ollama is running: `ollama serve`")
                st.stop()

    # Query interface
    st.header("🔍 Query Interface")

    col1, col2 = st.columns([4, 1])
    with col1:
        query = st.text_input("Enter your question:", placeholder="e.g., What was the operating income in 2023?")
    with col2:
        top_k = st.slider("Number of documents", 1, 10, 3)

    if st.button("🔎 Search", type="primary", use_container_width=True):
        if query:
            with st.spinner("Searching and reranking..."):
                result = st.session_state.rag_system.query(query, top_k)
                display_results(result)
        else:
            st.warning("Please enter a question.")

    # Usage instructions
    with st.expander("📖 How to use"):
        st.markdown("""
        1. **Upload documents**: Place PDF files in the `data/` folder
        2. **Configure models**: Select LLM and embedding models in the sidebar
        3. **Ask a question**: Type your question in the input field
        4. **Get answers**: The system finds relevant documents, reranks them, and generates an answer

        **Recommended models:**
        - For fast responses: `qwen3:8b` or `llama3.2`
        - For quality responses: `qwen3:14b` or `llama3.1`
        - For embeddings: `nomic-embed-text` (fast and lightweight)
        """)


if __name__ == "__main__":
    main()