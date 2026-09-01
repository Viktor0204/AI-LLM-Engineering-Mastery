import yt_dlp
import whisper
import os
import re
from typing import List, Dict, Optional

from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_ollama import ChatOllama
from langchain_chroma import Chroma
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.memory import ConversationBufferMemory
from langchain_classic.chains.summarize import load_summarize_chain
from langchain_core.documents import Document
from dotenv import load_dotenv

load_dotenv()


class EmbeddingModel:
    """Handles different embedding models"""

    def __init__(self, model_type="openai"):
        self.model_type = model_type
        if model_type == "openai":
            self.embedding_fn = OpenAIEmbeddings(
                model="text-embedding-3-small",
                openai_api_key=os.getenv("OPENAI_API_KEY"),
            )
        elif model_type == "chroma":
            from langchain_community.embeddings import HuggingFaceEmbeddings
            self.embedding_fn = HuggingFaceEmbeddings()
        elif model_type == "qwen3":
            from langchain_ollama import OllamaEmbeddings
            self.embedding_fn = OllamaEmbeddings(
                model="qwen3-embedding:4b", base_url="http://localhost:11434"
            )
        else:
            raise ValueError(f"Unsupported embedding type: {model_type}")


class LLMModel:
    """Handles different LLM models"""

    def __init__(self, model_type="openai", model_name="gpt-4"):
        self.model_type = model_type
        self.model_name = model_name

        if model_type == "openai":
            if not os.getenv("OPENAI_API_KEY"):
                raise ValueError("OpenAI API key is required for OpenAI models")
            self.llm = ChatOpenAI(model_name=model_name, temperature=0)
        elif model_type == "ollama":
            self.llm = ChatOllama(
                model=model_name,
                temperature=0,
                # format="json",
                timeout=120,
                num_predict=2000,
            )
        else:
            raise ValueError(f"Unsupported LLM type: {model_type}")


class YoutubeVideoSummarizer:
    def __init__(
            self, llm_type="openai", llm_model_name="gpt-4", embedding_type="openai"
    ):
        self.embedding_model = EmbeddingModel(embedding_type)
        self.llm_model = LLMModel(llm_type, llm_model_name)
        self.whisper_model = whisper.load_model("base")

    def get_model_info(self) -> Dict:
        return {
            "llm_type": self.llm_model.model_type,
            "llm_model": self.llm_model.model_name,
            "embedding_type": self.embedding_model.model_type,
        }

    def download_video(self, url: str) -> tuple[str, str]:
        """Download video and extract audio"""
        print("Downloading video...")
        os.makedirs("downloads", exist_ok=True)

        ydl_opts = {
            "format": "bestaudio/best",
            "postprocessors": [
                {
                    "key": "FFmpegExtractAudio",
                    "preferredcodec": "mp3",
                    "preferredquality": "192",
                }
            ],
            "outtmpl": "downloads/%(id)s.%(ext)s",
            "ignorecertificate": True,
            "no_check_certificate": True,
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            video_id = info.get("id", "video")
            video_title = info.get("title", "Unknown Title")
            audio_path = f"downloads/{video_id}.mp3"

            # Проверяем, что файл создался
            if not os.path.exists(audio_path):
                # Ищем любой MP3 в папке
                mp3_files = [f for f in os.listdir("downloads") if f.endswith(".mp3")]
                if mp3_files:
                    audio_path = os.path.join("downloads", mp3_files[-1])

            return audio_path, video_title

    def process_local_file(self, file_path: str) -> Dict:
        """Process local audio/video file (MP3, M4A, MP4, etc.)"""
        try:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")

            filename = os.path.basename(file_path)
            video_title = os.path.splitext(filename)[0]  # убираем расширение

            print(f"Processing local file: {filename}")

            # Whisper работает с большинством аудио/видео форматов через ffmpeg
            transcript = self.transcribe_audio(file_path)
            documents = self.create_documents(transcript, video_title)
            summary = self.generate_summary(documents)
            vector_store = self.create_vector_store(documents)
            qa_chain = self.setup_qa_chain(vector_store)

            return {
                "summary": summary,
                "qa_chain": qa_chain,
                "title": video_title,
                "full_transcript": transcript,
            }
        except Exception as e:
            print(f"Error processing file: {str(e)}")
            return None

    def transcribe_audio(self, audio_path: str) -> str:
        print("Transcribing audio...")
        result = self.whisper_model.transcribe(audio_path)
        return result["text"]

    def create_documents(self, text: str, video_title: str) -> List[Document]:
        print("Creating documents...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=100, separators=["\n\n", "\n", ". ", " ", ""]
        )
        texts = text_splitter.split_text(text)
        return [
            Document(page_content=chunk, metadata={"source": video_title})
            for chunk in texts
        ]

    def create_vector_store(self, documents: List[Document]) -> Chroma:
        print(f"Creating vector store using {self.embedding_model.model_type} embeddings...")
        return Chroma.from_documents(
            documents=documents,
            embedding=self.embedding_model.embedding_fn,
            collection_name=f"youtube_summary_{self.embedding_model.model_type}",
        )


    def generate_summary(self, documents: List[Document]) -> str:
        print("Generating summary...")

        # 1. Ограничиваем количество чанков
        docs_to_use = documents[:3] if len(documents) > 3 else documents
        full_text = "\n\n".join([doc.page_content for doc in docs_to_use])

        # 2. Жестко ограничиваем длину (первые 2000 символов)
        if len(full_text) > 2000:
            full_text = full_text[:2000] + "..."

        # 3. Простой промпт
        prompt = f"""Summarize this text concisely (3-5 sentences):
    
        {full_text}
    
        Summary:"""

        # 4. Вызов с таймаутом
        try:
            response = self.llm_model.llm.invoke(prompt)
            return response.content
        except Exception as e:
            print(f"Summary error: {e}")
            return "Summary generation failed. Please try again."

    def setup_qa_chain(self, vector_store: Chroma):
            memory = ConversationBufferMemory(
                memory_key="chat_history", return_messages=True
            )
            return ConversationalRetrievalChain.from_llm(
                llm=self.llm_model.llm,
                retriever=vector_store.as_retriever(),
                memory=memory,
                verbose=True,
            )

    def process_video(self, url: str) -> Dict:
        try:
            os.makedirs("downloads", exist_ok=True)
            audio_path, video_title = self.download_video(url)
            transcript = self.transcribe_audio(audio_path)
            documents = self.create_documents(transcript, video_title)
            summary = self.generate_summary(documents)
            vector_store = self.create_vector_store(documents)
            qa_chain = self.setup_qa_chain(vector_store)

            # Удаление с проверкой
            try:
                if os.path.exists(audio_path):
                    os.remove(audio_path)
                    print(f"Deleted: {audio_path}")
            except Exception as e:
                print(f"Warning: Could not delete {audio_path}: {e}")

            return {
                "summary": summary,
                "qa_chain": qa_chain,
                "title": video_title,
                "full_transcript": transcript,
            }
        except Exception as e:
            print(f"Error processing video: {str(e)}")
            return None


def select_source() -> tuple[str, Optional[str]]:
    """Select source: YouTube URL or local file"""
    print("\n" + "=" * 50)
    print("SELECT SOURCE:")
    print("1. YouTube URL")
    print("2. Local file (MP3, M4A, MP4)")
    print("=" * 50)

    choice = input("Enter choice (1/2): ").strip()

    if choice == "2":
        downloads_dir = "downloads"
        if not os.path.exists(downloads_dir):
            os.makedirs(downloads_dir, exist_ok=True)
            print("No files found. Downloads folder is empty.")
            return "youtube", None

        # Поддерживаемые форматы
        supported_extensions = ('.mp3', '.m4a', '.mp4', '.wav', '.flac')
        files = [f for f in os.listdir(downloads_dir)
                 if f.lower().endswith(supported_extensions)]

        if not files:
            print("No supported files found (MP3, M4A, MP4, WAV, FLAC)")
            return "youtube", None

        print("\nAvailable files:")
        for i, file in enumerate(files, 1):
            file_path = os.path.join(downloads_dir, file)
            file_size = os.path.getsize(file_path) / (1024 * 1024)
            print(f"  {i}. {file} ({file_size:.1f} MB)")

        while True:
            try:
                file_choice = input(f"\nChoose file (1-{len(files)}): ").strip()
                idx = int(file_choice) - 1
                if 0 <= idx < len(files):
                    audio_path = os.path.join(downloads_dir, files[idx])
                    return "local", audio_path
                else:
                    print(f"Please enter a number between 1 and {len(files)}")
            except ValueError:
                print("Please enter a valid number")

    return "youtube", None


def main():
    print("\n" + "=" * 50)
    print("🎬 YOUTUBE VIDEO SUMMARIZER & Q&A SYSTEM")
    print("=" * 50)

    # Выбор моделей
    print("\nAvailable LLM Models:")
    print("1. OpenAI GPT-4")
    print("2. Ollama Llama3.1")
    llm_choice = input("Choose LLM model (1/2): ").strip()

    print("\nAvailable Embeddings:")
    print("1. OpenAI")
    print("2. Chroma Default")
    print("3. Qwen3 (via Ollama)")
    embedding_choice = input("Choose embeddings (1/2/3): ").strip()

    llm_type = "openai" if llm_choice == "1" else "ollama"
    llm_model_name = "gpt-4" if llm_choice == "1" else "llama3.1"

    embedding_map = {"1": "openai", "2": "chroma", "3": "qwen3"}
    embedding_type = embedding_map.get(embedding_choice, "qwen3")

    try:
        summarizer = YoutubeVideoSummarizer(
            llm_type=llm_type,
            llm_model_name=llm_model_name,
            embedding_type=embedding_type,
        )

        model_info = summarizer.get_model_info()
        print("\n" + "=" * 50)
        print("CONFIGURATION:")
        print(f"  LLM: {model_info['llm_type']} ({model_info['llm_model']})")
        print(f"  Embeddings: {model_info['embedding_type']}")
        print("=" * 50)

        # Выбор источника
        source_type, source_path = select_source()

        if source_type == "local" and source_path:
            print(f"\n📁 Processing local file: {os.path.basename(source_path)}")
            result = summarizer.process_local_file(source_path)
        else:
            url = input("\nEnter YouTube URL: ").strip()
            print(f"\n📥 Processing video...")
            result = summarizer.process_video(url)

        if result:
            print("\n" + "=" * 50)
            print(f"📹 VIDEO TITLE: {result['title']}")
            print("=" * 50)

            print("\n📝 SUMMARY:")
            print("-" * 50)
            print(result["summary"])
            print("-" * 50)

            # Interactive Q&A
            print("\n💬 Q&A MODE (type 'quit' to exit)")
            while True:
                query = input("\n❓ Your question: ").strip()
                if query.lower() == "quit":
                    break
                if query:
                    try:
                        response = result["qa_chain"].invoke({"question": query})
                        print(f"\n🤖 Answer: {response['answer']}")
                    except Exception as e:
                        print(f"Error: {e}")

            # Full transcript
            if input("\n📄 Show full transcript? (y/n): ").lower() == "y":
                print("\n" + "=" * 50)
                print("FULL TRANSCRIPT:")
                print("=" * 50)
                print(result["full_transcript"])

        else:
            print("❌ Failed to process. Please check the source and try again.")

    except Exception as e:
        print(f"❌ Error: {str(e)}")
        print("Make sure required models and APIs are properly configured.")


if __name__ == "__main__":
    main()