import yt_dlp
import whisper
import os
import re
from typing import List, Dict, Optional
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

    def __init__(self, model_type="qwen3"):
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
        elif model_type == "nomic":
            from langchain_ollama import OllamaEmbeddings
            self.embedding_fn = OllamaEmbeddings(
                model="nomic-embed-text:latest", base_url="http://localhost:11434"
            )
        else:
            raise ValueError(f"Unsupported embedding type: {model_type}")


class LLMModel:
    """Handles different LLM models"""

    def __init__(self, model_type="ollama", model_name="llama3.1"):
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
                timeout=120,
            )
        else:
            raise ValueError(f"Unsupported LLM type: {model_type}")


class YoutubeVideoSummarizer:
    def __init__(
            self,
            llm_type="ollama",
            llm_model_name="llama3.1",
            embedding_type="qwen3",
            persist_directory="./chroma_db"
    ):
        self.embedding_model = EmbeddingModel(embedding_type)
        self.llm_model = LLMModel(llm_type, llm_model_name)
        self.whisper_model = whisper.load_model("base")
        self.subtitle_format = None
        self.persist_directory = persist_directory

        # Инициализируем единую базу знаний
        self.vector_store = None
        self.qa_chain = None
        self._initialize_knowledge_base()

    def _initialize_knowledge_base(self):
        """Initialize or load existing knowledge base"""
        print(f"Initializing knowledge base at {self.persist_directory}...")

        # Проверяем, существует ли база
        if os.path.exists(self.persist_directory) and os.listdir(self.persist_directory):
            print("Loading existing knowledge base...")
            try:
                self.vector_store = Chroma(
                    persist_directory=self.persist_directory,
                    embedding_function=self.embedding_model.embedding_fn,
                    collection_name="video_knowledge_base"
                )
                print(f"Loaded {self.vector_store._collection.count()} documents")
            except Exception as e:
                print(f"Error loading existing DB: {e}")
                self.vector_store = None
        else:
            print("Creating new knowledge base...")
            os.makedirs(self.persist_directory, exist_ok=True)

        # Если база не загружена, создаем новую
        if self.vector_store is None:
            # Инициализируем пустую коллекцию
            self.vector_store = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embedding_model.embedding_fn,
                collection_name="video_knowledge_base"
            )

        # Настраиваем QA цепочку
        self._setup_qa_chain()

    def _setup_qa_chain(self):
        """Setup QA chain with the current vector store"""
        if self.vector_store is not None:
            memory = ConversationBufferMemory(
                memory_key="chat_history", return_messages=True
            )
            self.qa_chain = ConversationalRetrievalChain.from_llm(
                llm=self.llm_model.llm,
                retriever=self.vector_store.as_retriever(search_kwargs={"k": 4}),
                memory=memory,
                verbose=True,
            )

    def get_model_info(self) -> Dict:
        return {
            "llm_type": self.llm_model.model_type,
            "llm_model": self.llm_model.model_name,
            "embedding_type": self.embedding_model.model_type,
        }

    def set_subtitle_format(self, format_choice: str):
        """Устанавливает формат субтитров"""
        self.subtitle_format = format_choice

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

            if not os.path.exists(audio_path):
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
            video_title = os.path.splitext(filename)[0]

            print(f"Processing local file: {filename}")

            # Создаем субтитры только если выбран не "0"
            if self.subtitle_format != "0":
                self.transcribe_with_timestamps(file_path, video_title)
            else:
                print("⏭️ Subtitles skipped (format '0' selected)")

            transcript = self.transcribe_audio(file_path)
            return self._process_transcript(transcript, video_title)

        except Exception as e:
            print(f"Error processing file: {str(e)}")
            return None

    def process_video(self, url: str) -> Dict:
        try:
            os.makedirs("downloads", exist_ok=True)
            audio_path, video_title = self.download_video(url)

            # Создаем субтитры только если выбран не "0"
            if self.subtitle_format != "0":
                self.transcribe_with_timestamps(audio_path, video_title)
            else:
                print("⏭️ Subtitles skipped (format '0' selected)")

            transcript = self.transcribe_audio(audio_path)

            # Удаляем аудиофайл
            try:
                if os.path.exists(audio_path):
                    os.remove(audio_path)
                    print(f"Deleted: {audio_path}")
            except Exception as e:
                print(f"Warning: Could not delete {audio_path}: {e}")

            return self._process_transcript(transcript, video_title)

        except Exception as e:
            print(f"Error processing video: {str(e)}")
            return None

    def _process_transcript(self, transcript: str, video_title: str) -> Dict:
        """Process transcript and add to knowledge base"""
        print("Creating documents...")
        documents = self.create_documents(transcript, video_title)

        # Добавляем в существующую базу
        print(f"Adding {len(documents)} chunks to knowledge base...")
        try:
            # Проверяем, что векторная база инициализирована
            if self.vector_store is None:
                raise ValueError("Vector store not initialized")

            # Добавляем документы
            self.vector_store.add_documents(documents)
            print(f"✅ Added {len(documents)} chunks to knowledge base")

            # Обновляем QA цепочку
            self._setup_qa_chain()

        except Exception as e:
            print(f"Error adding to knowledge base: {e}")
            return {"error": str(e)}

        # Генерируем суммаризацию
        summary = self.generate_summary(documents)

        return {
            "summary": summary,
            "title": video_title,
            "full_transcript": transcript,
            "chunks_added": len(documents),
        }

    def transcribe_audio(self, audio_path: str) -> str:
        print("Transcribing audio...")
        result = self.whisper_model.transcribe(audio_path)
        return result["text"]

    def transcribe_with_timestamps(self, audio_path: str, video_title: str) -> tuple[str, str]:
        """Transcribe audio, create original subtitles and translated subtitles"""
        print("Transcribing with timestamps...")

        # Получаем транскрипцию с метками времени
        result = self.whisper_model.transcribe(
            audio_path,
            word_timestamps=True
        )

        # Определяем язык оригинала
        detected_lang = result.get("language", "en")
        print(f"Detected language: {detected_lang}")

        # Создаем папку subtitles
        os.makedirs("subtitles", exist_ok=True)

        # Очищаем имя файла
        safe_title = re.sub(r'[^\w\s-]', '', video_title).strip()
        safe_title = re.sub(r'[-\s]+', '-', safe_title)

        # Определяем расширение в зависимости от выбранного формата
        ext = "srt" if self.subtitle_format == "1" else "vtt"

        original_path = f"subtitles/{safe_title}_{detected_lang}.{ext}"
        translated_path = f"subtitles/{safe_title}_ru.{ext}"

        # Получаем сегменты
        segments = self._split_into_segments(result["segments"])

        # 1. Создаем оригинальные субтитры
        if self.subtitle_format == "1":
            self._save_srt(original_path, segments)
        else:
            self._save_vtt(original_path, segments)
        print(f"✅ Original subtitles saved to: {original_path}")

        # 2. Переводим на русский через LLM
        print("Translating subtitles to Russian...")
        translated_segments = self._translate_segments(segments)

        if self.subtitle_format == "1":
            self._save_srt(translated_path, translated_segments)
        else:
            self._save_vtt(translated_path, translated_segments)
        print(f"✅ Translated subtitles saved to: {translated_path}")

        return original_path, translated_path

    def _translate_segments(self, segments):
        """Переводит сегменты на русский язык через LLM"""
        translated = []

        for i, segment in enumerate(segments):
            text = segment["text"].strip()
            if not text:
                continue

            prompt = f"Translate the following text to Russian. Output ONLY the translation, nothing else:\n\n{text}"

            try:
                response = self.llm_model.llm.invoke(prompt)
                if hasattr(response, 'content'):
                    translated_text = response.content.strip()
                else:
                    translated_text = str(response).strip()

                translated_text = translated_text.strip('"').strip("'")

                translated.append({
                    "start": segment["start"],
                    "end": segment["end"],
                    "text": translated_text
                })

                if (i + 1) % 5 == 0:
                    print(f"  Translated {i + 1}/{len(segments)} segments")

            except Exception as e:
                print(f"Error translating segment {i}: {e}")
                translated.append(segment)

        return translated

    def _split_into_segments(self, segments):
        """Разбивает аудио на смысловые сегменты по предложениям"""
        result = []

        for segment in segments:
            if len(segment["text"]) > 80:
                parts = segment["text"].split(". ")
                if len(parts) > 1:
                    for i, part in enumerate(parts):
                        if part:
                            start = segment["start"] + (i * (segment["end"] - segment["start"]) / len(parts))
                            end = start + ((segment["end"] - segment["start"]) / len(parts))
                            result.append({
                                "start": start,
                                "end": end,
                                "text": part + ("." if i < len(parts) - 1 else "")
                            })
                else:
                    result.append(segment)
            else:
                result.append(segment)

        return result

    def _save_srt(self, path: str, segments):
        """Сохраняет сегменты в SRT файл"""
        with open(path, "w", encoding="utf-8") as f:
            for i, segment in enumerate(segments, 1):
                start_time = self._format_timestamp_srt(segment["start"])
                end_time = self._format_timestamp_srt(segment["end"])
                text = segment["text"].strip()

                if text:
                    f.write(f"{i}\n")
                    f.write(f"{start_time} --> {end_time}\n")
                    f.write(f"{text}\n\n")

    def _save_vtt(self, path: str, segments):
        """Сохраняет сегменты в VTT файл"""
        with open(path, "w", encoding="utf-8") as f:
            f.write("WEBVTT\n\n")

            for segment in segments:
                start_time = self._format_timestamp_vtt(segment["start"])
                end_time = self._format_timestamp_vtt(segment["end"])
                text = segment["text"].strip()

                if text:
                    f.write(f"{start_time} --> {end_time}\n")
                    f.write(f"{text}\n\n")

    def _format_timestamp_srt(self, seconds):
        """Преобразует секунды в формат SRT (HH:MM:SS,mmm)"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        secs_int = int(secs)
        millis = int((secs - secs_int) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs_int:02d},{millis:03d}"

    def _format_timestamp_vtt(self, seconds):
        """Преобразует секунды в формат VTT (HH:MM:SS.mmm)"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        secs_int = int(secs)
        millis = int((secs - secs_int) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs_int:02d}.{millis:03d}"

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

    def generate_summary(self, documents: List[Document]) -> str:
        print("Generating summary...")
        map_prompt = ChatPromptTemplate.from_template(
            """Write a concise summary of the following transcript section:
            "{text}"
            CONCISE SUMMARY:"""
        )

        combine_prompt = ChatPromptTemplate.from_template(
            """Write a detailed summary of the following video transcript sections:
            "{text}"

            Include:
            - Main topics and key points
            - Important details and examples
            - Any conclusions or call to action

            DETAILED SUMMARY:"""
        )

        summary_chain = load_summarize_chain(
            llm=self.llm_model.llm,
            chain_type="map_reduce",
            map_prompt=map_prompt,
            combine_prompt=combine_prompt,
            verbose=True,
        )
        return summary_chain.invoke(documents)

    def ask_question(self, question: str) -> str:
        """Ask a question to the knowledge base"""
        if self.qa_chain is None:
            return "Knowledge base is not initialized. Please process some videos first."

        try:
            response = self.qa_chain.invoke({"question": question})
            return response["answer"]
        except Exception as e:
            return f"Error: {str(e)}"


def select_subtitle_format() -> str:
    """Выбор формата субтитров"""
    print("\n" + "-" * 50)
    print("SUBTITLE FORMAT:")
    print("0 - No subtitles (skip)")
    print("1 - SRT (most compatible)")
    print("2 - VTT (web standard)")
    print("-" * 50)

    while True:
        choice = input("Enter choice (0/1/2): ").strip()
        if choice in ["0", "1", "2"]:
            return choice
        print("Invalid choice. Please enter 0, 1, or 2.")


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


def select_llm_model() -> tuple[str, str]:
    """Select LLM model from available options"""
    print("\n" + "=" * 50)
    print("SELECT LLM MODEL:")
    print("1. OpenAI GPT-4 (requires API key)")
    print("2. Ollama llama3.1:latest")
    print("3. Ollama llama3.2:latest (lighter, faster)")
    print("4. Ollama qwen3:8b")
    print("5. Ollama qwen3:14b (larger, better quality)")
    print("6. Ollama gemma4:12b")
    print("7. Ollama qwen2.5-coder:7b")
    print("8. Ollama llava:latest (multimodal)")
    print("=" * 50)

    while True:
        choice = input("Enter choice (1-8): ").strip()
        if choice == "1":
            return "openai", "gpt-4"
        elif choice == "2":
            return "ollama", "llama3.1:latest"
        elif choice == "3":
            return "ollama", "llama3.2:latest"
        elif choice == "4":
            return "ollama", "qwen3:8b"
        elif choice == "5":
            return "ollama", "qwen3:14b"
        elif choice == "6":
            return "ollama", "gemma4:12b"
        elif choice == "7":
            return "ollama", "qwen2.5-coder:7b"
        elif choice == "8":
            return "ollama", "llava:latest"
        else:
            print("Invalid choice. Please enter a number between 1 and 8.")


def select_embedding_model() -> str:
    """Select embedding model from available options"""
    print("\n" + "=" * 50)
    print("SELECT EMBEDDING MODEL:")
    print("1. OpenAI Embeddings (requires API key)")
    print("2. Chroma Default (HuggingFace, lightweight)")
    print("3. Qwen3-embedding:4b (recommended for quality)")
    print("4. Nomic-embed-text:latest (fast, lightweight)")
    print("=" * 50)

    while True:
        choice = input("Enter choice (1-4): ").strip()
        if choice == "1":
            return "openai"
        elif choice == "2":
            return "chroma"
        elif choice == "3":
            return "qwen3"
        elif choice == "4":
            return "nomic"
        else:
            print("Invalid choice. Please enter a number between 1 and 4.")


def main_menu(summarizer) -> None:
    """Main interactive menu"""
    print("\n" + "=" * 50)
    print("📋 MAIN MENU")
    print("=" * 50)
    print("1. Process a new video (YouTube URL or local file)")
    print("2. Ask a question to the knowledge base")
    print("3. Show knowledge base statistics")
    print("4. Exit")
    print("=" * 50)

    while True:
        choice = input("\nEnter choice (1-4): ").strip()

        if choice == "1":
            # Process video
            source_type, source_path = select_source()
            subtitle_format = select_subtitle_format()
            summarizer.set_subtitle_format(subtitle_format)

            format_names = {"0": "None (skipped)", "1": "SRT", "2": "VTT"}
            print(f"\n📝 Subtitle format: {format_names[subtitle_format]}")

            if source_type == "local" and source_path:
                print(f"\n📁 Processing local file: {os.path.basename(source_path)}")
                result = summarizer.process_local_file(source_path)
            else:
                url = input("\nEnter YouTube URL: ").strip()
                if not url:
                    print("❌ No URL provided.")
                    continue
                print(f"\n📥 Processing video...")
                result = summarizer.process_video(url)

            if result and "error" not in result:
                print("\n" + "=" * 50)
                print(f"📹 VIDEO TITLE: {result['title']}")
                print("=" * 50)
                print("\n📝 SUMMARY:")
                print("-" * 50)
                print(result["summary"])
                print("-" * 50)
                print(f"\n✅ Added {result.get('chunks_added', 0)} chunks to knowledge base")

                if input("\n📄 Show full transcript? (y/n): ").lower() == "y":
                    print("\n" + "=" * 50)
                    print("FULL TRANSCRIPT:")
                    print("=" * 50)
                    print(result["full_transcript"])
            else:
                print("❌ Failed to process. Please check the source and try again.")

        elif choice == "2":
            # Ask question
            question = input("\n❓ Enter your question: ").strip()
            if not question:
                print("❌ No question entered.")
                continue

            print("\n🤔 Thinking...")
            answer = summarizer.ask_question(question)
            print("\n🤖 Answer:")
            print("-" * 50)
            print(answer)
            print("-" * 50)

        elif choice == "3":
            # Show stats
            print("\n" + "=" * 50)
            print("📊 KNOWLEDGE BASE STATISTICS")
            print("=" * 50)
            if summarizer.vector_store is not None:
                count = summarizer.vector_store._collection.count()
                print(f"Total documents in knowledge base: {count}")
            else:
                print("Knowledge base is empty.")

            model_info = summarizer.get_model_info()
            print(f"LLM Model: {model_info['llm_type']} ({model_info['llm_model']})")
            print(f"Embedding Model: {model_info['embedding_type']}")
            print("=" * 50)

        elif choice == "4":
            print("\n👋 Goodbye!")
            break
        else:
            print("❌ Invalid choice. Please enter 1, 2, 3, or 4.")


def main():
    print("\n" + "=" * 50)
    print("🎬 YOUTUBE VIDEO SUMMARIZER & KNOWLEDGE BASE")
    print("=" * 50)
    print("\nThis system processes YouTube videos or local files,")
    print("extracts transcripts, and builds a searchable knowledge base.")
    print("You can ask questions based on all processed content.\n")

    # Select models
    llm_type, llm_model_name = select_llm_model()
    embedding_type = select_embedding_model()

    print("\n" + "=" * 50)
    print("CONFIGURATION:")
    print(f"  LLM Model: {llm_type} ({llm_model_name})")
    print(f"  Embedding Model: {embedding_type}")
    print("=" * 50)

    try:
        # Initialize summarizer with persistent knowledge base
        summarizer = YoutubeVideoSummarizer(
            llm_type=llm_type,
            llm_model_name=llm_model_name,
            embedding_type=embedding_type,
            persist_directory="./chroma_db"
        )

        # Run main menu
        main_menu(summarizer)

    except Exception as e:
        print(f"\n❌ Error initializing system: {str(e)}")
        print("Make sure required models and APIs are properly configured.")
        print("\nTo check available Ollama models, run: ollama list")


if __name__ == "__main__":
    main()