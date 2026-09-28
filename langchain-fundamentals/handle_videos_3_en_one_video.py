import yt_dlp
import whisper
import os
import re
from typing import List, Dict, Optional
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()


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


class YoutubeVideoSubtitler:
    """
    Класс для извлечения субтитров из видео без перевода и без сохранения в ChromaDB.
    """

    def __init__(
            self,
            llm_type="ollama",
            llm_model_name="llama3.1",
    ):
        self.llm_model = LLMModel(llm_type, llm_model_name)
        self.whisper_model = whisper.load_model("base")
        self.subtitle_format = None

    def get_model_info(self) -> Dict:
        return {
            "llm_type": self.llm_model.model_type,
            "llm_model": self.llm_model.model_name,
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
                original_path = self.transcribe_with_timestamps(file_path, video_title)
                print(f"✅ Subtitles saved to: {original_path}")
                return {"status": "success", "title": video_title, "subtitles_path": original_path}
            else:
                print("⏭️ Subtitles skipped (format '0' selected)")
                return {"status": "skipped", "title": video_title}

        except Exception as e:
            print(f"Error processing file: {str(e)}")
            return {"status": "error", "error": str(e)}

    def process_video(self, url: str) -> Dict:
        """Process video from URL and extract subtitles"""
        try:
            os.makedirs("downloads", exist_ok=True)
            audio_path, video_title = self.download_video(url)

            # Создаем субтитры только если выбран не "0"
            if self.subtitle_format != "0":
                original_path = self.transcribe_with_timestamps(audio_path, video_title)
                print(f"✅ Subtitles saved to: {original_path}")
                result = {"status": "success", "title": video_title, "subtitles_path": original_path}
            else:
                print("⏭️ Subtitles skipped (format '0' selected)")
                result = {"status": "skipped", "title": video_title}

            # Удаляем аудиофайл
            try:
                if os.path.exists(audio_path):
                    os.remove(audio_path)
                    print(f"Deleted: {audio_path}")
            except Exception as e:
                print(f"Warning: Could not delete {audio_path}: {e}")

            return result

        except Exception as e:
            print(f"Error processing video: {str(e)}")
            return {"status": "error", "error": str(e)}

    def transcribe_audio(self, audio_path: str) -> str:
        """Transcribe audio and return text"""
        print("Transcribing audio...")
        result = self.whisper_model.transcribe(audio_path)
        return result["text"]

    def transcribe_with_timestamps(self, audio_path: str, video_title: str) -> str:
        """Transcribe audio and create original subtitles (no translation)"""
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

        # Получаем сегменты
        segments = self._split_into_segments(result["segments"])

        # Создаем оригинальные субтитры
        if self.subtitle_format == "1":
            self._save_srt(original_path, segments)
        else:
            self._save_vtt(original_path, segments)

        return original_path

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
    print("2. Ollama llama3.1:latest (recommended, ~5GB)")
    print("3. Ollama llama3.2:latest (light, ~2GB, fast)")
    print("4. Ollama qwen3:8b")
    print("5. Ollama qwen3:14b (large, ~9GB, best quality)")
    print("6. Ollama gemma4:12b (large, ~7.6GB)")
    print("7. Ollama qwen2.5-coder:7b")
    print("8. Ollama llava:latest (multimodal)")
    print("=" * 50)
    print("💡 For 16GB Mac: Choose 2, 3, or 4 for best performance")

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


def main_menu(subtitler) -> None:
    """Main interactive menu"""

    while True:
        print("\n" + "=" * 50)
        print("📋 MAIN MENU")
        print("=" * 50)
        print("1. Extract subtitles from video (YouTube URL or local file)")
        print("2. Exit")
        print("=" * 50)
        choice = input("\nEnter choice (1-2): ").strip()

        if choice == "1":
            # Process video
            source_type, source_path = select_source()
            subtitle_format = select_subtitle_format()
            subtitler.set_subtitle_format(subtitle_format)

            format_names = {"0": "None (skipped)", "1": "SRT", "2": "VTT"}
            print(f"\n📝 Subtitle format: {format_names[subtitle_format]}")

            if subtitle_format == "0":
                print("❌ Subtitles skipped. Please select a valid format next time.")
                continue

            if source_type == "local" and source_path:
                print(f"\n📁 Processing local file: {os.path.basename(source_path)}")
                result = subtitler.process_local_file(source_path)
            else:
                url = input("\nEnter YouTube URL: ").strip()
                if not url:
                    print("❌ No URL provided.")
                    continue
                print(f"\n📥 Processing video...")
                result = subtitler.process_video(url)

            if result.get("status") == "success":
                print("\n" + "=" * 50)
                print(f"📹 VIDEO TITLE: {result['title']}")
                print("=" * 50)
                print(f"✅ Subtitles saved to: {result['subtitles_path']}")
                print("=" * 50)
            elif result.get("status") == "error":
                print(f"❌ Error: {result.get('error')}")
            else:
                print("❌ Failed to process. Please check the source and try again.")

        elif choice == "2":
            print("\n👋 Goodbye!")
            break
        else:
            print("❌ Invalid choice. Please enter 1 or 2.")


def main():
    print("\n" + "=" * 50)
    print("🎬 YOUTUBE VIDEO SUBTITLE EXTRACTOR")
    print("=" * 50)
    print("\nThis system extracts subtitles from YouTube videos or local files.")
    print("Subtitles are saved in SRT or VTT format without translation.\n")
    print("💡 Press Ctrl+C at any time to stop processing\n")

    # Select LLM model (используется только для определения языка)
    llm_type, llm_model_name = select_llm_model()

    print("\n" + "=" * 50)
    print("CONFIGURATION:")
    print(f"  LLM Model: {llm_type} ({llm_model_name})")
    print("=" * 50)

    try:
        # Initialize subtitler
        subtitler = YoutubeVideoSubtitler(
            llm_type=llm_type,
            llm_model_name=llm_model_name,
        )

        # Run main menu
        main_menu(subtitler)

    except KeyboardInterrupt:
        print("\n\n⚠️ Processing interrupted by user. Exiting gracefully...")
        print("👋 Goodbye!")
    except Exception as e:
        print(f"\n❌ Error initializing system: {str(e)}")
        print("Make sure required models and APIs are properly configured.")
        print("\nTo check available Ollama models, run: ollama list")


if __name__ == "__main__":
    main()