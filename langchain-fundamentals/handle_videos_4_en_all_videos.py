import yt_dlp
import whisper
import os
import re
from typing import List, Dict, Optional
from dotenv import load_dotenv

load_dotenv()


class YoutubeVideoSubtitler:
    """
    Класс для извлечения субтитров из видео без перевода и без сохранения в ChromaDB.
    """

    def __init__(self):
        self.whisper_model = whisper.load_model("base")
        self.subtitle_format = None

    def set_subtitle_format(self, format_choice: str):
        """Устанавливает формат субтитров"""
        self.subtitle_format = format_choice

    def process_all_videos_in_downloads(self) -> List[Dict]:
        """
        Обрабатывает все видеофайлы в папке downloads по очереди.
        """
        downloads_dir = "downloads"

        if not os.path.exists(downloads_dir):
            print(f"❌ Папка '{downloads_dir}' не найдена.")
            return []

        # Поддерживаемые форматы видео и аудио
        supported_extensions = ('.mp3', '.m4a', '.mp4', '.wav', '.flac', '.mkv', '.avi', '.mov')

        # Получаем список всех файлов
        all_files = [f for f in os.listdir(downloads_dir)
                     if f.lower().endswith(supported_extensions)]

        if not all_files:
            print(f"❌ В папке '{downloads_dir}' нет поддерживаемых файлов.")
            print(f"Поддерживаемые форматы: {', '.join(supported_extensions)}")
            return []

        print(f"📁 Найдено {len(all_files)} файлов для обработки.")
        print("-" * 50)

        results = []
        total = len(all_files)

        for idx, filename in enumerate(all_files, 1):
            file_path = os.path.join(downloads_dir, filename)
            print(f"\n[{idx}/{total}] 📹 Обработка: {filename}")

            result = self.process_local_file(file_path)
            results.append(result)

            # Показываем прогресс
            if result.get("status") == "success":
                print(f"✅ Готово: {result['subtitles_path']}")
            else:
                print(f"❌ Ошибка: {result.get('error', 'Неизвестная ошибка')}")

            print("-" * 50)

        # Итоговая статистика
        success_count = sum(1 for r in results if r.get("status") == "success")
        error_count = sum(1 for r in results if r.get("status") == "error")
        skipped_count = sum(1 for r in results if r.get("status") == "skipped")

        print("\n" + "=" * 50)
        print("📊 ИТОГОВАЯ СТАТИСТИКА")
        print("=" * 50)
        print(f"✅ Успешно обработано: {success_count}")
        print(f"❌ Ошибок: {error_count}")
        print(f"⏭️ Пропущено: {skipped_count}")
        print("=" * 50)

        return results

    def process_local_file(self, file_path: str) -> Dict:
        """Process local audio/video file (MP3, M4A, MP4, etc.)"""
        try:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")

            filename = os.path.basename(file_path)
            video_title = os.path.splitext(filename)[0]

            # Создаем субтитры только если выбран не "0"
            if self.subtitle_format != "0":
                original_path = self.transcribe_with_timestamps(file_path, video_title)
                return {"status": "success", "title": video_title, "subtitles_path": original_path}
            else:
                return {"status": "skipped", "title": video_title}

        except Exception as e:
            return {"status": "error", "title": filename, "error": str(e)}

    def transcribe_with_timestamps(self, audio_path: str, video_title: str) -> str:
        """Transcribe audio and create original subtitles (no translation)"""
        print("  🔇 Транскрипция с таймкодами...")

        # Получаем транскрипцию с метками времени
        result = self.whisper_model.transcribe(
            audio_path,
            word_timestamps=True
        )

        # Определяем язык оригинала
        detected_lang = result.get("language", "en")
        print(f"  🌐 Обнаружен язык: {detected_lang}")

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


def main():
    print("\n" + "=" * 50)
    print("🎬 BATCH VIDEO SUBTITLE EXTRACTOR")
    print("=" * 50)
    print("\nЭта программа обрабатывает ВСЕ видео из папки 'downloads'")
    print("и создает для каждого английские субтитры.\n")

    # Выбор формата субтитров
    subtitle_format = select_subtitle_format()

    if subtitle_format == "0":
        print("\n❌ Выбран пропуск субтитров. Завершение программы.")
        return

    format_names = {"1": "SRT", "2": "VTT"}
    print(f"\n📝 Формат субтитров: {format_names[subtitle_format]}")
    print("=" * 50)

    try:
        # Initialize subtitler
        subtitler = YoutubeVideoSubtitler()
        subtitler.set_subtitle_format(subtitle_format)

        # Запускаем пакетную обработку
        results = subtitler.process_all_videos_in_downloads()

    except KeyboardInterrupt:
        print("\n\n⚠️ Processing interrupted by user. Exiting gracefully...")
        print("👋 Goodbye!")
    except Exception as e:
        print(f"\n❌ Error initializing system: {str(e)}")
        print("Make sure required models and APIs are properly configured.")
        print("\nTo check available Ollama models, run: ollama list")


if __name__ == "__main__":
    main()