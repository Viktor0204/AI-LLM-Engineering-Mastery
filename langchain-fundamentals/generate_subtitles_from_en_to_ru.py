import os
import re
from typing import List, Dict, Optional
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import time

load_dotenv()


class SubtitleTranslator:
    """
    Класс для перевода субтитров с английского на русский с помощью Ollama.
    """

    def __init__(self, model_name: str = "llama3.2"):
        self.model_name = model_name
        self.llm = ChatOllama(
            model=model_name,
            temperature=0.1,
            timeout=180,
            num_predict=2000,
        )

    def list_available_subtitles(self) -> List[str]:
        """Показывает все доступные английские субтитры в папке subtitles"""
        subtitles_dir = "subtitles"

        if not os.path.exists(subtitles_dir):
            print(f"❌ Папка '{subtitles_dir}' не найдена.")
            return []

        # Ищем файлы с английскими субтитрами (_en.srt или _en.vtt)
        en_files = [f for f in os.listdir(subtitles_dir)
                    if f.endswith(('_en.srt', '_en.vtt'))]

        if not en_files:
            print(f"❌ В папке '{subtitles_dir}' нет английских субтитров.")
            print("   Ожидаются файлы в формате: *_en.srt или *_en.vtt")
            return []

        return sorted(en_files)

    def select_subtitle_file(self, files: List[str]) -> Optional[str]:
        """Позволяет пользователю выбрать файл для перевода"""
        print("\n" + "=" * 50)
        print("📂 ДОСТУПНЫЕ СУБТИТРЫ:")
        print("=" * 50)
        for i, file in enumerate(files, 1):
            print(f"{i}. {file}")
        print("=" * 50)

        while True:
            try:
                choice = input(f"\nВыберите файл для перевода (1-{len(files)}): ").strip()
                idx = int(choice) - 1
                if 0 <= idx < len(files):
                    return files[idx]
                else:
                    print(f"❌ Введите число от 1 до {len(files)}")
            except ValueError:
                print("❌ Введите корректное число")

    def parse_subtitle_file(self, file_path: str) -> List[Dict]:
        """
        Парсит SRT или VTT файл и возвращает список сегментов.
        """
        segments = []

        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Определяем формат по расширению
        if file_path.endswith('.srt'):
            # SRT формат
            blocks = content.strip().split('\n\n')
            for block in blocks:
                lines = block.strip().split('\n')
                if len(lines) >= 3:
                    # Индекс
                    # Время
                    time_line = lines[1] if len(lines) > 1 else ''
                    # Текст
                    text_lines = lines[2:] if len(lines) > 2 else []
                    text = ' '.join(text_lines).strip()

                    if text and '-->' in time_line:
                        segments.append({
                            'time': time_line,
                            'text': text
                        })
        else:
            # VTT формат (игнорируем WEBVTT заголовок)
            lines = content.strip().split('\n')
            i = 0
            # Пропускаем WEBVTT заголовок и пустые строки
            while i < len(lines) and (lines[i].strip() == 'WEBVTT' or lines[i].strip() == ''):
                i += 1

            while i < len(lines):
                # Время
                if '-->' in lines[i]:
                    time_line = lines[i].strip()
                    i += 1
                    # Текст
                    text_lines = []
                    while i < len(lines) and lines[i].strip() != '':
                        text_lines.append(lines[i].strip())
                        i += 1
                    text = ' '.join(text_lines).strip()
                    if text:
                        segments.append({
                            'time': time_line,
                            'text': text
                        })
                else:
                    i += 1

        return segments

    def translate_segment(self, text: str) -> str:
        """Переводит один сегмент текста на русский язык"""
        prompt = ChatPromptTemplate.from_template(
            """Переведи следующий текст с английского на русский язык.
            Сохрани точный смысл и естественное звучание.
            Ответь ТОЛЬКО переводом, без дополнительных пояснений.

            Текст: {text}

            Перевод:"""
        )

        try:
            chain = prompt | self.llm
            response = chain.invoke({"text": text})
            result = response.content if hasattr(response, 'content') else str(response)
            return result.strip()
        except Exception as e:
            print(f"  ❌ Ошибка перевода: {e}")
            return text  # Возвращаем оригинал в случае ошибки

    def translate_subtitles(self, input_path: str, output_dir: str = "subtitles_ru") -> str:
        """
        Переводит все субтитры из файла на русский язык.
        """
        print(f"\n📖 Перевод субтитров: {os.path.basename(input_path)}")

        # Создаем папку для русских субтитров
        os.makedirs(output_dir, exist_ok=True)

        # Парсим субтитры
        print("  🔍 Парсинг субтитров...")
        segments = self.parse_subtitle_file(input_path)

        if not segments:
            print("  ❌ Не удалось распарсить субтитры")
            return ""

        print(f"  📝 Найдено {len(segments)} сегментов для перевода")

        # Определяем расширение исходного файла
        ext = '.srt' if input_path.endswith('.srt') else '.vtt'

        # Формируем имя выходного файла
        base_name = os.path.basename(input_path)
        # Заменяем _en на _ru
        output_name = base_name.replace('_en', '_ru')
        output_path = os.path.join(output_dir, output_name)

        print(f"  🔄 Начало перевода...")
        print("  0%", end="", flush=True)

        translated_segments = []
        total = len(segments)

        for i, segment in enumerate(segments):
            # Переводим текст
            translated_text = self.translate_segment(segment['text'])
            translated_segments.append({
                'time': segment['time'],
                'text': translated_text
            })

            # Показываем прогресс
            if (i + 1) % max(1, total // 10) == 0 or i + 1 == total:
                percent = int((i + 1) / total * 100)
                print(f"\r  {percent}%", end="", flush=True)

        print()  # Новая строка после прогресса

        # Сохраняем переведенные субтитры
        print(f"  💾 Сохранение переведенных субтитров...")
        self.save_subtitles(output_path, translated_segments, ext)

        print(f"  ✅ Перевод завершен: {output_path}")
        return output_path

    def save_subtitles(self, file_path: str, segments: List[Dict], ext: str):
        """Сохраняет переведенные субтитры в файл"""
        with open(file_path, 'w', encoding='utf-8') as f:
            if ext == '.vtt':
                f.write("WEBVTT\n\n")

            for i, segment in enumerate(segments, 1):
                if ext == '.srt':
                    f.write(f"{i}\n{segment['time']}\n{segment['text']}\n\n")
                else:
                    f.write(f"{segment['time']}\n{segment['text']}\n\n")


def select_translation_model() -> str:
    """Выбор модели для перевода"""
    print("\n" + "=" * 50)
    print("🤖 ВЫБОР МОДЕЛИ ДЛЯ ПЕРЕВОДА")
    print("=" * 50)
    print("1. llama3.2:latest (2 GB) — быстрая, стабильная")
    print("2. llama3.1:latest (4.9 GB) — хорошее качество")
    print("3. qwen3:8b (5.2 GB) — хорошее качество")
    print("4. qwen2.5-coder:7b (4.7 GB) — хорошее качество")
    print("5. gemma4:12b (7.6 GB) — высокое качество, может тормозить")
    print("6. qwen3:14b (9.3 GB) — очень высокое качество, может тормозить")
    print("=" * 50)
    print("💡 Для 16GB Mac: рекомендуются варианты 1-4")

    models = {
        '1': 'llama3.2:latest',
        '2': 'llama3.1:latest',
        '3': 'qwen3:8b',
        '4': 'qwen2.5-coder:7b',
        '5': 'gemma4:12b',
        '6': 'qwen3:14b',
    }

    while True:
        choice = input("\nВыберите модель (1-6): ").strip()
        if choice in models:
            return models[choice]
        print("❌ Неверный выбор. Введите число от 1 до 6.")


def main():
    print("\n" + "=" * 50)
    print("🎬 ПЕРЕВОД СУБТИТРОВ (EN → RU)")
    print("=" * 50)
    print("\nЭта программа переводит английские субтитры на русский язык")
    print("с помощью моделей Ollama.\n")

    # Выбор модели
    model_name = select_translation_model()
    print(f"\n✅ Выбрана модель: {model_name}")

    try:
        # Инициализация переводчика
        translator = SubtitleTranslator(model_name=model_name)

        while True:
            # Показываем доступные файлы
            files = translator.list_available_subtitles()

            if not files:
                print("\n❌ Нет доступных файлов для перевода.")
                print("Поместите английские субтитры (*_en.srt или *_en.vtt) в папку 'subtitles'")
                break

            # Выбор файла
            selected_file = translator.select_subtitle_file(files)
            if not selected_file:
                break

            input_path = os.path.join('subtitles', selected_file)

            # Перевод
            start_time = time.time()
            output_path = translator.translate_subtitles(input_path)
            elapsed = time.time() - start_time

            if output_path:
                print(f"\n⏱️ Время перевода: {elapsed:.1f} секунд")
                print(f"📁 Результат: {output_path}")

            # Спрашиваем, продолжить или выйти
            print("\n" + "-" * 50)
            cont = input("Перевести другой файл? (y/n): ").strip().lower()
            if cont != 'y':
                break

        print("\n👋 Программа завершена. До свидания!")

    except KeyboardInterrupt:
        print("\n\n⚠️ Прервано пользователем. Выход...")
    except Exception as e:
        print(f"\n❌ Ошибка: {str(e)}")


if __name__ == "__main__":
    main()