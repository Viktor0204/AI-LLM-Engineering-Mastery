import os
import re
import time
import psutil
from typing import List, Dict, Optional
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()


class SubtitleTranslator:
    """
    Класс для перевода субтитров с английского на русский с помощью Ollama.
    """

    # Модели с их характеристиками
    MODELS = {
        '1': {'name': 'llama3.2:latest', 'size_gb': 2.0, 'recommended': True, 'desc': 'быстрая, стабильная'},
        '2': {'name': 'llama3.1:latest', 'size_gb': 4.9, 'recommended': True, 'desc': 'хорошее качество'},
        '3': {'name': 'qwen3:8b', 'size_gb': 5.2, 'recommended': True, 'desc': 'хорошее качество'},
        '4': {'name': 'qwen2.5-coder:7b', 'size_gb': 4.7, 'recommended': True, 'desc': 'хорошее качество'},
        '5': {'name': 'gemma4:12b', 'size_gb': 7.6, 'recommended': False, 'desc': 'может тормозить'},
        '6': {'name': 'qwen3:14b', 'size_gb': 9.3, 'recommended': False, 'desc': 'может зависнуть на 16 ГБ'},
    }

    def __init__(self, model_name: str = "llama3.2"):
        self.model_name = model_name
        self.model_size_gb = self._get_model_size(model_name)
        self.llm = ChatOllama(
            model=model_name,
            temperature=0.1,
            timeout=300,  # увеличенный таймаут
            num_predict=2000,
        )

    def _get_model_size(self, model_name: str) -> float:
        """Возвращает размер модели в ГБ"""
        for key, info in self.MODELS.items():
            if info['name'] == model_name:
                return info['size_gb']
        return 5.0  # значение по умолчанию

    def _check_memory(self) -> bool:
        """
        Проверяет, достаточно ли памяти для выбранной модели.
        Возвращает True, если памяти достаточно.
        """
        try:
            mem = psutil.virtual_memory()
            available_gb = mem.available / (1024 ** 3)
            required_gb = self.model_size_gb + 2.0  # +2 ГБ для системы и контекста

            print(f"  📊 Доступно памяти: {available_gb:.1f} ГБ")
            print(f"  📊 Требуется для модели: {required_gb:.1f} ГБ")

            if available_gb < required_gb:
                print(f"  ⚠️ НЕДОСТАТОЧНО ПАМЯТИ! Нужно ~{required_gb:.1f} ГБ")
                print(f"  💡 Попробуйте использовать более легкую модель")
                return False
            return True
        except ImportError:
            print("  ⚠️ psutil не установлен, пропускаем проверку памяти")
            return True

    def list_available_subtitles(self) -> List[str]:
        """Показывает все доступные английские субтитры в папке subtitles"""
        subtitles_dir = "subtitles"

        if not os.path.exists(subtitles_dir):
            print(f"❌ Папка '{subtitles_dir}' не найдена.")
            return []

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
        """Парсит SRT или VTT файл и возвращает список сегментов."""
        segments = []

        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        if file_path.endswith('.srt'):
            # SRT формат
            blocks = content.strip().split('\n\n')
            for block in blocks:
                lines = block.strip().split('\n')
                if len(lines) >= 3:
                    time_line = lines[1] if len(lines) > 1 else ''
                    text_lines = lines[2:] if len(lines) > 2 else []
                    text = ' '.join(text_lines).strip()

                    if text and '-->' in time_line:
                        segments.append({
                            'time': time_line,
                            'text': text
                        })
        else:
            # VTT формат
            lines = content.strip().split('\n')
            i = 0
            while i < len(lines) and (lines[i].strip() == 'WEBVTT' or lines[i].strip() == ''):
                i += 1

            while i < len(lines):
                if '-->' in lines[i]:
                    time_line = lines[i].strip()
                    i += 1
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

    def translate_segment(self, text: str, retry_count: int = 0) -> str:
        """Переводит один сегмент текста на русский язык с повторными попытками"""
        prompt = ChatPromptTemplate.from_template(
            """Переведи следующий текст с английского на русский язык.
            Сохрани точный смысл и естественное звучание.
            Ответь ТОЛЬКО переводом, без дополнительных пояснений.

            Текст: {text}

            Перевод:"""
        )

        max_retries = 3
        try:
            chain = prompt | self.llm
            response = chain.invoke({"text": text})
            result = response.content if hasattr(response, 'content') else str(response)
            return result.strip()
        except Exception as e:
            if retry_count < max_retries:
                print(f"  ⚠️ Ошибка, повторная попытка ({retry_count + 1}/{max_retries})...")
                time.sleep(2)
                return self.translate_segment(text, retry_count + 1)
            else:
                print(f"  ❌ Ошибка перевода: {e}")
                return text

    def translate_subtitles(self, input_path: str, output_dir: str = "subtitles_ru") -> str:
        """
        Переводит все субтитры из файла на русский язык с защитой от зависаний.
        """
        print(f"\n📖 Перевод субтитров: {os.path.basename(input_path)}")

        # Проверка памяти перед началом
        if not self._check_memory():
            print("  ❌ Недостаточно памяти для перевода!")
            print("  💡 Рекомендуется выбрать более легкую модель")
            return ""

        os.makedirs(output_dir, exist_ok=True)

        print("  🔍 Парсинг субтитров...")
        segments = self.parse_subtitle_file(input_path)

        if not segments:
            print("  ❌ Не удалось распарсить субтитры")
            return ""

        total = len(segments)
        print(f"  📝 Найдено {total} сегментов для перевода")

        ext = '.srt' if input_path.endswith('.srt') else '.vtt'
        base_name = os.path.basename(input_path)
        output_name = base_name.replace('_en', '_ru')
        output_path = os.path.join(output_dir, output_name)

        print(f"  🔄 Начало перевода...")
        print("  0%", end="", flush=True)

        translated_segments = []
        failed_count = 0
        batch_size = 10  # для тяжелых моделей

        # Определяем задержку в зависимости от модели
        delay = 0.5 if self.model_size_gb > 7 else 0.1

        for i, segment in enumerate(segments):
            # Перевод с повторными попытками
            translated_text = self.translate_segment(segment['text'])

            if translated_text == segment['text'] and segment['text'].strip():
                failed_count += 1

            translated_segments.append({
                'time': segment['time'],
                'text': translated_text
            })

            # Прогресс
            if (i + 1) % max(1, total // 10) == 0 or i + 1 == total:
                percent = int((i + 1) / total * 100)
                print(f"\r  {percent}%", end="", flush=True)

            # Задержка для тяжелых моделей
            if delay > 0 and (i + 1) % batch_size == 0:
                time.sleep(delay)

        print()  # Новая строка после прогресса

        # Статистика
        success_count = total - failed_count
        print(f"\n  📊 Успешно переведено: {success_count}/{total}")
        if failed_count > 0:
            print(f"  ⚠️ Пропущено (ошибки): {failed_count}")

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
    """Выбор модели для перевода с учетом размера"""
    print("\n" + "=" * 50)
    print("🤖 ВЫБОР МОДЕЛИ ДЛЯ ПЕРЕВОДА")
    print("=" * 50)

    # Показываем модели с рекомендациями
    for key, info in SubtitleTranslator.MODELS.items():
        status = "⭐" if info['recommended'] else "⚠️"
        print(f"{key}. {info['name']} ({info['size_gb']:.1f} GB) — {info['desc']} {status}")

    print("=" * 50)
    print("💡 Для 16GB Mac: ⭐ рекомендуются варианты 1-4")
    print("⚠️ Варианты 5-6 могут привести к зависанию!")

    # Проверка свободной памяти
    try:
        mem = psutil.virtual_memory()
        available_gb = mem.available / (1024 ** 3)
        print(f"\n📊 Свободно памяти: {available_gb:.1f} ГБ")
        if available_gb < 4:
            print("⚠️ КРИТИЧЕСКИ МАЛО ПАМЯТИ! Рекомендуется выбрать вариант 1 (llama3.2)")
    except:
        pass

    while True:
        choice = input("\nВыберите модель (1-6): ").strip()
        if choice in SubtitleTranslator.MODELS:
            model_name = SubtitleTranslator.MODELS[choice]['name']
            size = SubtitleTranslator.MODELS[choice]['size_gb']

            # Предупреждение для тяжелых моделей
            if size > 7:
                print(f"⚠️ ВНИМАНИЕ: модель {model_name} ({size} ГБ) может зависнуть!")
                confirm = input("Продолжить? (y/n): ").strip().lower()
                if confirm != 'y':
                    continue
            return model_name
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
        translator = SubtitleTranslator(model_name=model_name)

        while True:
            files = translator.list_available_subtitles()

            if not files:
                print("\n❌ Нет доступных файлов для перевода.")
                print("Поместите английские субтитры (*_en.srt или *_en.vtt) в папку 'subtitles'")
                break

            selected_file = translator.select_subtitle_file(files)
            if not selected_file:
                break

            input_path = os.path.join('subtitles', selected_file)

            start_time = time.time()
            output_path = translator.translate_subtitles(input_path)
            elapsed = time.time() - start_time

            if output_path:
                print(f"\n⏱️ Время перевода: {elapsed:.1f} секунд")
                print(f"📁 Результат: {output_path}")

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