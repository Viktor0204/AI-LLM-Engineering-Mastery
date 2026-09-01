import os
from typing import List
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate

load_dotenv()


class QueryExpander:
    """
    A class to expand a single query into multiple semantically similar variations
    to improve retrieval coverage.
    """

    def __init__(self, temperature: float = 0.3, model_name: str = "llama3.1"):
        """
        Initialize the QueryExpander.

        Args:
            temperature: Controls randomness in LLM response. Lower values make output more focused.
            model_name: Name of the Ollama model to use (default: llama3.1)
        """
        self.llm = ChatOllama(
            model=model_name,
            temperature=temperature,
            timeout=120,
            num_predict=500,  # ограничиваем длину ответа
        )

        # Prompt template for query expansion
        self.query_expansion_prompt = PromptTemplate(
            input_variables=["question"],
            template="""Given the following question, generate 3 different versions of the question 
            that capture different aspects and perspectives of the original question. 
            Make the variations semantically diverse but relevant.

            Original Question: {question}

            Generate variations in the following format:
            1. [First variation]
            2. [Second variation]
            3. [Third variation]

            Only output the numbered variations, nothing else.""",
        )

    def expand_query(self, question: str) -> List[str]:
        """
        Expand a single query into multiple variations.

        Args:
            question: The original question to expand.

        Returns:
            List of query variations including the original question.

        Example:
            Original: "What are the effects of climate change?"
            Expanded: [
                "How does global warming impact our environment?",
                "What are the consequences of rising temperatures on Earth?",
                "What environmental changes are caused by greenhouse gases?",
            ]
        """
        try:
            # Get variations from LLM
            response = self.llm.invoke(
                self.query_expansion_prompt.format(question=question)
            )

            if hasattr(response, 'content'):
                content = response.content
            else:
                content = str(response)

            # Parse numbered list from response
            variations = []
            for line in content.strip().split("\n"):
                if line.strip() and ". " in line:
                    try:
                        variations.append(line.split(". ", 1)[1])
                    except:
                        continue

            # Если не удалось распарсить, пробуем другой формат
            if not variations:
                # Просто берем все строки с цифрами
                for line in content.strip().split("\n"):
                    if line.strip().startswith(tuple("123")):
                        variations.append(line.split(".", 1)[-1].strip())

            # Если всё равно пусто — возвращаем оригинал
            if not variations:
                return [question]

            # Add original question to variations
            variations.append(question)

            return variations

        except Exception as e:
            print(f"Error in query expansion: {e}")
            # If there's an error, return just the original question
            return [question]


def main():
    """
    Example usage of QueryExpander
    """
    # Initialize QueryExpander с Ollama
    expander = QueryExpander(model_name="llama3.1") 

    # Example questions
    questions = [
        "What are the main causes of global warming?",
        "How does exercise affect mental health?",
        "What are the benefits of renewable energy?",
    ]

    # Test query expansion
    for original_question in questions:
        print(f"\nOriginal Question: {original_question}")
        print("Expanded Queries:")

        expanded_queries = expander.expand_query(original_question)

        for i, query in enumerate(expanded_queries, 1):
            if query != original_question:
                print(f"{i}. {query}")

        print(f"Original: {original_question}")


if __name__ == "__main__":
    main()