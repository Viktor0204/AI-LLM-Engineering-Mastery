# from langchain.agents import create_agent
# from langchain_ollama import ChatOllama
#
# model = ChatOllama(model="llama3.1")
#
# def get_weather(city: str) -> str:
#     """Get weather for a given city."""
#     print(f"[DEBUG] Вызвана функция get_weather с городом: {city}")  # Отладка
#     return f"It's always sunny in {city}!"
#
# agent = create_agent(
#     model=model,
#     tools=[get_weather],
#     system_prompt="You are a helpful assistant",
# )
#
# # result = agent.invoke(
# #     {"messages": [{"role": "user", "content": "What's the weather in San Francisco?"}]}
# # )
# # print(result["messages"][-1].content_blocks)
#
#
# result = agent.invoke(
#     {"messages": [{"role": "user", "content": "What's the weather in San Francisco?"}]},
#     config={"recursion_limit": 10}
# )
#
# final_message = result["messages"][-1]
# print(final_message.content)

from langchain.agents import create_agent
from langchain_ollama import ChatOllama
from urllib3 import response

model = ChatOllama(model="llama3.1")

def get_weather(city: str) -> str:
    """Get weather for a given city."""
    print(f"[DEBUG] Вызвана функция get_weather с городом: {city}")
    return f"It's always sunny in {city}!"

agent = create_agent(
    model=model,
    tools=[get_weather],
    system_prompt="You are a helpful assistant",
)

# result = agent.invoke(
#     {"messages": [{"role": "user", "content": "What's the weather in San Francisco?"}]},
#     config={"recursion_limit": 10}
# )

# final_message = result["messages"][-1]
# print(final_message.content)




from langchain_core.prompts import ChatPromptTemplate

system_template = "Translate the following from English into {language}"

prompt_template = ChatPromptTemplate.from_messages([
    ("system", system_template),
    ("user", "{text}")
])

prompt = prompt_template.invoke({"language": "Italian", "text": "hi!"})
# print(prompt)

response = model.invoke(prompt)
print(response.content)