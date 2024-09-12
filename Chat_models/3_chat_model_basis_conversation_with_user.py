import os
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()
google_gen_ai_api = os.getenv("GEMINI_API")

llm = ChatGoogleGenerativeAI(model = "gemini-1.5-flash", timeout = None, temperature = 0, top_k = 1, top_p = 1, api_key = google_gen_ai_api)

chat_history = []

system_message = SystemMessage("You know every football in the world")
chat_history.append(system_message)

while True:
    query = input("User: ")
    if query == "exit":
        break
    chat_history.append(HumanMessage(query))
    res = llm.invoke(query)
    print("AI: ", res.content)
    chat_history.append(AIMessage(res.content))