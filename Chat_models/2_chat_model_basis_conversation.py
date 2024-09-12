
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from dotenv import load_dotenv
import os

load_dotenv()
google_gen_api = os.getenv("GEMINI_API")

llm = ChatGoogleGenerativeAI(model = "gemini-1.5-flash", api_key = google_gen_api, temperature = 0, top_k = 1, top_p = 1, timeout = None)

messages = [
    SystemMessage("You are the math expert"), # System message là để thiết lập bối cảnh cho cuộc trò chuyện. Ví dụ như ở dưới đây là chuyên gia toán học để trả lời cho các câu hỏi về Toán học.
    HumanMessage("What is 64 divided by 8"), # Câu hỏi của user
    AIMessage("64 divided by 8 is 8"), # Câu trả lời của LLM
    HumanMessage("What is 81 divided by 3")
]

res = llm.invoke(messages)
print(res.content)