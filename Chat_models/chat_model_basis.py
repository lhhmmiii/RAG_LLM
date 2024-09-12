# Chat Model Documents: https://python.langchain.com/v0.2/docs/integrations/chat/
# OpenAI Chat Model Documents: https://python.langchain.com/v0.2/api_reference/google_genai/chat_models/langchain_google_genai.chat_models.ChatGoogleGenerativeAI.html

from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os

load_dotenv()
google_gen_api = os.getenv("GEMINI_API")

llm = ChatGoogleGenerativeAI(model = "gemini-1.5-flash", api_key = google_gen_api, temperature = 0, top_k = 1, top_p = 1, timeout = None)

res = llm.invoke("Who is Ronaldo")
print("Full Result: ", res)
print("Only content:", res.content)