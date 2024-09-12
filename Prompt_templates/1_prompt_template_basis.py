import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser


load_dotenv()
google_gen_api = os.getenv("GEMINI_API")

llm = ChatGoogleGenerativeAI(model = "gemini-1.5-flash", api_key = google_gen_api, temperature = 0, top_k = 1, top_p = 1, timeout = None)

prompt = ChatPromptTemplate([
    ("system", "You are the expert about {field}"),
    ("user", "Tell me about {name}")
])

chain = prompt | llm | StrOutputParser()

print(chain.invoke({"field" : "football", "name" : "CR7"}))