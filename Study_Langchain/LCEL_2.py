# How to stream runnables
'''
Streamm là thay vì chờ có hết kết quả thôi mới đưa ra câu trả lời thì
kết quả tới đâu nó sẽ đưa tới đó.
'''

# Import thư viện
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

# 1. The pipe operator: |
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    api_key='AIzaSyAea0CgXrjzf-dwkmC-enWbVtIwrFhG3OI'
)

