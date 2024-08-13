# How to add message history
'''
- Việc truyền trạng thái hội thoại vào và ra khỏi chuỗi là rất quan trọng khi xây dựng chatbot. 
- Lớp RunnableWithMessageHistory cho phép chúng ta thêm lịch sử tin nhắn vào một số loại chuỗi nhất định.
- Lớp này cũng cho phép nhiều cuộc trò chuyện bằng cách lưu mỗi cuộc trò chuyện với một session_id - sau đó lớp này mong đợi một session_id được truyền vào cấu hình khi gọi runnable 
và sử dụng session_id đó để tra cứu lịch sử cuộc trò chuyện có liên quan.
'''

# Import library
import os
import numpy as np
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.messages import HumanMessage
from langchain_community.chat_message_histories import SQLChatMessageHistory
from operator import itemgetter

load_dotenv()
api_key = os.getenv("GEMINI_API")

# LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    api_key = api_key
)
