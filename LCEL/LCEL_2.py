# How to stream runnables
'''
Streamm là thay vì chờ có hết kết quả thôi mới đưa ra câu trả lời thì
kết quả tới đâu nó sẽ đưa tới đó.
Có 2 cách tiếp cận thông thường để stream nội dung:
1. sync stream và async astream: cách triển khai mặc định của stream đầu ra.
2. async astream_events và async astream_log: chúng cung cấp một cách để truyền trực tiếp các bước trung gian và đầu ra cuối cùng từ chuỗi.
(Đang trong giai đoạn beta nên tôi sẽ quay lại sau)
'''

# Import thư viện
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
import os
from dotenv import load_dotenv
import asyncio

load_dotenv()
api_key = os.getenv('GEMINI_API')

# LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    api_key = api_key
)

####-------------------Using Stream-----------------####
'''
Mọi runnable đều có thể dùng sync stream và async astream. Phương pháp được thiết kế để stream theo từng chunk.
'''

chunks = []
for chunk in llm.stream("what color is the sky"):
    chunks.append(chunk)
    print(chunk.content, end = '|', flush = True) # Tham số flush=True trong hàm print() trong Python được sử dụng để đảm bảo đầu ra được ghi ngay vào terminal hoặc luồng đầu ra đã chỉ định, thay vì được lưu vào bộ đệm

# Nếu bạn đang triển khai trên môi trường đồng bộ thì đoạn code sẽ như sau:
# async def generate_test():
#     chunks = []
#     for chunk in llm.astream("Please description a dog"):
#         chunks.append(chunk)
#         print(chunk.content, end = '|', flush = True)

# prompt = ChatPromptTemplate.from_template('Give me information about {name}.')
# chain = prompt | llm | StrOutputParser() 

# for chunk in chain.stream({"name": 'CR7'}):
#     print(chunk, end = '|', flush = True)

## -------- Non-streaming components --------------- ##
'''
Một số thành phần như Retriever không cung cấp chức năng streaming. Nếu ta vẫn cố
stream thì sẽ không có ích, nó chỉ đưa ra kết quả cuối cùng
'''

