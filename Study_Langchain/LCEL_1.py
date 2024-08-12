# How to: chain runnables
'''
Một trong những điểm mạnh của LCEL là 2 Runnables bất kì đều có thể gắn với nhau được.
Output của runables trước là input của runnable tiếp theo. 
Điều này có thể được thực hiện bằng cách sử dụng toán tử (|)
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

prompt = ChatPromptTemplate.from_template("tell me a joke about {topic}")

chain = prompt | llm | StrOutputParser()

# text = chain.invoke({"topic": "cristiano ronaldo"})

# print(text)

# 2. Coercion
'''
Dưới đây là cách kết hợp nhiều runables với nhau để tạo ra 1 chain khác.
'''

another_prompt = ChatPromptTemplate.from_template("Is this a funny joke? {joke}")

# composed_chain = {'joke': chain} | another_prompt | llm | StrOutputParser() # Cách 1
composed_chain = chain | (lambda input : {'joke': input}) | another_prompt | llm | StrOutputParser() # Cách 2

text = composed_chain.invoke({"topic" : "bears"})
print(text)
