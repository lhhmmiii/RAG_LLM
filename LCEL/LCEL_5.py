# How to run custom functions
'''
Khi bạn cần dùng hàm không được cung cấp bởi Langchain, và các hàm tùy chỉnh đó được gọi bởi RunnableLambdas.
Có 1 điều chú ý ở những hàm này là chỉ có 1 đối số duy nhất. Nếu hàm có nhiều đối số thì bạn phải đóng gói 
lại thành 1 dictionary rồi sau nó unpack sau.
'''

# Import library
import os
import numpy as np
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
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

## ------------------- Using the constructor ----------------------- ##

def get_information(input):
    if input == 1:
        return "all club"
    else:
        return "all title"
    
def get_name(input):
    if input == 1:
        return "Messi"
    else:
        return "Ronaldo"


template = '''
Tell information about {type} of {name}.
'''

prompt = PromptTemplate.from_template(template)

random_index = np.random.randint(0,1)

chain = ( # Như đã nói ở LCEL 3 thì khi ta dùng itemgetter khi muốn gán nó cho thông tin phía sau
    {
      "type" :  itemgetter("type") | RunnableLambda(get_information),
       "name" :  {"input" : itemgetter("name")} | RunnableLambda(get_name) # Khi có nhiều input trong 1 hàm thì đóng ngoặc thêm dict như này(hàm của tôi chỉ có 1 input nhưng nếu có thêm input thì thêm dấu , vào rồi viết tiếp)
    }
    | prompt
    | llm
    | StrOutputParser()
)

print(chain.invoke({"type": 1, "name": 1}))
