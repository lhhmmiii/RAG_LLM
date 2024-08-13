# How to use bind_tools

# Import thư viện
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.pydantic_v1 import BaseModel, Field

# LLM
load_dotenv()
api_key = os.getenv("GEMINI_API")

llm = ChatGoogleGenerativeAI(api_key=api_key, model="gemini-1.5-pro",temperature=0.9, timeout=None)

# ------------------ With prompt ------------------##
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Write out the following equation using algebraic symbols then solve it. Use the format\n\nEQUATION:...\nSOLUTION:...\n\n",
        ),
        ("human", "{equation_statement}"),
    ]
)

runnable = (
    {"equation_statement": RunnablePassthrough()} | prompt | llm | StrOutputParser()
)
dot = '_'.join('' for i in range(100))
print(dot)
print(runnable.invoke("x raised to the third plus seven equals 12"))
## -------------------- bind_tools() --------------------- #
'''
Ngoài cách sử dụng prompt thì ta có thể sử dụng bind_tools() để kiểm soát đầu ra của mô hình.
'''
class Solution(BaseModel):
    equation: str = Field(description='Equation of problem')
    solution: str = Field(description = 'How to solve equation')
    final_answer: str = Field(description="The final answer is")


print(dot)
structured_llm = llm.bind_tools(Solution)
print(structured_llm.invoke("x raised to the third plus seven equals 12"))
print(dot)
