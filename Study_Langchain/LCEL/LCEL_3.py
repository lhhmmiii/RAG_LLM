# How to invoke runnables in parallel
'''
RunableParallels ngoài hữu dụng cho việc tính toán song song, nó cũng hữu dụng khi
điều khiển đầu ra của một Runnable để khớp với định dạng đầu vào của Runnable tiếp theo
trong một chuỗi. Bạn có thể phân tách chain thành nhiều component để tính toán sau 
đó tổng hợp lại.
     Input
      / \
     /   \
 Branch1 Branch2
     \   /
      \ /
      Combine
'''

# Import thư viện
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_community.vectorstores import FAISS


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

## -------------------- Formatting with RunnableParallels ---------------##
vectorstore = FAISS.from_texts(
    ["harrison worked at kensho"], embedding=GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key = api_key)
)

retriever = vectorstore.as_retriever()
template = """Answer the question based only on the following context:
{context}

Question: {question}
"""

prompt = ChatPromptTemplate.from_template(template)

retrieval_chain = (
    {"context": retriever , "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

text = retrieval_chain.invoke("where did harrison work?")
print(text)

## ---------------- Using itemgetter as shorthand ----------------- ##

from operator import itemgetter

template = """Answer the question based only on the following context:
{context}

Question: {question}

Answer in the following language: {language}
"""

prompt = ChatPromptTemplate.from_template(template)
chain = (
    {
        "context" : itemgetter("question") | retriever, # Sử dụng khi bạn có retriever, nếu không có thì không cần mất công như vậy, chỉ cần invoke là xong.
        "question": itemgetter("question"),
        "language": itemgetter("language")
    }
    | prompt
    | llm
    | StrOutputParser()
)

text2 = chain.invoke({"question": "where did harrison work", "language": "Vietnamese"})
print(text2)

## ---------------- Parallelize steps ---------------------- ##

chain1 = ChatPromptTemplate.from_template("Tell me some pieces of personal information about {name}") | llm | StrOutputParser()
chain2 = ChatPromptTemplate.from_template("List of club and {thing} which {name} played") | llm | StrOutputParser()
map_chain = RunnableParallel(info = chain1, club = chain2) # Phải đặt tên cho mỗi chain

text3 = map_chain.invoke({"name": "CR7", "thing": "salary"})
print(text3)