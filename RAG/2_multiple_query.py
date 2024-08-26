'''
MultiQueryRetriever tự động hóa quá trình điều chỉnh prompt bằng cách sử dụng LLM để tạo nhiều truy vấn từ nhiều góc nhìn khác nhau cho một truy vấn đầu vào của người dùng nhất 
định. Đối với mỗi truy vấn, nó sẽ truy xuất một tập hợp các tài liệu có liên quan và lấy liên kết duy nhất trên tất cả các truy vấn để có được một tập hợp lớn hơn các tài liệu có 
khả năng liên quan. Bằng cách tạo ra nhiều góc nhìn về cùng một câu hỏi, MultiQueryRetriever có thể khắc phục một số hạn chế của truy xuất dựa trên khoảng cách và có được một tập 
hợp kết quả phong phú hơn.
'''

# Import thư viện
import os
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.retrievers.multi_query import MultiQueryRetriever

load_dotenv()
google_genai_api_key = os.getenv('GEMINI_API')

# LLM
llm = ChatGoogleGenerativeAI(model='gemini-pro-1.5', api_key=google_genai_api_key)


# Load blog post
loader = WebBaseLoader("https://en.wikipedia.org/wiki/List_of_career_achievements_by_Cristiano_Ronaldo")
docs = loader.load()

# Split
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
splits = text_splitter.split_documents(docs)

# VectorDB
embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=google_genai_api_key)
vectordb = Chroma.from_documents(documents=splits, embedding=embedding)

template = """You are an AI language model assistant. Your task is to generate five 
different versions of the given user question to retrieve relevant documents from a vector 
database. By generating multiple perspectives on the user question, your goal is to help
the user overcome some of the limitations of the distance-based similarity search. 
Provide these alternative questions separated by newlines. 
Original question: {question}
"""
prompt_perspectives = PromptTemplate.from_template(template)

generate_queries = (
    prompt_perspectives 
    | llm
    | StrOutputParser()
)

for chunk in generate_queries.stream('How many scores did Ronaldo make in UEFA Champions League?'):
    print(chunk, end = '|', flush = True)