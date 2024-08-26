import os
import bs4
from dotenv import load_dotenv
from langchain import hub
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.messages import AIMessage, HumanMessage
from langchain.tools.retriever import create_retriever_tool
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()
# Gemini api key
google_api_key = os.getenv('GEMINI_API') 
langchain_api_key = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
if not os.environ.get("LANGCHAIN_API_KEY"):
    os.environ["LANGCHAIN_API_KEY"] = langchain_api_key

loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("post-content", "post-title", "post-header")
        )
    ),
)

docs = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 200)
texts = splitter.split_documents(docs)

embeddings = GoogleGenerativeAIEmbeddings(model= "models/embedding-001" ,google_api_key=google_api_key)
vectordb = FAISS.from_documents(texts, embeddings)
retriever = vectordb.as_retriever()

llm = ChatGoogleGenerativeAI(model = 'gemini-1.5-pro', max_retries= 2, timeout= None, max_tokens = None, google_api_key=google_api_key)

## -------------------------- Tool ----------------------------------- ##
'''
Các công cụ để cho agent gọi(các cộng cụ này có thể là truy xuất dữ liệu, xử lí dữ liệu, gọi API,...)
'''
tool = create_retriever_tool(
    retriever,
    "blog_post_retriever",
    "Searches and returns excerpts from the Autonomous Agents blog post.",
)
tools = [tool]
result = tool.invoke("task decomposition")
print(result)

## ------------------------- Agent Constructor -------------------------- ##
'''
Thay vì chỉ sinh text như LLM thì Agent sẽ chọn 1 trong các công cụ dựa trên yêu cầu của người dùng
'''
agent_executor = create_react_agent(llm, tools)

query = "What is Task Decomposition?"

memory = MemorySaver() 

agent_executor = create_react_agent(llm, tools, checkpointer=memory)
config = {"configurable": {"thread_id": "abc123"}}

for s in agent_executor.stream(
    {"messages": [HumanMessage(content="Hi! I'm bob")]}, config=config
):
    print(s)
    print("----")

query = "What is Task Decomposition?"

for s in agent_executor.stream(
    {"messages": [HumanMessage(content=query)]}, config=config
):
    print(s)
    print("----")

query = "What according to the blog post are common ways of doing it? redo the search"

for s in agent_executor.stream(
    {"messages": [HumanMessage(content=query)]}, config=config
):
    print(s)
    print("----")