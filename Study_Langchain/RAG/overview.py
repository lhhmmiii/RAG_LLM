import os
import bs4
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma

load_dotenv()
google_genai_api_key = os.getenv('GEMINI_API')



# Load document
loader = WebBaseLoader(
    web_paths=("https://en.wikipedia.org/wiki/List_of_career_achievements_by_Cristiano_Ronaldo",),
    # bs_kwargs=dict(
    #     parse_only=bs4.SoupStrainer(
    #         class_=("post-content", "post-title", "post-header")
    #     )
    # ),
)
docs = loader.load()

# Split
text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 200)
splits = text_splitter.split_documents(docs)

# Embedding
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=google_genai_api_key)
vectostore = Chroma.from_documents(splits, embeddings)
retriever = vectostore.as_retriever()

# Prompt
template = """
You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
Question: {question} 
Context: {context} 
Answer:
"""
prompt = PromptTemplate.from_template(template)

# LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    api_key = google_genai_api_key
)

# Post-processing
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Chain
chain = (
    {
        'question': RunnablePassthrough(),
        'context': retriever | format_docs
    }
    | prompt
    | llm
    | StrOutputParser()
)

# Question
for chunk in chain.stream('How many scores did Ronaldo make in UEFA Champions League?'):
    print(chunk, end = '|',flush=True)



