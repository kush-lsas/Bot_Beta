import os
import streamlit as st
from langchain.callbacks.base import BaseCallbackHandler
from langchain_openai.chat_models import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import ChatPromptTemplate
from langchain_community.document_transformers import BeautifulSoupTransformer
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_community.document_transformers import BeautifulSoupTransformer
from langchain_community.vectorstores import docarray
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain.vectorstores import faiss
from langchain.docstore import in_memory
from langchain_pinecone import PineconeVectorStore

st.set_page_config(page_title="Interweb Explorer", page_icon="🌐")

OPENAI_API_KEY = st.sidebar.text_input('OpenAI API Key', type='password')
os.environ["PINECONE_API_KEY"] = "ef5d4ce9-1906-4d38-a83d-dbf18b753473"

def extract_and_split_html():
    extract_path = "Web_Pages"
    bs_transformer = BeautifulSoupTransformer()

    html_files_split = []

    for filename in os.listdir(extract_path):
        filepath = os.path.join(extract_path, filename)
        html_loader = TextLoader(filepath)
        doc = html_loader.load()
        doc_transformed = bs_transformer.transform_documents(doc, tags_to_extract=["p","li", "title", "h1", "h2", "h3","thead", "tbody", "th", "td"])
        splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 20)
        split_docs = splitter.split_documents(doc_transformed)
        for splits in split_docs:
            html_files_split.append(splits)
          
    return html_files_split

def generate_response(question):
    extract_path = "Web_Pages"
    bs_transformer = BeautifulSoupTransformer()

    html_files_split = []

    for filename in os.listdir(extract_path):
        filepath = os.path.join(extract_path, filename)
        html_loader = TextLoader(filepath)
        doc = html_loader.load()
        doc_transformed = bs_transformer.transform_documents(doc, tags_to_extract=["p","li", "title", "h1", "h2", "h3","thead", "tbody", "th", "td"])
        splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 20)
        split_docs = splitter.split_documents(doc_transformed)
        for splits in split_docs:
            html_files_split.append(splits)
    
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    #vector_store = docarray.DocArrayInMemorySearch.from_documents(html_files_split, embeddings)
    index_name = "uma-bot"
    pinecone = PineconeVectorStore.from_documents(
    html_files_split, embeddings, index_name=index_name
)

    model = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model="gpt-4-turbo-preview")
    parser = StrOutputParser()
    template = """
                The user is using STK and will ask questions about the usage of Connect commands for different use-cases
                Answer the question directly based on the learnings from the context, only providing the relevant answer without any fluff.
                Do not provide any information about the context in your answer.

                Context: {context}

                Question: {question}
                """
    prompt = ChatPromptTemplate.from_template(template)
    setup = RunnableParallel(context=pinecone.as_retriever(), question=RunnablePassthrough())

    chain = setup | prompt | model | parser

    #chain.invoke(question)

    st.info(chain.invoke(question))

st.sidebar.image("logo.webp")
st.header("`UMA Bot (Connect Commands) Beta`")
st.info("`I am an AI that can answer questions based on Connect commands that are used in STK. I can help you with the usage of Connect commands for different use-cases. Feel free to ask me anything!`")

#question = st.text_input("`Ask a question:`")

    # Write answer and sources
    #retrieval_streamer_cb = PrintRetrievalHandler(st.container())
with st.form('my_form'):
    text = st.text_area('Enter question:', 'Type here...')
    submitted = st.form_submit_button('Submit')
    if submitted:
        generate_response(text)