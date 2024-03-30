import os
import streamlit as st
from langchain_openai.chat_models import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import ChatPromptTemplate
from langchain.embeddings.openai import OpenAIEmbeddings

from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_pinecone import PineconeVectorStore

st.set_page_config(page_title="Interweb Explorer", page_icon="🌐")

OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]

def generate_embeddings():
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    index_name = "uma-bot"
    pcstore = PineconeVectorStore.from_existing_index(index_name, embeddings)
    return pcstore.as_retriever()

def setup_chain():
    model = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model="gpt-4-turbo-preview", streaming=True)
    parser = StrOutputParser()
    template = """
                You are an AI assistant helping users with STK Connect commands for various use-cases. Your goal is to provide accurate, concise, and relevant answers based on the context provided.

                Follow these guidelines when answering:
                1. Carefully analyze the question and the context to understand the user's specific needs.
                2. Provide a direct and targeted answer, focusing only on the relevant information.
                3. If the question lacks clarity or requires additional details, ask for clarification before providing an answer.
                4. If the context does not contain sufficient information to answer the question confidently, indicate that you don't have enough information and suggest resources or documentation where the user can find more details.
                5. Use clear and precise language, avoiding ambiguity or unnecessary technical jargon.
                6. Do not include any information about the context itself in your answer.

                Context: {context}

                Question: {question}
                """
    prompt = ChatPromptTemplate.from_template(template)
    retriever = generate_embeddings()
    setup = RunnableParallel(context=retriever, question=RunnablePassthrough())
    chain = setup | prompt | model | parser
    return chain

st.sidebar.image("logo.webp")
st.header("`UMA Bot (Connect Commands) Beta`")
st.info("`I am an AI that can answer questions based on Connect commands that are used in STK. I can help you with the usage of Connect commands for different use-cases. Feel free to ask me anything!`")

with st.form('my_form'):
    text = st.text_area('Enter question:', 'Type here...')

    submitted = st.form_submit_button('Submit')

    if submitted:
        chain = setup_chain()
        st.write_stream(chain.stream(text))
