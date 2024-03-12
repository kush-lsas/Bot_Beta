import os
import streamlit as st
from langchain_openai.chat_models import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import ChatPromptTemplate
from langchain_community.document_transformers import BeautifulSoupTransformer
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_community.document_transformers import BeautifulSoupTransformer
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
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
        doc_transformed = bs_transformer.transform_documents(doc, tags_to_extract=["p","title", "h1", "h2", "h3", "th", "td"])
        splitter = RecursiveCharacterTextSplitter(chunk_size = 2000, chunk_overlap = 20)
        split_docs = splitter.split_documents(doc_transformed)
        for splits in split_docs:
            html_files_split.append(splits)
          
    return html_files_split

def generate_embeddings():
    #html_files_split = extract_and_split_html()
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    index_name = "uma-bot"
    pcstore = PineconeVectorStore.from_existing_index(index_name, embeddings)
    #pcstore = PineconeVectorStore.from_documents(html_files_split, embeddings, index_name=index_name)
    return pcstore.as_retriever()

def setup_chain():
    model = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model="gpt-4-turbo-preview")
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

#question = st.text_input("`Ask a question:`")

    # Write answer and sources
    #retrieval_streamer_cb = PrintRetrievalHandler(st.container())
with st.form('my_form'):
    text = st.text_area('Enter question:', 'Type here...')
    submitted = st.form_submit_button('Submit')
    if submitted:
        chain = setup_chain()
        st.info(chain.invoke(text))