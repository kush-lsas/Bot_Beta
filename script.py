import os
import streamlit as st
from langchain_openai.chat_models import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import ChatPromptTemplate
from langchain.embeddings.openai import OpenAIEmbeddings

from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_pinecone import PineconeVectorStore

from agi.stk12.stkdesktop import STKDesktop
from agi.stk12.stkobjects import *
from agi.stk12.stkutil import *
from agi.stk12.vgt import *

st.set_page_config(page_title="Interweb Explorer", page_icon="🌐")

OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]

def generate_embeddings():
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    index_name = "uma-bot"
    pcstore = PineconeVectorStore.from_existing_index(index_name, embeddings)
    return pcstore.as_retriever()

def setup_chain():
    model = ChatOpenAI(temperature=0 , openai_api_key=OPENAI_API_KEY, model="gpt-4-turbo-preview", streaming=True)
    parser = StrOutputParser()
    temp = """
            You are an AI assistant specializing in translating natural language queries related into executable Connect commands for the Systems Tool Kit (STK) software. Your purpose is to help users interact with STK using natural language.

            You will be provided with the most suitable list of Connect commands and their usage as context. Use this information to generate accurate and relevant Connect scripts based on the user's input.

            When a user asks a question or makes a request, follow these steps:
            1. Analyze the question to identify the key objectives and parameters.
            2. Determine the appropriate STK objects, commands, and settings required to fulfill the user's request, using the information from context.
            3. Generate the corresponding Connect script using the correct syntax and structure, referring to context for accurate usage and formatting.
            4. Provide the Connect script as a code block in your response, along with a brief explanation of what the script does and how it utilizes the specific commands from context.

            Remember to:
            - Leverage the provided context to ensure accuracy and relevance in your responses.
            - Be concise and precise in your explanations, focusing on how the selected commands address the user's specific request.
            - Use clear variable names and adhere to STK's naming conventions.
            - Handle errors and edge cases gracefully, providing helpful feedback to the user.
            - Ask for clarification if the user's request is ambiguous or incomplete, or if no suitable commands are found in context.
            - Ensure the generated Connect scripts are syntactically correct and ready to execute.

            Your responses should focus on providing the Connect script, necessary explanations, and references to the specific commands used from context. Avoid engaging in extended conversations unrelated to the user's STK-related query.

            Context: {context}

            Question: {question}
            """
    
    template = """
            You are an AI assistant specializing in translating natural language queries related into executable Connect commands for the Systems Tool Kit (STK) software. Your purpose is to help users interact with STK using natural language.

            You will be provided with the most suitable list of Connect commands and their usage as context. Use this information to generate accurate and relevant Connect scripts based on the user's input.

            When a user asks a question or makes a request, follow these steps:
            1. Analyze the question to identify the key objectives and parameters.
            2. Determine the appropriate STK objects, commands, and settings required to fulfill the user's request, using the information from context.
            3. Generate the corresponding Connect script using the correct syntax and structure, referring to context for accurate usage and formatting.
            4. Provide the Connect script as a code block in your response.

            Remember to:
            - Leverage the provided context to ensure accuracy and relevance in your responses.
            - Be concise and precise in your explanations, focusing on how the selected commands address the user's specific request.
            - Use clear variable names and adhere to STK's naming conventions.
            - Handle errors and edge cases gracefully, providing helpful feedback to the user.
            - Ask for clarification if the user's request is ambiguous or incomplete, or if no suitable commands are found in context.
            - Ensure the generated Connect scripts are syntactically correct and ready to execute.

            Your responses should only contain the Connect script, without any explanations or extended conversations unrelated to the user's STK-related query.

            Context: {context}

            Question: {question}
            """
    
    
    
    prompt = ChatPromptTemplate.from_template(template)
    retriever = generate_embeddings()
    setup = RunnableParallel(context=retriever, question=RunnablePassthrough())
    chain = setup | prompt | model | parser
    return chain

def GenerateNewCommand(incorrect_command):
    model = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model="gpt-4-turbo-preview")
    parser = StrOutputParser()

    template = """
                        You are an AI assistant specializing in debugging and correcting Connect commands for the Systems Tool Kit (STK) software. Your purpose is to help users fix and optimize their Connect command by providing corrected commands that address the issues in the original, incorrect command.

                        You will be provided with a comprehensive list of Connect commands and their usage as context. Use this information to identify the problems in the given incorrect command and generate a corrected version of the command.

                        When a user provides an incorrect Connect command, follow these steps:
                        1. Analyze the provided command to identify the errors, such as incorrect command, incorrect syntax, or improper usage of command.
                        2. Consult the context to determine the appropriate corrections needed to fix the identified errors.
                        3. Generate a corrected version of the Connect command that addresses the issues found in the original command.
                        4. Provide the corrected Connect command as a code block in your response.

                        Remember to:
                        - Leverage the provided context to ensure accuracy and adherence to STK's conventions when correcting commands.
                        - Be specific and concise in your explanations, focusing on the errors found and the corrections made.
                        - Use clear variable names and adhere to STK's naming conventions in the corrected command.
                        - Handle cases where the provided command is too incomplete or ambiguous to correct, providing a helpful error message to the user.
                        - Ensure that the corrected Connect command is syntactically correct and ready to execute.

                        Your responses should only contain the Connect script, without any explanations or extended conversations unrelated to the user's STK-related query.

                        Context: {context}

                        Incorrect Command: {incorrect_command}
                        """
    
    prompt = ChatPromptTemplate.from_template(template)
    retriever = generate_embeddings()
    setup = RunnableParallel(context=retriever, incorrect_command=RunnablePassthrough())
    chain = setup | prompt | model | parser

    return chain.invoke(incorrect_command)

def ExecuteSTKScript(script, incorrect_count=0):
    if incorrect_count == 10:
        return
    stk = STKDesktop.AttachToApplication()
    root = stk.Root
    commands = script.split("\n")
    for command in commands:
        output = st.empty()
        if command.startswith("```"):
            continue
        try:
            output.write(f":blue[Executing command:] ```{command}```")
            result = root.ExecuteCommand(command)
            if result.IsSucceeded:
                output.write(f":green[Command:] ```{command}``` :green[executed successfully.]")
        except Exception as e:
            incorrect_count += 1
            output.write(f":red[Command:] ```{command}``` :red[failed to execute.]")
            new_command = GenerateNewCommand(command)
            ExecuteSTKScript(new_command, incorrect_count)

st.image("logo.png")
st.header("`UMA Bot (Connect Commands) Beta`")
st.info("`I am an AI that can answer questions based on Connect commands that are used in STK. I can help you with the usage of Connect commands for different use-cases. Feel free to ask me anything!`")

with st.form('my_form'):
    text = st.text_area('Enter question:', 'Type here...')

    submitted = st.form_submit_button('Submit')

    if submitted:
        chain = setup_chain()
        output = st.write_stream(chain.stream(text))
        ExecuteSTKScript(output)