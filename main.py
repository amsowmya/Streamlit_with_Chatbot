from langchain_openai.chat_models import AzureChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferWindowMemory
import streamlit as st
from streamlit_chat import message
import os
from dotenv import load_dotenv

load_dotenv()

AZURE_OPENAI_KEY = os.getenv("AZURE_OPENAI_API_KEY")

st.title("Smartbot")
query = st.text_input("Query: ", key="input")

if 'responses' not in st.session_state:
    st.session_state['responses'] = []

if 'requests' not in st.session_state:
    st.session_state['requests'] = []

llm = AzureChatOpenAI(
    api_version="2024-02-15-preview",
    azure_deployment="atttestgpt35turbo",
    azure_endpoint="https://cb-att-openai-instance.openai.azure.com/",
    api_key=AZURE_OPENAI_KEY
)

if 'buffer_memory' not in st.session_state:
    st.session_state.buffer_memory = ConversationBufferWindowMemory(k=3)

conversation = ConversationChain(
    llm=llm,
    memory=st.session_state.buffer_memory,
    verbose=True
)

if query:
    response = conversation.run(query)
    st.session_state.requests.append(query)
    st.session_state.responses.append(response)

if st.session_state['responses']:
    for i in range(len(st.session_state['responses'])-1, -1, -1):
        message(st.session_state['requests'][i], is_user=True, key=str(i) + '_user')
        message(st.session_state['responses'][i], key=str(i))