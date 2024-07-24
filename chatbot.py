import streamlit as st
import os

# importing necessary libraries
import requests
from bs4 import BeautifulSoup
#from urllib.parse import urljoin, urlparse
from langchain_huggingface import HuggingFaceEndpoint, HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.schema import Document
from huggingface_hub import login

#connection to huggingface
huggingface_token = st.secrets["df_token"]
login(token=huggingface_token)

# This info is at the top of each HuggingFace model page
hf_model = "mistralai/Mistral-7B-Instruct-v0.3"
llm = HuggingFaceEndpoint(repo_id = hf_model)

# Initialize HuggingFace embeddings
embedding_model = "sentence-transformers/all-MiniLM-l6-v2"
embeddings_folder = "https://github.com/Markus-DS-29/honeybot/blob/main/content/"
embeddings = HuggingFaceEmbeddings(model_name=embedding_model,
                                   cache_folder=embeddings_folder)


# Read FAISS vector store from github and store it in streamlit

# Create the directory if it doesn't exist
if not os.path.exists("./content"):
    os.makedirs("./content")

# Define the URL to the FAISS index file on GitHub
faiss_url = "https://github.com/Markus-DS-29/honeybot/raw/main/content/faiss_index"

# Define the local path to save the FAISS index file
faiss_local_path = "./content/faiss_index"

# Download the FAISS index file from GitHub
if not os.path.exists(faiss_local_path):
    response = requests.get(faiss_url)
    response.raise_for_status()
    with open(faiss_local_path, 'wb') as f:
        f.write(response.content)

# Use local streamlit path
faiss_path = "./content/faiss_index/"
vector_all_html_url_db = FAISS.load_local(faiss_path, embeddings, allow_dangerous_deserialization=True)
retriever = vector_all_html_url_db.as_retriever(search_kwargs={"k": 2})

###

@st.cache_resource
def init_memory(_llm):
    return ConversationBufferMemory(
        llm=llm,
        output_key='answer',
        memory_key='chat_history',
        return_messages=True)
memory = init_memory(llm)

input_template = """
#Context#
Your are answering for an honey expert who is specialised in honey from local beekeepers.
#Objective#
1. Most important: Check if the user provides information about personal preferences in regards of consistancy (cremy or liquid) and honey-colour (light or dark). 
2. Most important: If information about consistancy and honey-colour is not provided: Ask for cosistancy and honey-colour!
You want to provide a high quality answer to an honey loving customer. A high quality answer must contain information about colour: light or dark, consistancy: cremy or liquid. 
#Style#
Please use a gentle, short style.
#Tone#
Your tone should be persuasive and light.
#Audience#
Honey-Lovers between 30 and 40 years old.
#Response#
1. Most important: Check if the user provides information about personal preferences in regards of consistancy (cremy or liquid) and honey-colour (light or dark). 
2. If information about consistancy and honey-colour is not provided: Ask for cosistancy and honey-colour!
Most important: If the question doesnot contain consistancy and honey-colour: Ask for cosistancy and honey-colour!

Answer the question based only on the following context.
Keep your answers short and succinct, but always use whole sentences.
Most Important: Always add an according link from the domain https://heimathonig.de/honig to your answer and make sure it is a real link starting with https! 
Make sure to give importance to location if mentionend in the question.
All answers in German.

Previous conversation:
{chat_history}

Context to answer question:
{context}

Question to be answered: {question}

Response:"""

prompt = PromptTemplate(template=input_template,
                        input_variables=["context", "question"])        

#conversation
chain = ConversationalRetrievalChain.from_llm(llm,
                                                retriever = retriever,
                                                memory = memory,
                                                return_source_documents = False,
                                                combine_docs_chain_kwargs = {"prompt": prompt})


# Start the conversation

# Title
st.title("Welcome to the Boney, the Honey Bot")
# Markdown
st.markdown("""
Just give me a minute, I will be right with you.
""")


##### streamlit #####

# Initialise chat history
# Chat history saves the previous messages to be displayed
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("Welche Art Honig magst du am liebsten?"):

    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)

    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Begin spinner before answering question so it's there for the duration
    with st.spinner("Suche im Bienenstock nach einer Antwort..."):

        # send question to chain to get answer
        answer = chain(prompt)

        # extract answer from dictionary returned by chain
        response = answer["answer"]

        # Display chatbot response in chat message container
        with st.chat_message("assistant"):
            st.markdown(answer["answer"])

        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})
