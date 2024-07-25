from langchain_huggingface import HuggingFaceEndpoint, HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
import streamlit as st


# llm
hf_model = "mistralai/Mistral-7B-Instruct-v0.3"
llm = HuggingFaceEndpoint(repo_id=hf_model)

# embeddings
embedding_model = "sentence-transformers/all-MiniLM-l6-v2"
embeddings_folder = "/content/"

embeddings = HuggingFaceEmbeddings(model_name=embedding_model,
                                   cache_folder=embeddings_folder)

# load Vector Database
# allow_dangerous_deserialization is needed. Pickle files can be modified to deliver a malicious payload that results in execution of arbitrary code on your machine
vector_db = FAISS.load_local("/content/faiss_index", embeddings, allow_dangerous_deserialization=True)

# retriever
retriever = vector_db.as_retriever(search_kwargs={"k": 2})

# memory
@st.cache_resource
def init_memory(_llm):
    return ConversationBufferMemory(
        llm=llm,
        output_key='answer',
        memory_key='chat_history',
        return_messages=True)
memory = init_memory(llm)

# prompt
template = """You are a nice chatbot having a conversation with a human. Answer the question based only on the following context and previous conversation. Keep your answers short and succinct.And always say the source of the information.

Previous conversation:
{chat_history}

Context to answer question:
{context}

New human question: {question}
Response:"""

prompt = PromptTemplate(template=template,
                        input_variables=["context", "question"])

# chain
chain = ConversationalRetrievalChain.from_llm(llm,
                                              retriever=retriever,
                                              memory=memory,
                                              return_source_documents=True,
                                              combine_docs_chain_kwargs={"prompt": prompt})


##### streamlit #####

st.title("Asian Vaccine")

# Markdown
st.markdown("""
Ana y Nuria y sus Vaccunas, jeje...

Streamlit is a powerful Python library for creating web apps. It is easy to use and has a wide range of features, including:

* **Pensamos q mas poner:** Referencias de donde sacamos las cosas?.

""")



# Map
map_data = [
 {'name': 'India', 'lat': 20.5937, 'lon': 78.9629},
 {'name': 'Viet Nam', 'lat': 14.0583, 'lon': 108.2772},
 {'name': 'China', 'lat': 35.8617, 'lon': 104.1954},
 {'name': 'Papua New Guinea', 'lat': -6.314993, 'lon': 143.95555},
 {'name': 'Myanmar', 'lat': 21.9162, 'lon': 95.956},
 {'name': 'Malaysia', 'lat': 4.2105, 'lon': 101.9758},
 {'name': 'Indonesia', 'lat': -0.7893, 'lon': 113.9213},
 {'name': 'Bangladesh', 'lat': 23.685, 'lon': 90.3563},
 {'name': 'Philippines', 'lat': 12.8797, 'lon': 121.774},
 {'name': 'Sri Lanka', 'lat': 7.8731, 'lon': 80.7718},
 {'name': "Lao People's Democratic Republic", 'lat': 19.8563, 'lon': 102.4955},
 {'name': 'Nepal', 'lat': 28.3949, 'lon': 84.124},
 {'name': 'Mongolia', 'lat': 46.8625, 'lon': 103.8467},
 {'name': 'Thailand', 'lat': 15.87, 'lon': 100.9925},
 {'name': "Democratic People's Republic of Korea",
  'lat': 40.3399,
  'lon': 127.5101},
 {'name': 'Republic of Korea', 'lat': 35.9078, 'lon': 127.7669},
 {'name': 'Vanuatu', 'lat': -15.3767, 'lon': 166.9592},
 {'name': 'Palau', 'lat': 7.5149, 'lon': 134.5825},
 {'name': 'Kiribati', 'lat': -3.3704, 'lon': -168.734},
 {'name': 'Cambodia', 'lat': 12.5657, 'lon': 104.991},
 {'name': 'Guam', 'lat': 13.4443, 'lon': 144.7937},
 {'name': 'Tonga', 'lat': -21.1789, 'lon': -175.1982},
 {'name': 'Bhutan', 'lat': 27.5142, 'lon': 90.4336},
 {'name': 'Maldives', 'lat': 3.2028, 'lon': 73.2207},
 {'name': 'Cook Islands', 'lat': -21.2367, 'lon': -159.7777},
 {'name': 'Japan', 'lat': 36.2048, 'lon': 138.2529},
 {'name': 'Samoa', 'lat': -13.759, 'lon': -172.1046},
 {'name': 'Solomon Islands', 'lat': -9.6457, 'lon': 160.1562},
 {'name': 'Brunei Darussalam', 'lat': 4.5353, 'lon': 114.7277},
 {'name': 'Singapore', 'lat': 1.3521, 'lon': 103.8198},
 {'name': 'Niue', 'lat': -19.0544, 'lon': -169.8672},
 {'name': 'Marshall Islands', 'lat': 7.1315, 'lon': 171.1845},
 {'name': 'New Caledonia', 'lat': -20.9043, 'lon': 165.618},
 {'name': 'Fiji', 'lat': -17.7134, 'lon': 178.065},
 {'name': 'Tuvalu', 'lat': -7.1095, 'lon': 179.194},
 {'name': 'Tokelau', 'lat': -9.2003, 'lon': -171.8484},
 {'name': 'Australia', 'lat': -25.2744, 'lon': 133.7751},
 {'name': 'French Polynesia', 'lat': -17.6797, 'lon': -149.4068},
 {'name': 'American Samoa', 'lat': -14.27, 'lon': -170.1322},
 {'name': 'Federated States of Micronesia', 'lat': 7.4256, 'lon': 150.5508},
 {'name': 'New Zealand', 'lat': -40.9006, 'lon': 174.886},
 {'name': 'Nauru', 'lat': -0.5228, 'lon': 166.9315},
 {'name': 'Wallis and Futuna', 'lat': -13.7681, 'lon': -177.1561},
 {'name': 'Northern Mariana Islands', 'lat': 15.0979, 'lon': 145.6739},
 {'name': 'Timor-Leste', 'lat': -8.8742, 'lon': 125.7275},

 ]

st.map(map_data)

# Initialise chat history
# Chat history saves the previous messages to be displayed
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("Mad Scientists Wanted!"):

    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)

    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Begin spinner before answering question so it's there for the duration
    with st.spinner("Going down the protocols for answers..."):

        # send question to chain to get answer
        answer = chain(prompt)

        # extract answer from dictionary returned by chain
        response = answer["answer"]

        # Display chatbot response in chat message container
        with st.chat_message("assistant"):
            st.markdown(answer["answer"])

        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})
