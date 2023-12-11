import streamlit as st
from dotenv import load_dotenv
import os
from pypdf import PdfReader
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings, HuggingFaceInferenceAPIEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.llms import HuggingFaceHub
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders import PyPDFLoader
from html_template import css, bot_template, user_template
from pprint import pprint


def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    
    return text


def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=100, length_function=len)
    chunks = text_splitter.split_text(text)
    return chunks


def get_pdf_chunks(pdf_docs):
    chunks = []
    chunk_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    for pdf in pdf_docs:
        print("pdf:", pdf)
        documents = PyPDFLoader(pdf)
        chunks += chunk_splitter.split_documents(documents)
    print("chunks:\n")
    pprint(chunks)
    return chunks


def get_vectorstore(chunks):
    # embeddings = OpenAIEmbeddings()
    # embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    embeddings = HuggingFaceInferenceAPIEmbeddings(model_name=model_name, api_key=os.getenv("HUGGINGFACEHUB_API_TOKEN"))
    print("Got embeddings from the API!", embeddings)
    vectorstore = FAISS.from_texts(texts=chunks, embedding=embeddings)
    return vectorstore


def get_conversation_chain(vectorstore):
    # llm = ChatOpenAI()
    repo_id = "tiiuae/falcon-7b-instruct"
    llm = HuggingFaceHub(
            repo_id=repo_id,
            model_kwargs={"temperature": 0.1}
        )
    memory = ConversationBufferMemory(memory_key="chat_history", input_key='question', output_key='answer', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory,
        return_source_documents=True,
        verbose=True,
    )
    return conversation_chain


def handle_userinput(user_question):
    response = st.session_state.conversation({
        'question': user_question,
    })
    print("response:\n")
    pprint(response)
    st.session_state.chat_history = response['chat_history']
    
    # Write raw response
    with st.expander("Raw response from function"):
        st.write(response)

    # Write to chat
    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
    


def main():
    load_dotenv()

    st.set_page_config(page_title="Chat with multiple PDFs", page_icon=":books:")
    st.write(css, unsafe_allow_html=True)
    
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
    
    if "raw_response" not in st.session_state:
        st.session_state.raw_response = st.expander("Raw response")

    st.header("Chat with multiple PDFs :books:")
    user_question = st.text_input("Ask a question about your documents:")
    if user_question:
        handle_userinput(user_question)

    # st.write(user_template.replace("{{MSG}}", "Hello chatbot!"), unsafe_allow_html=True)
    # st.write(bot_template.replace("{{MSG}}", "Hello human!"), unsafe_allow_html=True)

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader("Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):
                # Chunk pdf so we get page numbers too
                # chunks = get_pdf_chunks(pdf_docs)

                # Get the pdf text
                raw_text = get_pdf_text(pdf_docs)
                # st.write(raw_text)

                # Get the text chunks
                text_chunks = get_text_chunks(raw_text)
                # st.write(text_chunks)

                # Create vector store
                vectorstore = get_vectorstore(text_chunks)

                # Create conversation chain
                st.session_state.conversation = get_conversation_chain(vectorstore)

if __name__ == "__main__":
    main()
