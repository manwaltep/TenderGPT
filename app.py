from dotenv import load_dotenv
import openai as OpenAI  # Add this import
import tiktoken
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import ChatOpenAI
from langchain.callbacks import get_openai_callback


def main():
    load_dotenv()
    st.set_page_config(page_title="Ask your PDF")
    st.header("TenderGPT")

    # Input your API key
    user_secret = st.text_input(
        label=":blue[Enter your API key]", placeholder="Paste your API key here sk-", type="password")

    if user_secret:
        ChatOpenAI.api_key = user_secret

    # select your model
    model = st.selectbox("Select a model", ("gpt-3.5-turbo", "gpt-4"))

    # upload file
    pdf = st.file_uploader(
        "Upload your PDF", accept_multiple_files=False, type="pdf")

    # extract the text
    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()

        # count token chunks
        tokenizer = tiktoken.get_encoding("cl100k_base")

        def tiktoken_len(text):
            tokens = tokenizer.encode(
                text,
                disallowed_special=()
            )
            return len(tokens)

        # split into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100,
            length_function=tiktoken_len,
            separators=["\n\n", "\n", " ", ""]
        )
        chunks = text_splitter.split_text(text)

        # create embeddings
        embeddings = OpenAIEmbeddings()
        knowledge_base = FAISS.from_texts(chunks, embeddings)

        # show user input
        user_question = st.text_input(label="Ask a question about your PDF:", )
        if user_question:
            docs = knowledge_base.similarity_search(user_question)
            llm = ChatOpenAI(model_name=(model))
            chain = load_qa_chain(llm, chain_type="stuff")
            with get_openai_callback() as cb:
                response = chain.run(input_documents=docs,
                                     question=user_question)

            # if the user has entered a question, show button
            if st.button(label="Enter", type='primary'):
                with get_openai_callback() as cb:
                    response = chain.run(input_documents=docs,
                                         question=user_question)

            # show the response
            st.write(response, cb)


if __name__ == '__main__':
    main()
