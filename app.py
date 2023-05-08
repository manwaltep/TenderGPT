from dotenv import load_dotenv
import openai as OpenAI
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
    # Load environment variables
    load_dotenv()

    # Set up Streamlit page
    st.set_page_config(page_title="Ask your PDF")
    st.header("ChatPDF")

    # Get user's API key
    user_secret = st.text_input(
        "Enter your API key", placeholder="Paste your API key here sk-", type="password")
    if user_secret:
        ChatOpenAI.api_key = user_secret

    # Allow user to select a model
    model = st.selectbox("Select a model", ("gpt-3.5-turbo", "gpt-4"))

    # Get PDF file from user
    pdf = st.file_uploader(
        "Upload your PDF", accept_multiple_files=False, type="pdf")

    if pdf is not None:
        # Read text from PDF
        with pdf.open("rb") as f:
            pdf_reader = PdfReader(f)
            text = "".join(page.extract_text() for page in pdf_reader.pages)

        # Initialize tokenizer
        tokenizer = tiktoken.get_encoding("cl100k_base")

        # Define a function for counting tokens in text
        def tiktoken_len(text):
            tokens = tokenizer.encode(text, disallowed_special=())
            return len(tokens)

        # Set up a splitter to split the text into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500, chunk_overlap=100, length_function=tiktoken_len, separators=["\n\n", "\n", " ", ""])
        chunks = text_splitter.split_text(text)

        # Generate embeddings for text chunks
        embeddings = OpenAIEmbeddings()
        knowledge_base = FAISS.from_texts(chunks, embeddings)

        # Get user's question
        user_question = st.text_input(label="Ask a question about your PDF:")

        # If a question is entered, and the user clicks 'Enter', answer the question
        if user_question and st.button(label="Enter", type='primary'):
            docs = knowledge_base.similarity_search(user_question)
            llm = ChatOpenAI(model_name=(model))
            chain = load_qa_chain(llm, chain_type="stuff")

            # Get the response from the chat model
            with get_openai_callback() as cb:
                response = chain.run(input_documents=docs,
                                     question=user_question)

            # Display the response
            st.write(response)


if __name__ == '__main__':
    main()
