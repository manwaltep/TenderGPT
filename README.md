# TenderGPT

# TenderGPT README

## Overview

TenderGPT is a user-friendly tool that allows users to ask questions about a PDF document and get relevant answers using OpenAI's GPT models. It uses OpenAI's API and a number of libraries, including Streamlit, to build an interactive web application.

## How It Works

1. Users enter their OpenAI API key to authenticate.
2. Users select a GPT model (either gpt-3.5-turbo or gpt-4) to use for the question-answering process.
3. Users upload a PDF document.
4. The application extracts text from the PDF document.
5. The extracted text is split into smaller chunks and tokenized.
6. These chunks are transformed into embeddings using OpenAIEmbeddings.
7. The embeddings are stored in a knowledge base using FAISS (a library for similarity search).
8. Users can then ask a question about the PDF document.
9. The application finds the most relevant chunks of text from the knowledge base and uses the selected GPT model to generate a response to the user's question.

## Dependencies

- dotenv
- openai
- tiktoken
- streamlit
- PyPDF2
- langchain (a custom package)
  - text_splitter
  - embeddings
  - vectorstores
  - chains.question_answering
  - chat_models
  - callbacks

## Usage

To use TenderGPT, make sure you have all the required dependencies installed, and simply run the main script:

```bash
python app.py
```

This will launch the Streamlit web application, and you can interact with it through your web browser.
