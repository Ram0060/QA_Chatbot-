import streamlit as st
import openai
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

import os
from dotenv import load_dotenv
load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "Simple Q&A app with OPENAI"

# Define the prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI bot. Please respond to the following queries."),
    ("human", "{Question}"),
])

# Define response generator function
def generate_response(question, model_name, temperature, max_tokens):
    llm = ChatOpenAI(
        model_name=model_name,
        temperature=temperature,
        max_tokens=max_tokens,
        openai_api_key=os.environ["OPENAI_API_KEY"]
    )
    output_parser = StrOutputParser()
    chain = prompt | llm | output_parser
    return chain.invoke({"Question": question})

# Streamlit App UI
st.title("Simple Q&A App with OpenAI")

st.sidebar.title("Settings")
model_name = st.sidebar.selectbox("Select OpenAI Model", ["gpt-4", "gpt-4o", "gpt-4-turbo"])
temperature = st.sidebar.slider("Temperature", min_value=0.0, max_value=1.0, value=0.7)
max_tokens = st.sidebar.slider("Max Tokens", min_value=50, max_value=300, value=150)

st.write("Ask a question:")
user_input = st.text_input("You:")

if user_input:
    response = generate_response(user_input, model_name, temperature, max_tokens)
    st.write("AI:", response)
else:
    st.write("Please enter a question to get a response from the AI.")
