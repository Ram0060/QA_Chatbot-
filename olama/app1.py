from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import OllamaLLM  # ✅ Updated import
import streamlit as st
import os
from dotenv import load_dotenv

load_dotenv()

# Optional LangChain env variables
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "Simple Q&A app with Ollama"

# Prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI bot. Please respond to the following queries."),
    ("human", "{Question}"),
])

# Function to generate response
def generate_response(question, model_name, temperature):
    llm = OllamaLLM(  # ✅ Updated class name
        model=model_name,
        temperature=temperature,
        # ❌ max_tokens removed
    )
    output_parser = StrOutputParser()
    chain = prompt | llm | output_parser
    return chain.invoke({"Question": question})

# Streamlit UI
st.title("Simple Q&A App with Ollama")

st.sidebar.title("Settings")
model_name = st.sidebar.selectbox("Select Ollama Model", ["llama3:latest", "llama3.2:1b", "gemma3:1b"])  # ✅ Adjust per your ollama list
temperature = st.sidebar.slider("Temperature", min_value=0.0, max_value=1.0, value=0.7)

st.write("Ask a question:")
user_input = st.text_input("You:")

if user_input:
    response = generate_response(user_input, model_name, temperature)
    st.write("AI:", response)
else:
    st.write("Please enter a question to get a response from the AI.")
