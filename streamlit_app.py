import streamlit as st
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

# Initialize the language model and prompt template
llm = ChatOpenAI(model='gpt-4', max_tokens=100)
str_parser = StrOutputParser()
template = (
    "Please answer the questions based on the following content and your own judgment:\n"
    "{context}\n"
    "Question: {question}"
)
prompt = ChatPromptTemplate.from_template(template)

# Streamlit App
st.title("LangChain LLM Q&A")

# Initialize FAISS retriever
pdf_retriever = None
try:
    # Load pre-indexed FAISS database and metadata with dangerous deserialization enabled
    db_pdf = FAISS.load_local("Database/PDF", OpenAIEmbeddings(), allow_dangerous_deserialization=True)
    pdf_retriever = db_pdf.as_retriever()
    st.write("Loaded pre-indexed FAISS data successfully.")
except Exception as e:
    st.write("Error loading FAISS index:", e)

# Function to process the question and get the answer
def get_answer():
    question = st.session_state.question_input
    if question and pdf_retriever:
        # Retrieve context relevant to the question
        retrieved_docs = pdf_retriever.get_relevant_documents(question)
        context_texts = "\n".join([doc.page_content for doc in retrieved_docs])

        # Format and retrieve the answer from the LLM
        inputs = {"context": context_texts, "question": question}
        answer = llm(prompt.format(**inputs))

        # Display the answer
        st.session_state.answer_output = answer.content
    else:
        st.session_state.answer_output = "Please enter a question."

# User input for the question
st.text_input(
    "Ask me anything:",
    key="question_input",
    on_change=get_answer
)

# Display the answer if available
if "answer_output" in st.session_state:
    st.write("Answer:", st.session_state.answer_output)
