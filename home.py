import streamlit as st
import time  # For loading animation

# Import your RAG system components
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import HuggingFaceHub
import os


# Streamlit Web App
st.set_page_config(page_title="مساعد موني مون", page_icon="💬", layout="centered")

# Header
st.title("💬 مساعد موني مون")
st.write("أهلا وسهلا بك في مساعد **موني مون**! اطرح سؤالك حول خدماتنا وستحصل على الإجابة مباشرة. 😊")
huggingface_api = st.text_input("api ادخل", placeholder="اكتب هنا...")
apiButton= st.button("اضغط لتحميل المودل")
# Input Area
question = st.text_input("📝 اسألني سؤال", placeholder="اكتب سؤالك هنا...")
model='mistralai/Mistral-7B-Instruct-v0.3'

if apiButton:
    @st.cache_resource
    def load_llm():
        return HuggingFaceHub(
            repo_id=model,
            model_kwargs={"temperature": 0.1, 'stream':True},
            huggingfacehub_api_token=huggingface_api,
        )

    llm = load_llm()
    model_name= 'Linq-AI-Research/Linq-Embed-Mistral'
    # Load embeddings using HuggingFaceEmbeddings
    @st.cache_resource
    def load_embeddings():
        return HuggingFaceEmbeddings(model_name='Linq-AI-Research/Linq-Embed-Mistral')

    embeddings = load_embeddings()

    # Cache ChromaDB for improved performance
    @st.cache_resource
    def load_chroma():
        return Chroma(
            embedding_function=embeddings,
            persist_directory='./chroma_db'  # Load from disk
        )

    vectordb = load_chroma()
    # Create a retriever for querying the vector database
    retriever = vectordb.as_retriever(score_threshold= 1)


    system_prompt = """
    Given the following context, generate an answer based on this context only. Don't try to make up an answer.
    In the reply, try to provide as much text as possible from the "Answer" section in the source document context without making many changes.
    If the answer is not found in the context, state "ما عندنا معلومات كافية بحيث نقدر نخدمك. نعتذر."
    CONTEXT: {context}
    """

    # Create the prompt template
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "السؤال: {input}\n\nالإجابة:")  # Improves formatting
        ]
    )

    # Create the document combination chain
    question_answer_chain = create_stuff_documents_chain(llm, prompt)

    # Create the retrieval chain
    chain = create_retrieval_chain(retriever, question_answer_chain)
    def get_answer(question):
        """
        Queries the retrieval chain with a given question and extracts the clean answer.

        Args:
            question (str): The user's question.

        Returns:
            None (prints the answer directly).
        """
        try:
            # Invoke the retrieval chain with the question
            response = chain.invoke({"input": question})
            
            # Check if the response contains the expected output, assuming it's a dictionary
            if 'answer' in response:
                # Remove unnecessary context
                clean_answer = response["answer"]
                if "Human: السؤال:" in clean_answer:
                    clean_answer = clean_answer.split("Human: السؤال:")[-1].strip()
                if "الإجابة:" in clean_answer:
                    clean_answer = clean_answer.split("الإجابة:")[-1].strip()
                
                if "معلومات" in clean_answer:
                    clean_answer = "ما عندنا معلومات كافية بحيث نقدر نخدمك. نعتذر."
                
                
    #Format with the template
                template = f"""
            \n  هلا وسهلا عميلنا العزيز 
            \n  معاك مساعدك الشخصي في موني مون 
            \n {clean_answer}\n
                \nاتمنى ان تكون مشكلتك انحلت 😊\n
                """
                
                return template.strip()  # Print the final response 
            else:
                return "No valid output received from the retrieval chain."
        
        except Exception as e:
            print(f"An error occurred: {str(e)}")


    # Answer Area
    if st.button("🔍 اسأل"):
        if question.strip() != "":
            # Show loading animation
            with st.spinner("⏳ جاري البحث عن الإجابة..."):
                answer = get_answer(question)
            # Show Response
            if "ما عندنا معلومات كافية" in answer:
                st.warning(answer)
            else:
                st.success(answer)
        else:
            st.error("❗️يرجى كتابة السؤال قبل الضغط على زر 'اسأل'")
