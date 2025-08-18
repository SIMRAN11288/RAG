import streamlit as st
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
from langchain.text_splitter import NLTKTextSplitter,RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
import requests
import urllib.parse
import os
import nltk
nltk.download("punkt")
nltk.download("punkt_tab")  # for newer NLTK versions
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
st.title('WHAT DO YOU WANT TO USE- DICTIONARY| CALCULATOR |student mentor- Ask FAQs on maths of calss 10 and 12.Also on enhancement of english vocabulary')
def dictionary(word):
    url=f"https://api.dictionaryapi.dev/api/v2/entries/en/{word}"
    req=requests.get(url)
    if req.status_code==200:
        data=req.json()
        return data[0]['meanings'][0]['definitions'][0]['definition']
    return None

def cal(problem):
    problem=urllib.parse.quote(problem,safe='')
    url=f"https://api.mathjs.org/v4/?expr={problem}"
    res=requests.get(url)
    if res.status_code==200:
        return res.text
    else:
        return f'ERROR:{res.status_code}'
        
def product_info(query):
    loader=PyPDFLoader('student_mentor.pdf')
    document=loader.load()
    embedding_model=GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    #document returned a list of document objects (page_content + meta_data)
    
    #This gives you a list of strings, one per page.
    
    #1st splitter 
    sentence_splitter = NLTKTextSplitter() #Here you loop over each page and split it into sentences
    documents=sentence_splitter.split_documents(document)
    #2nd splitter 
    sentence_splitter2=RecursiveCharacterTextSplitter(chunk_size=300,chunk_overlap=0)
    #split_text() expects a single string, not a list, so if you pass sentences (which is a list), it won’t work correctly.
    docs=sentence_splitter2.split_documents(documents)
    
    #vector store
    
    vector_store=FAISS.from_documents(docs,embedding=embedding_model)
    retriever=vector_store.as_retriever (search_type='similarity',search_kwargs={'k':1})
    result=retriever.invoke(query)
    for output in result:
        return output.page_content
    # for output in result:
    #     return output.page_content
    
box=st.selectbox("you want to perform:",['Dictionary','Calculator','Student Mentor'])

if box=='Dictionary':
    word=st.text_input("ENTER A WORD TO GET ITS MEANING")
    if st.button('Get Meaning'):
        if word:
            meaning= dictionary(word)
            if meaning:
                st.success(f"{word} Meaning: {meaning}")
            else:
                st.error("Meaning not found")
        else:
            st.warning("enter a word")
                
elif box=='Calculator':
    problem=st.text_input('Enter your MATHEMATICAL PROBLEM')
    if st.button('SOLVE'):
        if problem:
            solution=cal(problem)
            st.success(f"Calculation DONE:{solution}")
        else:
            st.warning("please enter an expression")
     
elif box=='Student Mentor':
    query=st.text_input("FAQs")
    if st.button('Fetch Information'):
        if query:
            result=product_info(query)
            st.success(f"Info fetched: {result}")
        else:
            st.warning("NO INPUT FOUND")
else:

    st.write("Not able to process your query")
















