import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_text_splitters import TokenTextSplitter
from langchain.docstore.document import Document
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain import embeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from src.prompt import *
from langchain.chains import RetrievalQA


load_dotenv()
GEMINI_KEY = os.getenv("GOOGLE_API_KEY")
os.environ["GEMINI_KEY"] = GEMINI_KEY


def file_processing(filepath):
    loader = PyPDFLoader(filepath)
    pages = loader.load()

    question_gen = ""
    for page in pages:
        question_gen += page.page_content

    splitter_ques_gen = TokenTextSplitter(
        model_name="gpt-3.5-turbo",
        chunk_size=10000,
        chunk_overlap=200)
    texts_token = splitter_ques_gen.split_text(question_gen)

    document_ques_gen = [Document(page_content=t) for t in texts_token]

    splitter_ans_gen = TokenTextSplitter(
        model_name="gpt-3.5-turbo",
        chunk_size=1000,
        chunk_overlap=100)
    document_ans_gen = splitter_ans_gen.split_documents(document_ques_gen)
    return document_ques_gen, document_ans_gen

def llm_pipeline(filepath):
    document_ques_gen, document_ans_gen=file_processing(filepath)

    gemini_llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-preview-04-17",
                                        temperature=0.3, top_p=0.85, google_api_key=GEMINI_KEY)

    PROMPT_QUESTIONS = PromptTemplate(
        template=prompt_template,
        input_variables=['text'])

    REFINE_QUESTIONS = PromptTemplate(
        template=refine_template,
        input_variables=['text', 'existing_answer'])

    ques_gen_chain = load_summarize_chain(llm=gemini_llm,
                                          chain_type='refine',
                                          question_prompt=PROMPT_QUESTIONS,
                                          refine_prompt=REFINE_QUESTIONS
                                          )
    ques=ques_gen_chain.invoke(document_ques_gen)

    gemini_embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GEMINI_KEY)

    vector_store = FAISS.from_documents(document_ans_gen, gemini_embeddings)

    ques_list = [q for q in ques['output_text'].split("\n") if q.strip() and not q.strip() == "QUESTIONS:"]

    answer_generation_chain = RetrievalQA.from_chain_type(llm=gemini_llm,
                                                          chain_type="stuff",
                                                          retriever=vector_store.as_retriever())

    return answer_generation_chain, ques_list


