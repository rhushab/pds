from flask import Flask, request, jsonify
from pymongo import MongoClient
from flask_cors import CORS
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_cohere import CohereEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Cohere
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.schema import Document

app = Flask(__name__)
CORS(app)

CONNECTION_STRING = "mongodb+srv://risvarrt:esL2FmMzdvyZxxPL@cluster.otyarm0.mongodb.net/"
DB_NAME = "csci6409"
COLLECTION_NAME = "embeddings"
persist_directory = 'docs/chroma/'

client = MongoClient(CONNECTION_STRING)
db = client[DB_NAME]
collection = db[COLLECTION_NAME]

os.environ['COHERE_API_KEY'] = 'ACs49HfYWoXI2X58AfNe0pFhLRgPN22KTio8pwmk'

@app.route('/upload', methods=['POST'])
def upload_pdf():
    files = request.files.getlist('files')
    file_paths = []

    for file in files:
        file_path = f"./uploads/{file.filename}"
        file.save(file_path)
        file_paths.append(file_path)

    splits = load_and_split_pdfs(file_paths)
    embeddings = generate_embeddings(splits)
    store_embeddings(splits, embeddings, collection)

    return jsonify({"message": "Files processed and embeddings stored. Please ask your question."})

def load_and_split_pdfs(files, chunk_size=1000, chunk_overlap=100):
    loaders = [PyPDFLoader(file) for file in files]
    docs = []
    for loader in loaders:
        docs.extend(loader.load())
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    splits = text_splitter.split_documents(docs)
    return splits

def generate_embeddings(splits):
    user_agent = "contextual_knowledge_search/1.0"
    embedding = CohereEmbeddings(model="embed-multilingual-v2.0", user_agent=user_agent)
    texts = [chunk.page_content for chunk in splits]
    embeddings = embedding.embed(texts)
    return embeddings

def store_embeddings(splits, embeddings, collection):
    docs_for_db = [
        {
            "text": splits[i].page_content,
            "embedding": embeddings[i]
        }
        for i in range(len(splits))
    ]
    collection.insert_many(docs_for_db)

@app.route('/query', methods=['POST'])
def query():
    question = request.json.get('question')
    langchain_documents = fetch_documents(collection)
    vectordb = Chroma.from_documents(
        documents=langchain_documents,
        embedding=CohereEmbeddings(model="embed-multilingual-v2.0", user_agent="contextual_knowledge_search/1.0"),
        persist_directory=persist_directory
    )
    qa_chain = create_qa_chain(vectordb)
    answer = process_query(question, qa_chain)
    return jsonify({"answer": answer})

def fetch_documents(collection):
    documents = list(collection.find({}))
    return [
        Document(page_content=doc['text'], metadata={})
        for doc in documents
    ]

def create_qa_chain(vectordb):
    llm = Cohere(model="command", temperature=0)
    template = """Use the following pieces of context to answer the question:
    {context}
    Question: {question}
    Helpful Answer:"""
    QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context", "question"], template=template)
    qa_chain = RetrievalQA.from_chain_type(llm, retriever=vectordb.as_retriever(), return_source_documents=True, chain_type_kwargs={"prompt": QA_CHAIN_PROMPT})
    return qa_chain

def process_query(question, qa_chain):
    result = qa_chain({"query": question})
    return result['result']

if __name__ == '__main__':
    app.run(debug=True)
