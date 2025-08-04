import os
import streamlit as st
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
from langchain_community.llms import HuggingFacePipeline
from dotenv import load_dotenv

import nltk
nltk.download('punkt')

load_dotenv()

os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.getenv("HUGGINGFACEHUB_API_TOKEN")

MODEL_NAME = "google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer, max_length=512)
llm = HuggingFacePipeline(pipeline=pipe)

def load_documents(path="Docs"):
    documents = []
    for file in os.listdir(path):
        if file.endswith(".txt"):
            try:
                loader = TextLoader(os.path.join(path, file), encoding="utf-8")
                documents.extend(loader.load())
            except Exception as e:
                st.warning(f"⚠️ Skipping {file} due to error: {e}")
    return documents

def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
    return splitter.split_documents(documents)

def embed_and_store(docs, persist_dir="chroma_db"):
    embedder = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")
    db = Chroma.from_documents(docs, embedder, persist_directory=persist_dir)
    db.persist()
    return db

def create_rag_chain(db):
    retriever = db.as_retriever(search_kwargs={"k": 3})
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        return_source_documents=True
    )
    return chain

def remove_repeated_sentences(text):
    seen = set()
    result = []
    for sentence in text.split('. '):
        sentence = sentence.strip()
        if sentence and sentence not in seen:
            result.append(sentence)
            seen.add(sentence)
    return '. '.join(result)

def evaluate_response_detailed(pred, gold):
    pred_tokens = pred.lower().split()
    gold_tokens = gold.lower().split()
    pred_set = set(pred_tokens)
    gold_set = set(gold_tokens)

    true_positive = len(pred_set & gold_set)
    false_positive = len(pred_set - gold_set)
    false_negative = len(gold_set - pred_set)

    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return {
        'true_positive': true_positive,
        'false_positive': false_positive,
        'false_negative': false_negative,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

st.set_page_config(page_title="RAG Sports Event QA App", layout="centered")
st.title("🔍 RAG: Sports Event Report App")

if 'db_loaded' not in st.session_state:
    st.session_state.db_loaded = False
if 'last_result' not in st.session_state:
    st.session_state.last_result = ''
if 'ground_truth' not in st.session_state:
    st.session_state.ground_truth = ''
if 'metrics' not in st.session_state:
    st.session_state.metrics = None

if st.button("📄 Load & Process Documents"):
    with st.spinner("Processing documents..."):
        docs = load_documents()
        if docs:
            split_docs = split_documents(docs)
            embed_and_store(split_docs)
            st.session_state.db_loaded = True
            st.success("✅ Documents Loaded and Embedded into Vector Store")
        else:
            st.warning("⚠️ No .txt documents found in /Docs folder.")

query = st.text_input("🔎 Enter your question about the sports event:", key="query_input")

prompting_style = st.radio(
    "🧠 Choose Prompting Style",
    ["Normal", "Chain of Thought"],
    horizontal=True,
    key="prompting_style",
)

if st.button("Ask"):
    if not st.session_state.db_loaded:
        st.warning("⚠️ Please load documents first.")
    elif query.strip() == "":
        st.warning("⚠️ Please enter a valid query.")
    else:
        with st.spinner("Searching for answers..."):
            embedder = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")
            db = Chroma(persist_directory="chroma_db", embedding_function=embedder)
            chain = create_rag_chain(db)

            if prompting_style == "Chain of Thought":
                prompt_query = f"{query.strip()}\n\nLet's think step by step about each detail relevant to this question."
            else:
                prompt_query = query.strip()

            output = chain.invoke({"query": prompt_query})
            # Remove repeated sentences from the answer
            cleaned_result = remove_repeated_sentences(output["result"])
            st.session_state.last_result = cleaned_result
            st.session_state.ground_truth = ""  # Reset ground truth on 

if st.session_state.last_result:
    st.write("📢 **Answer:**", st.session_state.last_result)

    ground_truth = st.text_input(
        "✍️ Enter expected answer (for F1/metrics evaluation):", 
        value=st.session_state.ground_truth,
        key="ground_truth_input"
    )
    
    if ground_truth != st.session_state.ground_truth:
        st.session_state.ground_truth = ground_truth
        if ground_truth.strip() != "":
            st.session_state.metrics = evaluate_response_detailed(st.session_state.last_result, ground_truth)
        else:
            st.session_state.metrics = None

    if st.session_state.metrics:
        m = st.session_state.metrics
        st.write("### 📝 Evaluation Metrics")
        st.write(f"- **True Positives:** {m['true_positive']}")
        st.write(f"- **False Positives:** {m['false_positive']}")
        st.write(f"- **False Negatives:** {m['false_negative']}")
        st.write(f"- **Precision:** {m['precision']:.2f}")
        st.write(f"- **Recall:** {m['recall']:.2f}")
        st.write(f"- **F1 Score:** {m['f1']:.2f}")
        st.info(
            "#### What these metrics mean:\n"
            "- **Precision:** Portion of predicted words that are actually present in the correct answer.\n"
            "- **Recall:** Portion of actual correct answer words recovered by the prediction.\n"
            "- **F1 Score:** Harmonic mean of precision & recall; balances both for fairness.\n"
            f"- **True Positives:** Correctly predicted words.\n"
            f"- **False Positives:** Extra words in the answer that aren't in the ground truth.\n"
            f"- **False Negatives:** Words missed from the correct answer."
        )
