import streamlit as st
from pypdf import PdfReader
import docx
import re

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# --- Helpers ---
def safe_context(text: str) -> str:
    """Escape curly braces so LangChain doesn't treat them as variables."""
    return text.replace("{", "{{").replace("}", "}}")

def read_file(file):
    if file is None:
        return ""
    if file.type == "text/plain":
        return file.read().decode("utf-8")
    elif file.type == "application/pdf":
        pdf = PdfReader(file)
        return "\n".join([page.extract_text() for page in pdf.pages if page.extract_text()])
    elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        doc_obj = docx.Document(file)
        return "\n".join([para.text for para in doc_obj.paragraphs])
    elif file.type == "text/html":
        return file.read().decode("utf-8")
    else:
        return ""

def strip_references(text):
    """Remove everything after 'References' to avoid noise."""
    lowered = text.lower()
    if "references" in lowered:
        return text[:lowered.index("references")]
    return text

# --- LLM setup ---
# Explicitly pass API key from Streamlit secrets
openai_key = st.secrets["openai"]["api_key"]

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    api_key=openai_key
)

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    api_key=openai_key
)

# Prompt for Q&A
qa_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a question-answering assistant. Use the provided context to answer the question. "
               "Summarize clearly even if the information is described narratively rather than in tables. "
               "Prefer quoting exact words when possible, but paraphrase if needed. "
               "If the context contains a list (e.g. control variables, hypotheses), enumerate them explicitly. "
               "If information is spread across multiple chunks, combine it into a complete answer. "
               "Only reply 'Not found in document' if the concept truly does not appear anywhere."),
    ("human", "Question: {question}\n\nContext:\n{context}")
])

# Prompt for abbreviation index
abbr_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an assistant that formats abbreviation candidates. "
               "Only output in the format 'ABBR: definition'. Do not summarize or add."),
    ("human", "Candidates:\n{candidates}")
])

def ask_llm(question, context):
    chain = qa_prompt | llm
    response = chain.invoke({"question": question, "context": safe_context(context)})
    return response.content

def extract_abbreviations(context):
    body_only = strip_references(context)

    # Regex patterns to catch multiple definition styles
    patterns = [
        r"\b([A-Z][A-Za-z\s]+?)\s\(([A-Z]{2,})\)",   # Term (ABBR)
        r"\b([A-Z]{2,})\s*[:=]\s*([A-Za-z\s]+)",     # ABBR: Term or ABBR = Term
        r"\b([A-Z]{2,}),\s*([A-Za-z\s]+)",           # ABBR, Term
        r"\b([A-Z]{2,})\s+is\s+([A-Za-z\s]+)"        # ABBR is Term
    ]

    seen = {}
    for pat in patterns:
        for full, abbr in re.findall(pat, body_only):
            abbr = abbr.strip()
            full = full.strip()
            if abbr not in seen:
                seen[abbr] = full

    if not seen:
        return "None found."

    candidates = [f"{abbr}: {seen[abbr]}" for abbr in seen]

    chain = abbr_prompt | llm
    response = chain.invoke({"candidates": "\n".join(candidates)})
    return response.content

# --- Streamlit UI ---
st.title("Document Q&A and Abbreviation Index")

mode = st.radio("Choose mode:", ["Q&A", "Abbreviation Index"])
question = None
if mode == "Q&A":
    question = st.text_input("Enter your question")

files = st.file_uploader("Upload one or more files", type=["txt", "pdf", "docx", "html"], accept_multiple_files=True)

if st.button("Run"):
    if files:
        for file in files:
            st.subheader(f"Results for {file.name}")
            context = read_file(file)

            if context.strip():
                # Structure-aware chunking
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=400,
                    chunk_overlap=150,
                    separators=["\n\n", "\n", ".", " "]
                )
                docs = splitter.create_documents([strip_references(context)])

                vectorstore = FAISS.from_documents(docs, embeddings)

                if mode == "Q&A" and question:
                    retrieved_docs = vectorstore.similarity_search(question, k=50)
                    chosen_context = "\n\n---\n\n".join([d.page_content for d in retrieved_docs[:30]])
                    answer = ask_llm(question, chosen_context)
                    st.write("AI Response:")
                    st.write(answer)

                elif mode == "Abbreviation Index":
                    combined_context = "\n\n---\n\n".join([d.page_content for d in docs])
                    abbr_index = extract_abbreviations(combined_context)
                    st.write("Abbreviation Index:")
                    st.write(abbr_index)
            else:
                st.write("No text extracted from this file.")
    else:
        st.write("Please upload at least one file.")
