
import os
import re
import uuid
import tempfile
from pathlib import Path

import streamlit as st
from huggingface_hub import snapshot_download
import torch
import faiss

from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from sentence_transformers import SentenceTransformer

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)

from google import genai
from google.genai import types

import fitz
from docx import Document


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_DIR = Path(
    snapshot_download(
        repo_id="kivideveloper/ehr-conversation-router",
        repo_type="model"
    )
)

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

GEMINI_MODEL = "gemini-3.5-flash-lite"

TOP_K = 3


# ============================================================
# STREAMLIT PAGE
# ============================================================

st.set_page_config(
    page_title="Health AI Assistant",
    page_icon="🩺",
    layout="wide"
)

st.title("🩺 Privacy-Preserving Health AI Assistant")

st.caption(
    "Conversational health information and "
    "evidence-grounded medical report assistance."
)


# ============================================================
# GEMINI API
# ============================================================

try:
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
except Exception:
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:

    st.error(
        "Gemini API key is not configured. "
        "Please add GEMINI_API_KEY to Streamlit secrets."
    )

    st.stop()

client = genai.Client(
    api_key=GEMINI_API_KEY
)


# ============================================================
# LOAD ROUTER
# ============================================================

@st.cache_resource
def load_router():

    tokenizer = AutoTokenizer.from_pretrained(
        str(MODEL_DIR)
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        str(MODEL_DIR)
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    model.to(device)
    model.eval()

    return tokenizer, model, device


# ============================================================
# LOAD EMBEDDING MODEL
# ============================================================

@st.cache_resource
def load_embedding_model():

    return SentenceTransformer(
        EMBEDDING_MODEL_NAME
    )


with st.spinner("Loading AI models..."):

    tokenizer, router_model, device = load_router()

    embedding_model = load_embedding_model()


# ============================================================
# INTENTS
# ============================================================

INTENT_LABELS = [
    "FOLLOW_UP",
    "GENERAL_HEALTH_QUERY",
    "LAB_VALUE_QUERY",
    "MEDICAL_TERM_EXPLANATION",
    "MEDICATION_QUERY",
    "REPORT_QUESTION",
    "REPORT_SUMMARY",
    "REPORT_VALUE_QUESTION",
    "SYMPTOM_QUERY",
    "URGENT_CARE"
]


REPORT_KEYWORDS = {
    "report",
    "lab",
    "laboratory",
    "result",
    "results",
    "value",
    "values",
    "reference range",
    "cholesterol",
    "ldl",
    "hdl",
    "triglycerides",
    "glucose",
    "hemoglobin",
    "creatinine",
    "alt",
    "tsh"
}


FOLLOW_UP_PHRASES = {
    "is that normal",
    "is this normal",
    "what about that",
    "what about this",
    "what does that mean",
    "what does this mean",
    "what about it",
    "that value",
    "this value",
    "the previous result"
}


# ============================================================
# SESSION STATE
# ============================================================

if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []

if "messages" not in st.session_state:
    st.session_state.messages = []

if "report_id" not in st.session_state:
    st.session_state.report_id = None

if "report_chunks" not in st.session_state:
    st.session_state.report_chunks = []

if "report_index" not in st.session_state:
    st.session_state.report_index = None


# ============================================================
# PII REDACTION
# ============================================================

def redact_pii(text):

    patterns = [
        r"\b[\w\.-]+@[\w\.-]+\.\w+\b",
        r"\b(?:\+91[\s-]?)?[6-9]\d{9}\b",
        r"\b\d{1,3}[/.-]\d{1,3}[/.-]\d{2,4}\b",
        r"\b(?:Patient\s*(?:ID|Id|id)|ID)\s*[:#-]?\s*[A-Za-z0-9_-]+\b"
    ]

    for pattern in patterns:

        text = re.sub(
            pattern,
            "[REDACTED]",
            text,
            flags=re.IGNORECASE
        )

    return text


# ============================================================
# PDF EXTRACTION
# ============================================================

def extract_pdf_text(file_bytes):

    pages = []

    with fitz.open(
        stream=file_bytes,
        filetype="pdf"
    ) as document:

        for page_number, page in enumerate(document):

            text = page.get_text()

            if text.strip():

                pages.append({
                    "page": page_number + 1,
                    "text": text
                })

    return pages


# ============================================================
# DOCX EXTRACTION
# ============================================================

def extract_docx_text(file_bytes):

    temp = tempfile.NamedTemporaryFile(
        suffix=".docx",
        delete=False
    )

    temp.write(file_bytes)
    temp.close()

    try:

        document = Document(
            temp.name
        )

        text = "\n".join(
            paragraph.text
            for paragraph in document.paragraphs
            if paragraph.text.strip()
        )

        return [{
            "page": 1,
            "text": text
        }]

    finally:

        os.unlink(temp.name)


# ============================================================
# CHUNKING
# ============================================================

def create_chunks(
    pages,
    chunk_size=700,
    overlap=120
):

    chunks = []

    for page_data in pages:

        text = page_data["text"]

        start = 0

        while start < len(text):

            end = start + chunk_size

            chunk = text[start:end].strip()

            if chunk:

                chunks.append({
                    "text": redact_pii(chunk),
                    "page": page_data["page"]
                })

            start += chunk_size - overlap

    return chunks


# ============================================================
# PROCESS REPORT
# ============================================================

def process_report(uploaded_file):

    file_bytes = uploaded_file.getvalue()

    filename = uploaded_file.name.lower()

    if filename.endswith(".pdf"):

        pages = extract_pdf_text(
            file_bytes
        )

    elif filename.endswith(".docx"):

        pages = extract_docx_text(
            file_bytes
        )

    else:

        raise ValueError(
            "Please upload a PDF or DOCX file."
        )

    chunks = create_chunks(
        pages
    )

    if not chunks:

        raise ValueError(
            "No readable text found in report."
        )

    texts = [
        item["text"]
        for item in chunks
    ]

    embeddings = embedding_model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=False
    )

    embeddings = embeddings.astype(
        "float32"
    )

    index = faiss.IndexFlatIP(
        embeddings.shape[1]
    )

    index.add(
        embeddings
    )

    report_id = str(
        uuid.uuid4()
    )

    st.session_state.report_id = report_id
    st.session_state.report_chunks = chunks
    st.session_state.report_index = index

    return report_id, chunks


# ============================================================
# RETRIEVE REPORT EVIDENCE
# ============================================================

def retrieve_report_evidence(
    query,
    top_k=TOP_K
):

    if (
        st.session_state.report_index is None
        or not st.session_state.report_chunks
    ):

        return []

    query_embedding = embedding_model.encode(
        [query],
        normalize_embeddings=True,
        show_progress_bar=False
    ).astype("float32")

    k = min(
        top_k,
        len(st.session_state.report_chunks)
    )

    scores, indices = (
        st.session_state.report_index.search(
            query_embedding,
            k
        )
    )

    evidence = []

    for score, index in zip(
        scores[0],
        indices[0]
    ):

        if index < 0:
            continue

        item = st.session_state.report_chunks[index]

        evidence.append({
            "text": item["text"],
            "page": item["page"],
            "similarity": float(score)
        })

    return evidence


# ============================================================
# TRAINED ROUTER
# ============================================================

def trained_router(
    current_message,
    history
):

    history_text = "\n".join(
        f"{role}: {message}"
        for role, message in history[-6:]
    )

    mode = (
        "report_chat"
        if st.session_state.report_id
        else "health_chat"
    )

    router_text = (
        "[MODE]\n"
        + mode
        + "\n\n"
        + "[CONVERSATION HISTORY]\n"
        + history_text
        + "\n\n"
        + "[CURRENT MESSAGE]\n"
        + current_message
    )

    inputs = tokenizer(
        router_text,
        truncation=True,
        padding=True,
        max_length=512,
        return_tensors="pt"
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    with torch.no_grad():

        outputs = router_model(
            **inputs
        )

        probabilities = torch.softmax(
            outputs.logits,
            dim=-1
        )

        prediction = int(
            torch.argmax(
                probabilities,
                dim=-1
            ).item()
        )

        confidence = float(
            probabilities[
                0,
                prediction
            ].item()
        )

    label = INTENT_LABELS[prediction]

    return label, confidence


# ============================================================
# HYBRID ROUTER
# ============================================================

def route_message(
    message,
    history
):

    trained_intent, confidence = trained_router(
        message,
        history
    )

    text = message.lower().strip()

    has_report_language = any(
        keyword in text
        for keyword in REPORT_KEYWORDS
    )

    is_follow_up = any(
        phrase in text
        for phrase in FOLLOW_UP_PHRASES
    )

    if (
        st.session_state.report_id
        and is_follow_up
    ):

        return (
            "REPORT_QUESTION",
            confidence
        )

    if (
        st.session_state.report_id
        and has_report_language
    ):

        if any(
            word in text
            for word in [
                "value",
                "level",
                "reading",
                "number"
            ]
        ):

            return (
                "REPORT_VALUE_QUESTION",
                confidence
            )

        return (
            "REPORT_QUESTION",
            confidence
        )

    return (
        trained_intent,
        confidence
    )


# ============================================================
# NUMERIC VERIFICATION
# ============================================================

def extract_numeric_values(text):

    pattern = (
        r"\b\d+(?:\.\d+)?\s*"
        r"(?:mg/dL|g/dL|U/L|mIU/L|mmol/L|%)?\b"
    )

    values = re.findall(
        pattern,
        text,
        flags=re.IGNORECASE
    )

    return [
        value.strip().lower()
        for value in values
    ]


def verify_answer_against_evidence(
    answer,
    evidence
):

    answer_values = extract_numeric_values(
        answer
    )

    evidence_text = " ".join(
        item["text"]
        for item in evidence
    ).lower()

    normalized_evidence = (
        evidence_text.replace(" ", "")
    )

    unsupported = []

    for value in answer_values:

        normalized = (
            value.lower()
            .replace(" ", "")
        )

        if normalized not in normalized_evidence:

            unsupported.append(
                value
            )

    return len(unsupported) == 0


# ============================================================
# GEMINI RESPONSE
# ============================================================

def generate_gemini_response(
    user_message,
    intent,
    history,
    evidence=None
):

    evidence = evidence or []

    history_text = "\n".join(
        f"{role}: {message}"
        for role, message in history[-8:]
    )

    if evidence:

        evidence_text = "\n\n".join(
            (
                "Evidence "
                + str(i + 1)
                + " (Page "
                + str(item.get("page", "N/A"))
                + "):\n"
                + item["text"]
            )
            for i, item in enumerate(evidence)
        )

        system_instruction = (
            "You are a careful healthcare information assistant.\n"
            "You are not a doctor and must not diagnose.\n"
            "For report questions, use ONLY the supplied evidence.\n"
            "Do not invent numerical values or reference ranges.\n"
            "Do not expose private identifying information.\n"
            "Explain results in simple patient-friendly language.\n"
            "An abnormal result does not by itself establish a diagnosis."
        )

        prompt = (
            "Conversation history:\n"
            + history_text
            + "\n\nCurrent question:\n"
            + user_message
            + "\n\nDetected intent:\n"
            + intent
            + "\n\nMedical report evidence:\n"
            + evidence_text
            + "\n\nGive a concise evidence-grounded answer."
        )

    else:

        system_instruction = (
            "You are a healthcare information assistant.\n"
            "Provide general health information, not diagnosis.\n"
            "Do not claim certainty about a diagnosis.\n"
            "Do not prescribe or change medication.\n"
            "Ask useful follow-up questions when appropriate.\n"
            "For severe or life-threatening symptoms, advise urgent care."
        )

        prompt = (
            "Conversation history:\n"
            + history_text
            + "\n\nCurrent question:\n"
            + user_message
            + "\n\nDetected intent:\n"
            + intent
            + "\n\nProvide a helpful healthcare-information response."
        )

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=0.2
        )
    )

    return response.text.strip()


# ============================================================
# REPORT ANSWER
# ============================================================

def generate_report_answer(
    user_message,
    intent,
    history
):

    evidence = retrieve_report_evidence(
        user_message
    )

    if not evidence:

        return (
            "I could not find relevant evidence in "
            "the uploaded report for that question.",
            False
        )

    answer = generate_gemini_response(
        user_message,
        intent,
        history,
        evidence
    )

    verified = verify_answer_against_evidence(
        answer,
        evidence
    )

    if not verified:

        return (
            "I could not safely verify all numerical "
            "details against the uploaded report. "
            "Please ask about a specific report value.",
            False
        )

    return (
        answer,
        True
    )


# ============================================================
# URGENT RESPONSE
# ============================================================

def urgent_response():

    return (
        "Your symptoms may require urgent medical assessment. "
        "If you are experiencing severe, rapidly worsening, "
        "or potentially life-threatening symptoms, please "
        "seek immediate medical attention or contact your "
        "local emergency service."
    )


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("Medical Report")

    uploaded_file = st.file_uploader(
        "Upload PDF or DOCX",
        type=["pdf", "docx"]
    )

    if uploaded_file is not None:

        if st.button(
            "Process Report",
            use_container_width=True
        ):

            with st.spinner(
                "Processing report..."
            ):

                try:

                    report_id, chunks = process_report(
                        uploaded_file
                    )

                    st.success(
                        "Report processed successfully."
                    )

                    st.caption(
                        "Evidence chunks: "
                        + str(len(chunks))
                    )

                except Exception as error:

                    st.error(
                        "Report processing failed: "
                        + str(error)
                    )

    if st.session_state.report_id:

        st.success(
            "Active report ready"
        )

        if st.button(
            "Clear Report",
            use_container_width=True
        ):

            st.session_state.report_id = None
            st.session_state.report_chunks = []
            st.session_state.report_index = None

            st.rerun()

    st.divider()

    st.caption(
        "This system provides health information "
        "and report explanation. It does not replace "
        "professional medical care."
    )


# ============================================================
# CHAT HISTORY
# ============================================================

for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )

        if message.get("verified"):

            st.caption(
                "✓ Evidence verified against uploaded report"
            )


# ============================================================
# CHAT INPUT
# ============================================================

user_message = st.chat_input(
    "Ask a health question or ask about your report..."
)


if user_message:

    st.session_state.messages.append({
        "role": "user",
        "content": user_message
    })

    st.session_state.conversation_history.append(
        ("User", user_message)
    )

    with st.chat_message("user"):

        st.markdown(
            user_message
        )

    with st.chat_message("assistant"):

        with st.spinner("Thinking..."):

            try:

                intent, confidence = route_message(
                    user_message,
                    st.session_state.conversation_history
                )

                if intent == "URGENT_CARE":

                    answer = urgent_response()
                    verified = False

                elif (
                    st.session_state.report_id
                    and intent in {
                        "REPORT_QUESTION",
                        "REPORT_SUMMARY",
                        "REPORT_VALUE_QUESTION",
                        "FOLLOW_UP"
                    }
                ):

                    answer, verified = generate_report_answer(
                        user_message,
                        intent,
                        st.session_state.conversation_history
                    )

                else:

                    answer = generate_gemini_response(
                        user_message,
                        intent,
                        st.session_state.conversation_history
                    )

                    verified = False

                st.markdown(
                    answer
                )

                if verified:

                    st.caption(
                        "✓ Evidence verified against uploaded report"
                    )

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "verified": verified
                })

                st.session_state.conversation_history.append(
                    ("Assistant", answer)
                )

            except Exception as error:

                answer = (
                    "I encountered an error while processing "
                    "your request. Please try again."
                )

                st.error(
                    answer
                )

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "verified": False
                })

                st.session_state.conversation_history.append(
                    ("Assistant", answer)
                )
