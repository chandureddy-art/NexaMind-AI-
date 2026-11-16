import hashlib
from io import BytesIO

import chromadb
from pypdf import PdfReader
from google.genai import types


# =====================================
# CONFIGURATION
# =====================================

GEMINI_MODEL = "gemini-3.1-flash-lite"
EMBEDDING_MODEL = "gemini-embedding-001"

TOP_K = 3
MAX_CONTEXT_CHARS = 9000

# =====================================
# CHROMA DATABASE
# =====================================

db_client = chromadb.PersistentClient(
    path="vector_db"
)

collection = db_client.get_or_create_collection(
    name="nexamind_pdf_documents",
    metadata={"hnsw:space": "cosine"}
)


# =====================================
# GEMINI EMBEDDINGS
# =====================================

def create_embedding(client, text, task_type):

    response = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=text,
        config=types.EmbedContentConfig(
            task_type=task_type
        )
    )

    return response.embeddings[0].values


# =====================================
# TEXT CHUNKING
# =====================================

def split_text(text, chunk_size=1200, overlap=150):

    chunks = []

    start = 0
    step = chunk_size - overlap

    while start < len(text):

        chunk = text[start:start + chunk_size].strip()

        if chunk:
            chunks.append(chunk)

        start += step

    return chunks


# =====================================
# PDF EXTRACTION
# =====================================

def extract_pdf_chunks(file_bytes, filename):

    reader = PdfReader(BytesIO(file_bytes))

    all_chunks = []

    for page_number, page in enumerate(reader.pages, start=1):

        page_text = page.extract_text() or ""

        if not page_text.strip():
            continue

        chunks = split_text(page_text)

        for chunk_number, chunk in enumerate(chunks):

            all_chunks.append({
                "text": chunk,
                "page": page_number,
                "filename": filename,
                "chunk_number": chunk_number
            })

    return all_chunks


# =====================================
# INDEX PDF
# =====================================

def index_pdf(client, uploaded_file):

    file_bytes = uploaded_file.getvalue()
    filename = uploaded_file.name

    chunks = extract_pdf_chunks(
        file_bytes,
        filename
    )

    if not chunks:
        return 0

    file_hash = hashlib.sha256(file_bytes).hexdigest()

    ids = []
    documents = []
    embeddings = []
    metadatas = []

    existing_ids = set()

    for start in range(0, len(chunks), 16):

        batch = chunks[start:start + 16]

        batch_texts = [item["text"] for item in batch]

        response = client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=batch_texts,
            config=types.EmbedContentConfig(
                task_type="RETRIEVAL_DOCUMENT"
            )
        )

        for offset, item in enumerate(batch):

            index = start + offset
            chunk_id = f"{file_hash}_{index}"

            ids.append(chunk_id)
            documents.append(item["text"])
            embeddings.append(
                response.embeddings[offset].values
            )

            metadatas.append({
                "filename": filename,
                "page": item["page"],
                "chunk_number": item["chunk_number"]
            })

    existing = collection.get(ids=ids)
    existing_ids = set(existing["ids"])

    new_records = [
        i for i, item_id in enumerate(ids)
        if item_id not in existing_ids
    ]

    if not new_records:
        return 0

    collection.add(
        ids=[ids[i] for i in new_records],
        documents=[documents[i] for i in new_records],
        embeddings=[embeddings[i] for i in new_records],
        metadatas=[metadatas[i] for i in new_records]
    )

    return len(new_records)


# =====================================
# FAST DOCUMENT RETRIEVAL
# =====================================

def search_documents(client, question, top_k=TOP_K):

    total = collection.count()

    if total == 0:
        return []

    query_embedding = create_embedding(
        client,
        question,
        "QUESTION_ANSWERING"
    )

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, total),
        include=["documents", "metadatas", "distances"]
    )

    retrieved = []

    for document, metadata, distance in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0]
    ):

        retrieved.append({
            "text": document,
            "filename": metadata["filename"],
            "page": metadata["page"],
            "distance": distance
        })

    return retrieved


# =====================================
# PREPARE SMALL CONTEXT
# =====================================

def prepare_pdf_question(client, question):

    results = search_documents(
        client,
        question,
        top_k=TOP_K
    )

    if not results:
        return None, []

    context_parts = []
    total_chars = 0

    for item in results:

        source = (
            f"\nSource: {item['filename']} "
            f"| Page: {item['page']}\n"
        )

        content = source + item["text"]

        remaining = MAX_CONTEXT_CHARS - total_chars

        if remaining <= 0:
            break

        content = content[:remaining]

        context_parts.append(content)
        total_chars += len(content)

    context = "\n\n".join(context_parts)

    prompt = f"""
You are NexaMind AI, a PDF document assistant.

Answer the user's question using the document
information below.

Rules:
- Be accurate and direct.
- Use simple English.
- Do not invent information.
- If the answer is unavailable, say so.
- Mention source filename and page numbers.

DOCUMENT INFORMATION:
{context}

USER QUESTION:
{question}

ANSWER:
"""

    return prompt, results


# =====================================
# STREAM GEMINI ANSWERS
# =====================================

def stream_pdf_answer(client, prompt):

    response_stream = client.models.generate_content_stream(
        model=GEMINI_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.2,
            max_output_tokens=500
        )
    )

    for chunk in response_stream:

        if chunk.text:
            yield chunk.text


# =====================================
# NORMAL ANSWER FUNCTION
# =====================================

def ask_pdf_question(client, question):

    prompt, sources = prepare_pdf_question(
        client,
        question
    )

    if not prompt:
        return (
            "No indexed PDF content found. "
            "Please upload a PDF first."
        ), []

    answer = "".join(
        stream_pdf_answer(client, prompt)
    )

    return answer, sources


# =====================================
# DATABASE STATISTICS
# =====================================

def get_document_stats():

    return collection.count()