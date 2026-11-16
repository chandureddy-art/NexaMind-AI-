import os
from dotenv import load_dotenv
from google import genai
from google.genai import types
import streamlit as st
from datetime import datetime
from modules.document_ai import (
    index_pdf,
    prepare_pdf_question,
    stream_pdf_answer,
    get_document_stats
)
import io
import html

from reportlab.lib.pagesizes import A4
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer
)
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    try:
        API_KEY = st.secrets["GEMINI_API_KEY"]
    except Exception:
        API_KEY = None
client = None

if API_KEY:
    client = genai.Client(api_key=API_KEY)
    
def get_ai_response(messages):

    if client is None:
        return "Gemini API key is missing. Please check your .env file."

    try:
        history = []

        for message in messages:
            role = "model" if message["role"] == "assistant" else "user"

            history.append(
                types.Content(
                    role=role,
                    parts=[types.Part(text=message["content"])]
                )
            )

        response = client.models.generate_content(
            model="gemini-3.1-flash-lite",
            contents=history,
            config=types.GenerateContentConfig(
                system_instruction=(
                    "You are NexaMind AI, an intelligent research "
                    "and knowledge assistant. Give accurate, clear, "
                    "professional and helpful answers. Explain "
                    "technical topics in beginner-friendly language."
                )
            )
        )

        return response.text or "I couldn't generate a response."

    except Exception as e:
        return f"AI Error: {str(e)}"   
def generate_research(topic, depth):

    if client is None:
        return "Gemini API key is missing. Check your .env file."

    depth_instructions = {
        "Basic": "Provide a beginner-friendly overview with essential concepts.",
        "Detailed": "Provide an in-depth explanation, examples, key findings and analysis.",
        "Advanced": "Provide an advanced technical analysis, limitations, challenges, implications and future research directions."
    }

    prompt = f"""
You are NexaMind AI Research Assistant.

Conduct a structured knowledge-based research analysis
on the following topic:

TOPIC: {topic}

RESEARCH DEPTH: {depth}

INSTRUCTIONS:
{depth_instructions[depth]}

Use the following structure:

# 1. Research Overview
# 2. Introduction
# 3. Main Concepts and Explanation
# 4. Key Findings and Insights
# 5. Advantages and Applications
# 6. Challenges and Limitations
# 7. Future Scope
# 8. Conclusion
# 9. Further Research Questions

Requirements:
- Give accurate and useful information.
- Use clear professional language.
- Explain technical terms.
- Use bullet points wherever useful.
- Do not invent statistics or citations.
- Clearly state when information needs verification.
"""

    try:
        response = client.models.generate_content(
            model="gemini-3.1-flash-lite",
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.4
            )
        )

        return response.text or "No research generated."

    except Exception as e:
        return f"Research generation error: {str(e)}"   
def generate_report(topic, report_type, details):

    if client is None:
        return "Gemini API key is missing."

    prompt = f"""
You are NexaMind AI Professional Report Generator.

Create a high-quality, professional report.

REPORT TITLE: {topic}

REPORT TYPE: {report_type}

USER REQUIREMENTS:
{details}

Follow this structure where appropriate:

1. Executive Summary
2. Introduction
3. Background and Context
4. Main Analysis and Discussion
5. Key Findings
6. Recommendations
7. Future Scope
8. Conclusion

Instructions:
- Use professional language.
- Include clear headings and subheadings.
- Explain important concepts.
- Use bullet points where appropriate.
- Keep the report relevant to the user's requirements.
- Do not invent statistics, facts or references.
- Clearly identify information requiring verification.
"""

    try:
        response = client.models.generate_content(
            model="gemini-3.1-flash-lite",
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.4
            )
        )

        return response.text or "Report generation failed."

    except Exception as e:
        return f"Report generation error: {str(e)}"


def create_report_pdf(report_text, report_title):

    buffer = io.BytesIO()

    document = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=0.7 * inch,
        leftMargin=0.7 * inch,
        topMargin=0.7 * inch,
        bottomMargin=0.7 * inch
    )

    styles = getSampleStyleSheet()

    story = []

    story.append(
        Paragraph(
            html.escape(report_title),
            styles["Title"]
        )
    )

    story.append(Spacer(1, 20))

    for line in report_text.split("\n"):

        line = line.strip()

        if not line:
            story.append(Spacer(1, 7))
            continue

        if line.startswith("# "):
            story.append(
                Paragraph(
                    html.escape(line[2:]),
                    styles["Heading1"]
                )
            )

        elif line.startswith("## "):
            story.append(
                Paragraph(
                    html.escape(line[3:]),
                    styles["Heading2"]
                )
            )

        elif line.startswith(("- ", "* ")):
            story.append(
                Paragraph(
                    "&#8226; " + html.escape(line[2:]),
                    styles["BodyText"]
                )
            )

        else:
            story.append(
                Paragraph(
                    html.escape(line),
                    styles["BodyText"]
                )
            )

        story.append(Spacer(1, 5))

    document.build(story)

    buffer.seek(0)

    return buffer.getvalue()     



# -----------------------------------
# PAGE CONFIGURATION
# -----------------------------------

st.set_page_config(
    page_title="NexaMind AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------
# CUSTOM CSS
# -----------------------------------

st.markdown("""
<style>
.stApp {
    background-color: #0b1020;
    color: #f1f5f9;
}

[data-testid="stSidebar"] {
    background-color: #111a2e;
    border-right: 1px solid #26334d;
}

[data-testid="stSidebar"] h1 {
    color: #a78bfa;
}

h1, h2, h3 {
    color: #f8fafc;
}

p, label {
    color: #cbd5e1;
}

.hero {
    background: linear-gradient(120deg, #1e1b4b, #172554);
    border: 1px solid #3730a3;
    padding: 30px;
    border-radius: 18px;
    margin-bottom: 24px;
}

.hero h1 {
    color: white;
    font-size: 36px;
    margin-bottom: 8px;
}

.hero p {
    color: #c7d2fe;
    font-size: 16px;
}

.metric-card {
    background: #151f35;
    border: 1px solid #293956;
    border-radius: 14px;
    padding: 20px;
    min-height: 120px;
}

.metric-number {
    color: #a78bfa;
    font-size: 30px;
    font-weight: bold;
}

.metric-label {
    color: #94a3b8;
    font-size: 14px;
}

.section-card {
    background: #151f35;
    border: 1px solid #293956;
    border-radius: 14px;
    padding: 20px;
    margin-bottom: 15px;
}

.stButton > button {
    border-radius: 9px;
    border: 1px solid #3b4770;
    background-color: #1b2742;
    color: white;
    width: 100%;
    min-height: 42px;
}

.stButton > button:hover {
    background-color: #4338ca;
    border-color: #818cf8;
    color: white;
}

.stTextInput input, .stTextArea textarea {
    background-color: #151f35;
    color: white;
}

[data-testid="stChatMessage"] {
    background-color: #151f35;
    border: 1px solid #293956;
    border-radius: 12px;
}

#MainMenu {visibility: hidden;}
footer {visibility: hidden;}

</style>
""", unsafe_allow_html=True)

# -----------------------------------
# SESSION STATE
# -----------------------------------

if "page" not in st.session_state:
    st.session_state.page = "Overview"

if "messages" not in st.session_state:
    st.session_state.messages = []

if "documents" not in st.session_state:
    st.session_state.documents = []
if "research_result" not in st.session_state:
    st.session_state.research_result = None

if "research_topic" not in st.session_state:
    st.session_state.research_topic = ""   
if "report_result" not in st.session_state:
    st.session_state.report_result = None

if "report_title" not in st.session_state:
    st.session_state.report_title = ""     

# -----------------------------------
# SIDEBAR
# -----------------------------------

with st.sidebar:

    st.markdown("# 🧠 NexaMind AI")
    st.caption("INTELLIGENT KNOWLEDGE PLATFORM")

    st.divider()

    pages = {
        "🏠 Overview": "Overview",
        "💬 AI Chat": "AI Chat",
        "📄 Document AI": "Document AI",
        "🔎 Research Lab": "Research Lab",
        "📝 Report Generator": "Report Generator"
    }

    for label, page in pages.items():

        if st.button(
            label,
            key=page,
            use_container_width=True
        ):
            st.session_state.page = page

    st.divider()

    st.markdown("### ⚙️ Workspace")
    st.caption("Personal AI workspace")

    st.markdown("### 🟢 System Status")
    st.caption("Dashboard: Online")
    st.caption("AI connection: Connected")

    st.divider()

    st.caption("NexaMind AI | Version 1.0")
    st.caption("Built with Python + Streamlit")

# -----------------------------------
# HEADER
# -----------------------------------

st.markdown(
    """
    <div style="display:flex; justify-content:space-between;
    align-items:center; margin-bottom:20px;">
        <div>
            <p style="color:#94a3b8; margin:0;">
                INTELLIGENT AI WORKSPACE
            </p>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# -----------------------------------
# OVERVIEW PAGE
# -----------------------------------

if st.session_state.page == "Overview":

    st.markdown("""
    <div class="hero">
        <h1>Welcome to NexaMind AI 🧠</h1>
        <p>
        Your intelligent AI research and personal
        knowledge assistant.
        </p>
        <p>
        Chat, analyze documents, research ideas
        and generate professional reports.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.subheader("Workspace Overview")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-number">AI</div>
            <div class="metric-label">Intelligent Assistant</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-number">
                {len(st.session_state.documents)}
            </div>
            <div class="metric-label">Uploaded Documents</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-number">
                {len(st.session_state.messages)}
            </div>
            <div class="metric-label">Chat Messages</div>
        </div>
        """, unsafe_allow_html=True)

    st.write("")

    st.subheader("Explore Your AI Workspace")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class="section-card">
            <h3>💬 AI Chat</h3>
            <p>Interact with your intelligent AI assistant.</p>
        </div>
        """, unsafe_allow_html=True)

        if st.button("Open AI Chat", key="open_chat"):
            st.session_state.page = "AI Chat"
            st.rerun()

        st.markdown("""
        <div class="section-card">
            <h3>📄 Document Intelligence</h3>
            <p>Upload and understand PDF documents.</p>
        </div>
        """, unsafe_allow_html=True)

        if st.button("Open Documents", key="open_docs"):
            st.session_state.page = "Document AI"
            st.rerun()

    with col2:
        st.markdown("""
        <div class="section-card">
            <h3>🔎 Research Lab</h3>
            <p>Explore topics and organize research ideas.</p>
        </div>
        """, unsafe_allow_html=True)

        if st.button("Open Research", key="open_research"):
            st.session_state.page = "Research Lab"
            st.rerun()

        st.markdown("""
        <div class="section-card">
            <h3>📝 Report Generator</h3>
            <p>Prepare structured and professional reports.</p>
        </div>
        """, unsafe_allow_html=True)

        if st.button("Open Reports", key="open_reports"):
            st.session_state.page = "Report Generator"
            st.rerun()

# -----------------------------------
# AI CHAT PAGE
# -----------------------------------
elif st.session_state.page == "AI Chat":

    st.title("💬 NexaMind AI Chat")
    st.caption("Powered by Google Gemini")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    prompt = st.chat_input("Ask NexaMind anything...")

    if prompt:

        st.session_state.messages.append({
            "role": "user",
            "content": prompt
        })

        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):

            with st.spinner("NexaMind is thinking..."):

                answer = get_ai_response(
                    st.session_state.messages
                )

                st.markdown(answer)

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer
        })
# -----------------------------------
# DOCUMENT AI PAGE
# -----------------------------------
elif st.session_state.page == "Document AI":

    st.title("📄 NexaMind Document Intelligence")

    st.caption(
        "Upload PDFs, understand their contents "
        "and ask questions using AI."
    )

    total_chunks = get_document_stats()

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Indexed Text Sections", total_chunks)

    with col2:
        st.metric("AI Engine", "Gemini RAG")

    st.divider()

    st.subheader("📤 Upload PDF Documents")

    uploaded_files = st.file_uploader(
        "Choose your PDF files",
        type=["pdf"],
        accept_multiple_files=True
    )

    if uploaded_files:

        if st.button("🚀 Process Documents",
                     use_container_width=True):

            if client is None:
                st.error("Gemini API key is missing.")

            else:
                total_added = 0

                try:
                    progress = st.progress(0)

                    with st.spinner(
                        "Reading and indexing your PDFs..."
                    ):

                        for i, uploaded_file in enumerate(
                            uploaded_files
                        ):

                            added = index_pdf(
                                client,
                                uploaded_file
                            )

                            total_added += added

                            progress.progress(
                                (i + 1) / len(uploaded_files)
                            )

                    st.success(
                        f"Processing completed! "
                        f"{total_added} new text sections indexed."
                    )

                    if total_added == 0:
                        st.info(
                            "These documents may already be indexed "
                            "or may not contain extractable text."
                        )

                    st.rerun()

                except Exception as e:
                    st.error(
                        f"Document processing error: {e}"
                    )

    st.divider()

    # PDF QUESTION ANSWERING
    st.subheader("💬 Chat With Your Documents")

    question = st.text_area(
        "Ask a question about your PDF",
        placeholder="Example: Summarize the main points.",
        height=80
    )

    if st.button(
        "Ask NexaMind AI",
        use_container_width=True
    ):

        if not question.strip():
            st.warning("Please enter a question.")

        elif client is None:
            st.error("Gemini API key is missing.")

        elif get_document_stats() == 0:
            st.warning("Upload and process a PDF first.")

        else:
            try:
                import time

                start_time = time.perf_counter()

                with st.spinner("Searching your documents..."):

                    prompt, sources = prepare_pdf_question(
                        client,
                        question
                    )

                if not prompt:
                    st.warning("No relevant PDF content found.")

                else:
                    st.markdown("### 🧠 NexaMind Answer")

                    answer = st.write_stream(
                        stream_pdf_answer(client, prompt)
                    )

                    elapsed = time.perf_counter() - start_time

                    st.caption(
                        f"Answer generated in {elapsed:.1f} seconds"
                    )

                    st.markdown("### 📚 Sources")

                    for source in sources:
                        st.caption(
                            f"📄 {source['filename']} "
                            f"| Page {source['page']}"
                        )

            except Exception as e:
                st.error(f"AI Error: {e}")
       
# -----------------------------------
# RESEARCH LAB PAGE
# -----------------------------------

elif st.session_state.page == "Research Lab":

    st.title("🔎 NexaMind AI Research Lab")

    st.caption(
        "AI-powered research and knowledge discovery"
    )

    st.markdown("""
    <div class="section-card">
        <h3>🧠 Intelligent Research Assistant</h3>
        <p>
        Explore topics, discover insights and generate
        structured research using Google Gemini.
        </p>
    </div>
    """, unsafe_allow_html=True)

    topic = st.text_input(
        "Enter your research topic",
        placeholder="Example: Future of Generative AI"
    )

    research_depth = st.selectbox(
        "Select Research Depth",
        ["Basic", "Detailed", "Advanced"]
    )

    if st.button(
        "🚀 Start AI Research",
        use_container_width=True
    ):

        if not topic.strip():
            st.warning("Please enter a research topic.")

        elif client is None:
            st.error("Gemini API key is missing.")

        else:
            with st.spinner(
                "NexaMind is researching your topic..."
            ):

                result = generate_research(
                    topic,
                    research_depth
                )

                st.session_state.research_result = result
                st.session_state.research_topic = topic

    if "research_result" in st.session_state:

        st.divider()

        st.subheader("📚 Research Results")

        st.caption(
            f"Topic: {st.session_state.research_topic}"
        )

        st.markdown(
            st.session_state.research_result
        )

        st.divider()

        st.download_button(
            label="📥 Download Research",
            data=st.session_state.research_result,
            file_name="NexaMind_Research.md",
            mime="text/markdown",
            use_container_width=True
        )          
                 
# -----------------------------------
# REPORT GENERATOR PAGE
# -----------------------------------

elif st.session_state.page == "Report Generator":

    st.title("📝 NexaMind AI Report Generator")

    st.caption(
        "Generate professional reports using Google Gemini."
    )

    st.markdown("""
    <div class="section-card">
        <h3>📑 Intelligent Report Creation</h3>
        <p>
        Transform your ideas into structured,
        professional and downloadable reports.
        </p>
    </div>
    """, unsafe_allow_html=True)

    report_topic = st.text_input(
        "Report Title",
        placeholder="Example: Impact of Generative AI"
    )

    report_type = st.selectbox(
        "Select Report Type",
        [
            "Professional Report",
            "Research Report",
            "Technical Report",
            "Project Report"
        ]
    )

    report_details = st.text_area(
        "Describe your report requirements",
        height=150,
        placeholder="Explain what your report should contain..."
    )

    if st.button(
        "🚀 Generate AI Report",
        use_container_width=True
    ):

        if not report_topic.strip():
            st.warning("Please enter a report title.")

        elif not report_details.strip():
            st.warning("Please enter report requirements.")

        elif client is None:
            st.error("Gemini API key is missing.")

        else:

            with st.spinner(
                "NexaMind is preparing your report..."
            ):

                result = generate_report(
                    report_topic,
                    report_type,
                    report_details
                )

                if result and not result.startswith(
                    "Report generation error:"
                ) and not result.startswith(
                    "Gemini API key is missing."
                ):

                    st.session_state.report_result = result
                    st.session_state.report_title = report_topic

                else:
                    st.error(result)

    if st.session_state.report_result:

        st.divider()

        st.subheader("📄 Generated Report")

        st.caption(
            f"Title: {st.session_state.report_title}"
        )

        st.markdown(
            st.session_state.report_result
        )

        try:

            pdf_file = create_report_pdf(
                st.session_state.report_result,
                st.session_state.report_title
            )

            st.download_button(
                "📥 Download Professional PDF",
                data=pdf_file,
                file_name="NexaMind_AI_Report.pdf",
                mime="application/pdf",
                use_container_width=True
            )

        except Exception as e:
            st.error(f"PDF creation error: {e}")


# -----------------------------------
# FOOTER
# -----------------------------------

st.divider()

st.caption(
    "© 2026 NexaMind AI | Intelligent Research & "
    "Knowledge Assistant"
)