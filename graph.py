import os, asyncio, json, time, PyPDF2, docx
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from elevenlabs import  stream
from elevenlabs.client import ElevenLabs
import chromadb, pathlib

llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=os.getenv("GOOGLE_API_KEY"))
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
client = ElevenLabs(api_key=os.getenv("ELEVENLABS_API_KEY"))
chroma_client = chromadb.HttpClient(host="localhost", port=8000)

# ------------- helpers -------------
def load_text(path):
    suf = pathlib.Path(path).suffix.lower()
    if suf == ".pdf":
        from langchain_community.document_loaders import PyPDFLoader
        return "\n".join(p.page_content for p in PyPDFLoader(path).load())
    if suf == ".docx":
        from langchain_community.document_loaders import Docx2txtLoader
        return "\n".join(p.page_content for p in Docx2txtLoader(path).load())
    return pathlib.Path(path).read_text(encoding="utf-8")

async def speak(text: str):
    audio = client.text_to_speech.stream(text=text, voice_id="JBFqnCBsd6RMkjVDRZzb", model_id="eleven_flash_v2_5")
    await asyncio.get_running_loop().run_in_executor(None, lambda: stream(audio))

def generate_questions(db, role, resume_context):
    """Generate contextual interview questions based on role and resume content."""
    role_topics = {
        "SDE": [
            "technical projects and their complexity",
            "programming languages and frameworks",
            "system design and architecture",
            "problem-solving approach",
            "team collaboration and code quality"
        ],
        "Data Scientist": [
            "data analysis projects",
            "machine learning models",
            "statistical analysis",
            "data visualization",
            "business impact of data solutions"
        ],
        "Product Manager": [
            "product development experience",
            "stakeholder management",
            "metrics and KPIs",
            "user research and feedback",
            "project prioritization"
        ]
    }

    # Get role-specific topics
    topics = role_topics.get(role, [])
    
    # Generate contextual questions using LLM
    prompt = f"""Based on the candidate's resume and the role of {role}, generate 5 specific interview questions.
    Use these topics as guidance: {topics}
    
    Resume context:
    {resume_context}
    
    Generate 5 specific questions that:
    1. Reference specific points from their resume
    2. Are relevant to the {role} role
    3. Allow the candidate to elaborate on their experience
    4. Help assess their skills and expertise
    
    Format: Return only the questions, one per line, numbered 1-5."""

    questions = llm.invoke(prompt).content.strip().split('\n')
    return [q.strip() for q in questions if q.strip()]

# ------------- resume -------------
def resume_processor(state):
    # Load and chunk the resume
    text = load_text(state["resume_path"])
    chunks = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50).split_text(text)
    
    # Create vector DB
    collection_name = f"r{int(time.time())}"
    db = Chroma(client=chroma_client, collection_name=collection_name, embedding_function=embeddings)
    db.add_texts(chunks)
    
    # Get relevant resume sections for question generation
    role_query = f"experience and skills relevant to {state['role']}"
    resume_context = "\n".join([doc.page_content for doc in db.similarity_search(role_query, k=3)])
    
    # Generate role-specific questions based on resume content
    questions = generate_questions(db, state["role"], resume_context)
    
    # Generate introduction
    intro = llm.invoke(
        f"You are an AI interviewer. Greet warmly and ask the candidate to introduce themselves. Role: {state['role']}"
    ).content
    
    # Store DB reference for later use
    state["db"] = db
    state["collection_name"] = collection_name
    
    return {"intro": intro, "questions": questions, "speak": speak, "history": []}

# ------------- question / eval -------------
async def ask_question(state, q):
    await speak(q)
    state["last_q"] = q
    return state

def eval_answer(state, answer):
    question = state.get("last_q", "Please introduce yourself")
    
    # Get relevant context from resume for evaluation
    if state.get("db"):
        context = state["db"].similarity_search(question, k=1)[0].page_content
    else:
        context = ""
    
    prompt = f"""Rate the answer 1-10 and give specific feedback in one sentence.
    
    Question: {question}
    Answer: {answer}
    Relevant Resume Context: {context}
    
    Consider:
    1. Relevance to the question
    2. Depth of explanation
    3. Connection to their experience
    4. Clarity of communication"""
    
    score = llm.invoke(prompt).content
    return f"Score: {score}"

# ------------- graph -------------
def create_graph():
    # not used; we drive manually for clarity
    pass