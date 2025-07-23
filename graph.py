import os, asyncio, time 
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from elevenlabs import  stream
from elevenlabs.client import ElevenLabs
import chromadb, pathlib
from typing import List
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate
from pydantic import BaseModel, Field

llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=os.getenv("GOOGLE_API_KEY"))
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
client = ElevenLabs(api_key=os.getenv("ELEVENLABS_API_KEY"))
chroma_client = chromadb.HttpClient(host="localhost", port=8000)

class InterviewQuestions(BaseModel):
    questions: List[str] = Field(description="List of 5 interview questions")

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

# ------------- resume -------------
def generate_questions(db, role, resume_context):
    """Generate contextual interview questions based on role and resume content using retrieval and output parser."""
    
    # Define role-specific search queries for retrieval
    print("generating questions")
    role_queries = {
        "SDE": [
            "programming languages projects software development",
            "technical skills algorithms data structures",
            "system design architecture frameworks"
        ],
        "Data Scientist": [
            "data analysis machine learning models",
            "statistics python R SQL databases",
            "visualization analytics business insights"
        ],
        "Product Manager": [
            "product management stakeholder business",
            "project management leadership team",
            "metrics KPIs user research strategy"
        ]
    }
    
    # Retrieve relevant chunks from resume
    queries = role_queries.get(role, ["experience skills projects"])
    relevant_chunks = []
    
    for query in queries:
        docs = db.similarity_search(query, k=2)  # Get top 2 chunks per query
        relevant_chunks.extend([doc.page_content for doc in docs])
    
    # Remove duplicates while preserving order
    seen = set()
    unique_chunks = []
    for chunk in relevant_chunks:
        if chunk not in seen:
            seen.add(chunk)
            unique_chunks.append(chunk)
    
    # Combine relevant chunks
    retrieved_context = "\n\n".join(unique_chunks[:5])  # Limit to top 5 unique chunks
    
    # Setup output parser
    parser = PydanticOutputParser(pydantic_object=InterviewQuestions)
    
    # Create prompt template
    prompt_template = PromptTemplate(
        template="""Based on the candidate's resume content and the role of {role}, generate exactly 5 specific interview questions.

Retrieved relevant resume content:
{retrieved_context}

Role: {role}

Generate 5 specific questions that:
1. Reference specific points from their resume content
2. Are highly relevant to the {role} role
3. Allow the candidate to elaborate on their experience
4. Help assess their skills and expertise for this specific role
5. Are personalized based on their background

{format_instructions}

Important: Generate exactly 5 questions in the specified format.""",
        input_variables=["role", "retrieved_context"],
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )
    
    # Create the chain
    chain = prompt_template | llm | parser
    
    try:
        # Generate questions
        result = chain.invoke({
            "role": role,
            "retrieved_context": retrieved_context
        })
        print("printing questions",result.questions)
        return result.questions
    except Exception as e:
        print(f"Error parsing output: {e}")
        # Fallback to manual parsing
        fallback_prompt = f"""Based on the candidate's resume and the role of {role}, generate exactly 5 specific interview questions.

        Retrieved resume content:
        {retrieved_context}

        Role: {role}

        Generate 5 questions that reference their specific experience. Return only the questions, one per line."""
        
        response = llm.invoke(fallback_prompt).content.strip()
        questions = [q.strip().lstrip('1234567890.-) ') for q in response.split('\n') if q.strip()]
        return questions[:5]
def resume_processor(state):
    text = load_text(state["resume_path"])
    chunks = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50).split_text(text)
    db = Chroma(client=chroma_client, collection_name=f"r{int(time.time())}", embedding_function=embeddings)
    db.add_texts(chunks)

    role_topics = {
        "SDE": ["data structures", "algorithms", "system design", "testing"],
        "Data Scientist": ["ML models", "statistics", "data cleaning", "viz"],
        "Product Manager": ["roadmaps", "stakeholders", "metrics", "experiments"],
    }
    questions = generate_questions(db, state["role"], text)


    intro = llm.invoke(
        f"You are an AI interviewer.  Greet warmly and ask the candidate to introduce themselves. Role: {state['role']}"
    ).content
    return {"intro": intro, "questions": questions, "speak": speak, "history": []}

# ------------- question / eval -------------
async def ask_question(state, q):
    await speak(q)
    state["last_q"] = q
    return state

def eval_answer(state, answer):
    question = state.get("last_q", "Please introduce yourself")
    score = llm.invoke(
        f"Rate 1-10 and give 1-sentence feedback:\nQ: {question}\nA: {answer}"
    ).content
    return f"Score: {score}"

# ------------- graph -------------
def create_graph():
    # not used; we drive manually for clarity
    pass