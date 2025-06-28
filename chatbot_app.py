from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import tempfile
import os
from typing import Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# LangChain imports
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.chat_models import init_chat_model
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain import hub
from langchain_core.documents import Document
from langgraph.graph import StateGraph, START
from typing_extensions import List, TypedDict
import faiss

# Initialize FastAPI app
app = FastAPI(title="PDF RAG Chatbot", description="Upload PDF and ask questions about it")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
vector_store = None
llm = None
embeddings = None
graph = None

# Pydantic models
class QuestionRequest(BaseModel):
    question: str

class ChatResponse(BaseModel):
    answer: str
    success: bool
    message: Optional[str] = None

# Initialize AI components
def initialize_ai_components():
    global llm, embeddings
    
    # Set up Google API key
    if not os.environ.get("GOOGLE_API_KEY"):
        # For production, you should set this as an environment variable
        # For now, we'll use a placeholder - you need to set this
        raise HTTPException(status_code=500, detail="GOOGLE_API_KEY not set in environment variables")
    
    # Initialize LLM and embeddings
    llm = init_chat_model("gemini-2.0-flash", model_provider="google_genai")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# RAG State and functions
class State(TypedDict):
    question: str
    context: List[Document]
    answer: str

def retrieve(state: State):
    if vector_store is None:
        raise HTTPException(status_code=400, detail="No PDF uploaded yet")
    retrieved_docs = vector_store.similarity_search(state["question"])
    return {"context": retrieved_docs}

def generate(state: State):
    prompt = hub.pull("rlm/rag-prompt")
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    messages = prompt.invoke({"question": state["question"], "context": docs_content})
    response = llm.invoke(messages)
    return {"answer": response.content}

def create_rag_graph():
    graph_builder = StateGraph(State).add_sequence([retrieve, generate])
    graph_builder.add_edge(START, "retrieve")
    return graph_builder.compile()

@app.on_event("startup")
async def startup_event():
    """Initialize AI components on startup"""
    try:
        initialize_ai_components()
        print("AI components initialized successfully")
    except Exception as e:
        print(f"Error initializing AI components: {e}")

@app.get("/", response_class=HTMLResponse)
async def read_root():
    """Serve a simple HTML interface"""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>PDF RAG Chatbot</title>
        <style>
            body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
            .upload-section, .chat-section { margin: 20px 0; padding: 20px; border: 1px solid #ddd; border-radius: 8px; }
            input[type="file"], input[type="text"] { width: 100%; padding: 8px; margin: 5px 0; }
            button { background-color: #4CAF50; color: white; padding: 10px 20px; border: none; border-radius: 4px; cursor: pointer; }
            button:hover { background-color: #45a049; }
            .chat-history { height: 300px; overflow-y: auto; border: 1px solid #ccc; padding: 10px; margin: 10px 0; }
            .message { margin: 10px 0; padding: 8px; border-radius: 4px; }
            .user-message { background-color: #e3f2fd; text-align: right; }
            .bot-message { background-color: #f5f5f5; }
        </style>
    </head>
    <body>
        <h1>PDF RAG Chatbot</h1>
        
        <div class="upload-section">
            <h2>Upload PDF</h2>
            <input type="file" id="pdfFile" accept=".pdf">
            <button onclick="uploadPDF()">Upload PDF</button>
            <div id="uploadStatus"></div>
        </div>
        
        <div class="chat-section">
            <h2>Ask Questions</h2>
            <div id="chatHistory" class="chat-history"></div>
            <input type="text" id="questionInput" placeholder="Ask a question about the PDF..." onkeypress="handleKeyPress(event)">
            <button onclick="askQuestion()">Ask Question</button>
        </div>

        <script>
            async function uploadPDF() {
                const fileInput = document.getElementById('pdfFile');
                const file = fileInput.files[0];
                
                if (!file) {
                    alert('Please select a PDF file');
                    return;
                }
                
                const formData = new FormData();
                formData.append('file', file);
                
                document.getElementById('uploadStatus').innerHTML = 'Uploading...';
                
                try {
                    const response = await fetch('/upload-pdf/', {
                        method: 'POST',
                        body: formData
                    });
                    
                    const result = await response.json();
                    
                    if (result.success) {
                        document.getElementById('uploadStatus').innerHTML = '<span style="color: green;">PDF uploaded successfully!</span>';
                    } else {
                        document.getElementById('uploadStatus').innerHTML = '<span style="color: red;">Upload failed: ' + result.message + '</span>';
                    }
                } catch (error) {
                    document.getElementById('uploadStatus').innerHTML = '<span style="color: red;">Upload error: ' + error.message + '</span>';
                }
            }
            
            async function askQuestion() {
                const questionInput = document.getElementById('questionInput');
                const question = questionInput.value.trim();
                
                if (!question) {
                    alert('Please enter a question');
                    return;
                }
                
                // Add user message to chat
                addMessageToChat(question, 'user-message');
                questionInput.value = '';
                
                try {
                    const response = await fetch('/ask-question/', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({ question: question })
                    });
                    
                    const result = await response.json();
                    
                    if (result.success) {
                        addMessageToChat(result.answer, 'bot-message');
                    } else {
                        addMessageToChat('Error: ' + result.message, 'bot-message');
                    }
                } catch (error) {
                    addMessageToChat('Error: ' + error.message, 'bot-message');
                }
            }
            
            function addMessageToChat(message, className) {
                const chatHistory = document.getElementById('chatHistory');
                const messageDiv = document.createElement('div');
                messageDiv.className = 'message ' + className;
                messageDiv.textContent = message;
                chatHistory.appendChild(messageDiv);
                chatHistory.scrollTop = chatHistory.scrollHeight;
            }
            
            function handleKeyPress(event) {
                if (event.key === 'Enter') {
                    askQuestion();
                }
            }
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.post("/upload-pdf/")
async def upload_pdf(file: UploadFile = File(...)):
    """Upload and process a PDF file"""
    global vector_store, graph
    
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")
    
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_file_path = tmp_file.name
        
        # Load and process PDF
        loader = PyPDFLoader(tmp_file_path)
        pages = loader.load_and_split()
        
        # Create vector store
        embedding_dim = len(embeddings.embed_query("hello world"))
        index = faiss.IndexFlatL2(embedding_dim)
        
        vector_store = FAISS(
            embedding_function=embeddings,
            index=index,
            docstore=InMemoryDocstore(),
            index_to_docstore_id={},
        )
        
        # Add documents to vector store
        vector_store.add_documents(pages)
        
        # Create RAG graph
        graph = create_rag_graph()
        
        # Clean up temporary file
        os.unlink(tmp_file_path)
        
        return ChatResponse(
            answer="",
            success=True,
            message=f"PDF uploaded successfully! {len(pages)} pages processed."
        )
        
    except Exception as e:
        # Clean up temporary file if it exists
        if 'tmp_file_path' in locals():
            try:
                os.unlink(tmp_file_path)
            except:
                pass
        
        raise HTTPException(status_code=500, detail=f"Error processing PDF: {str(e)}")

@app.post("/ask-question/", response_model=ChatResponse)
async def ask_question(request: QuestionRequest):
    """Ask a question about the uploaded PDF"""
    global graph
    
    if graph is None:
        return ChatResponse(
            answer="",
            success=False,
            message="Please upload a PDF first"
        )
    
    try:
        result = graph.invoke({"question": request.question})
        return ChatResponse(
            answer=result["answer"],
            success=True
        )
        
    except Exception as e:
        return ChatResponse(
            answer="",
            success=False,
            message=f"Error processing question: {str(e)}"
        )

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "pdf_uploaded": vector_store is not None}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
