import os
import tempfile
import shutil
from typing import List
from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
from openai import AzureOpenAI
from azure.storage.blob import BlobServiceClient
import logging
from dotenv import load_dotenv
 
# Load environment variables
load_dotenv()
 
from extract import DocumentExtractor
from chunk import TextChunker
from embed import EmbeddingService
 
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
 
# Initialize FastAPI app
app = FastAPI(title="RAG Document Q&A System", version="1.0.0")
 
# Initialize services
extractor = DocumentExtractor()
chunker = TextChunker(chunk_size=1000, overlap=200)
embedding_service = EmbeddingService()
 
# Initialize Azure Blob Storage for file uploads
blob_service_client = BlobServiceClient.from_connection_string(
    os.getenv("AZURE_STORAGE_CONNECTION_STRING")
)
 
# Initialize OpenAI client for chat
openai_client = AzureOpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    api_version="2024-02-01",
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
)
 
class QuestionRequest(BaseModel):
    question: str
 
@app.get("/")
async def root():
    """Root endpoint for health check"""
    return {"message": "RAG Document Q&A System API", "status": "running"}
 
@app.post("/upload")
async def upload_documents(files: List[UploadFile] = File(...)):
    """Upload and process multiple documents"""
    try:
        if not files or len(files) == 0:
            raise HTTPException(status_code=400, detail="No files uploaded")
       
        # Create temporary directory
        temp_dir = tempfile.mkdtemp()
        temp_files = []
       
        try:
            # Save uploaded files temporarily and upload to blob storage
            for file in files:
                if not file.filename.lower().endswith(('.pdf', '.docx')):
                    raise HTTPException(
                        status_code=400,
                        detail=f"Unsupported file type: {file.filename}. Only PDF and DOCX files are supported."
                    )
               
                # Save to temp directory for processing
                temp_path = os.path.join(temp_dir, file.filename)
                with open(temp_path, "wb") as buffer:
                    shutil.copyfileobj(file.file, buffer)
                temp_files.append(temp_path)
               
                # Upload original file to blob storage
                try:
                    file.file.seek(0)  # Reset file pointer
                    blob_client = blob_service_client.get_blob_client(
                        container="uploaded-files",
                        blob=file.filename
                    )
                    blob_client.upload_blob(file.file, overwrite=True)
                    logger.info(f"Uploaded {file.filename} to blob storage")
                except Exception as e:
                    logger.error(f"Error uploading {file.filename} to blob: {str(e)}")
           
            # Extract text from documents
            logger.info(f"Extracting text from {len(temp_files)} files...")
            extracted_docs = extractor.extract_multiple_files(temp_files)
           
            # Create chunks
            logger.info("Creating chunks...")
            chunks = chunker.chunk_documents(extracted_docs)
           
            if not chunks:
                raise HTTPException(status_code=400, detail="No text could be extracted from the uploaded files")
           
            # Upload to Azure Search (chunks will be stored in blob storage)
            logger.info("Processing chunks and uploading to Azure Search...")
            success = embedding_service.upload_documents(chunks)
           
            if not success:
                raise HTTPException(status_code=500, detail="Failed to upload documents to search index")
           
            return {
                "message": "Documents processed successfully",
                "files_processed": len([doc for doc in extracted_docs if "error" not in doc]),
                "total_chunks": len(chunks),
                "storage_info": {
                    "original_files": "uploaded-files container",
                    "processed_chunks": "processed-chunks container",
                    "embeddings": "Azure Search index"
                }
            }
           
        finally:
            # Clean up temporary files
            shutil.rmtree(temp_dir, ignore_errors=True)
           
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing documents: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing documents: {str(e)}")
 
@app.post("/ask")
async def ask_question(request: QuestionRequest):
    """Ask a question about the uploaded documents"""
    try:
        if not request.question.strip():
            raise HTTPException(status_code=400, detail="Question cannot be empty")
       
        # Search for relevant documents
        logger.info(f"Searching for: {request.question}")
        search_results = embedding_service.search_documents(request.question, top_k=5)
       
        if not search_results:
            return "I couldn't find any relevant information in the uploaded documents to answer your question."
       
        # Prepare context from search results
        context = "\n\n".join([
            f"Source: {result['filename']} (Chunk {result['chunk_index']})\n{result['text']}"
            for result in search_results
        ])
       
        # Generate answer using Azure OpenAI
        system_prompt = """You are a helpful assistant that answers questions based on the provided context from documents.
        Use only the information provided in the context to answer questions. If the context doesn't contain enough information
        to answer the question, say so. Be concise but comprehensive in your answers."""
       
        user_prompt = f"""Context from documents:
{context}
 
Question: {request.question}
 
Please answer the question based on the provided context."""
       
        response = openai_client.chat.completions.create(
            model=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=500,
            temperature=0.1
        )
       
        answer = response.choices[0].message.content
        return answer
       
    except Exception as e:
        logger.error(f"Error answering question: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error answering question: {str(e)}")
 
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8010)
 