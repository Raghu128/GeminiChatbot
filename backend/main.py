from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any # Added missing imports from previous improved version
import os
import google.generativeai as genai # CORRECTED IMPORT

# Added for async operations and robust error logging
import asyncio 
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize FastAPI app
app = FastAPI(title="Gemini API Backend", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # CONFIGURE THIS PROPERLY FOR PRODUCTION - DO NOT USE "*"
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Gemini client
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "xxxxxxx") # Placeholder is still there

# *** CORRECTED INITIALIZATION ***
# Use genai.configure for global API key setup
genai.configure(api_key=GEMINI_API_KEY)

# You can add a check here if the key isn't set, similar to what was suggested before
if not GEMINI_API_KEY or GEMINI_API_KEY == "Axxxxxxx30":
    logging.warning("⚠️  Warning: GEMINI_API_KEY is not set or is using a placeholder. "
                    "Please set your GEMINI_API_KEY environment variable. "
                    "Example: export GEMINI_API_KEY='your-actual-api-key'")
    # For a demo, you might let it proceed, but for production, you might raise an error here
    # raise ValueError("GEMINI_API_KEY environment variable not set or is a placeholder.")


# Pydantic models for request/response (copied from previous robust version for completeness)
class ChatRequest(BaseModel):
    message: str
    model: Optional[str] = "gemini-1.5-flash" # Changed default to 1.5-flash
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None

class ChatResponse(BaseModel):
    response: str
    model_used: str
    success: bool

class HealthResponse(BaseModel):
    status: str
    message: str

# Added Pydantic models for batch processing for robustness
class SingleBatchResponse(BaseModel):
    response: Optional[str] = None
    model_used: str
    success: bool
    error: Optional[str] = None

class BatchChatResponse(BaseModel):
    responses: List[SingleBatchResponse]

# Health check endpoint
@app.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(status="healthy", message="Gemini API backend is running")

# Main chat endpoint
@app.post("/chat", response_model=ChatResponse)
async def chat_with_gemini(request: ChatRequest):
    try:
        # Prepare the generation config
        generation_config_params = {}
        if request.max_tokens:
            generation_config_params["max_output_tokens"] = request.max_tokens
        if request.temperature:
            generation_config_params["temperature"] = request.temperature
        if request.top_p:
            generation_config_params["top_p"] = request.top_p
        if request.top_k:
            generation_config_params["top_k"] = request.top_k
        
        generation_config = genai.types.GenerationConfig(**generation_config_params) if generation_config_params else None
        
        # *** CORRECTED API CALL ***
        # Use genai.GenerativeModel directly, and run in a thread to avoid blocking
        model = genai.GenerativeModel(model_name=request.model)
        response = await asyncio.to_thread(
            model.generate_content,
            request.message,
            generation_config=generation_config
        )
        
        return ChatResponse(
            response=response.text,
            model_used=request.model,
            success=True
        )
    
    except Exception as e:
        logging.error(f"Error generating response: {e}", exc_info=True) # Log full traceback
        raise HTTPException(status_code=500, detail=f"Error generating response: {str(e)}")

# Streaming chat endpoint
@app.post("/chat/stream")
async def stream_chat_with_gemini(request: ChatRequest):
    from fastapi.responses import StreamingResponse
    import json
    
    async def generate_stream():
        try:
            # Prepare the generation config
            generation_config_params = {}
            if request.max_tokens:
                generation_config_params["max_output_tokens"] = request.max_tokens
            if request.temperature:
                generation_config_params["temperature"] = request.temperature
            if request.top_p:
                generation_config_params["top_p"] = request.top_p
            if request.top_k:
                generation_config_params["top_k"] = request.top_k
            
            generation_config = genai.types.GenerationConfig(**generation_config_params) if generation_config_params else None

            # *** CORRECTED API CALL FOR STREAMING ***
            # Use genai.GenerativeModel with stream=True, and run in a thread
            model = genai.GenerativeModel(model_name=request.model)
            response_iterator = await asyncio.to_thread(
                model.generate_content,
                request.message,
                generation_config=generation_config,
                stream=True # THIS IS THE KEY FOR TRUE STREAMING
            )
            
            for chunk in response_iterator:
                if hasattr(chunk, 'text'):
                    yield f"data: {json.dumps({'chunk': chunk.text, 'done': False})}\n\n"
            
            yield f"data: {json.dumps({'chunk': '', 'done': True})}\n\n"
            
        except Exception as e:
            logging.error(f"Error in streaming: {e}", exc_info=True) # Log full traceback
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
    
    return StreamingResponse(
        generate_stream(),
        media_type="text/event-stream", # Standard for Server-Sent Events (SSE)
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
    )

# Get available models
@app.get("/models")
async def get_available_models():
    try:
        # *** CORRECTED API CALL FOR LISTING MODELS ***
        # Use genai.list_models directly, and run in a thread
        models_response = await asyncio.to_thread(genai.list_models)
        
        available_models = sorted([
            m.name for m in models_response 
            if 'generateContent' in m.supported_generation_methods # Filter for models that can generate content
        ])
        
        return {"models": available_models}
    
    except Exception as e:
        logging.error(f"Error fetching models: {e}", exc_info=True) # Log full traceback
        raise HTTPException(status_code=500, detail=f"Error fetching models: {str(e)}")

# Batch processing endpoint
@app.post("/chat/batch", response_model=BatchChatResponse)
async def batch_chat_with_gemini(requests: List[ChatRequest]): # Changed to List[ChatRequest]
    async def process_single_request(req: ChatRequest):
        try:
            generation_config_params = {}
            if req.max_tokens:
                generation_config_params["max_output_tokens"] = req.max_tokens
            if req.temperature:
                generation_config_params["temperature"] = req.temperature
            if req.top_p:
                generation_config_params["top_p"] = req.top_p
            if req.top_k:
                generation_config_params["top_k"] = req.top_k
            
            generation_config = genai.types.GenerationConfig(**generation_config_params) if generation_config_params else None

            # *** CORRECTED API CALL IN BATCH PROCESSING ***
            model = genai.GenerativeModel(model_name=req.model)
            response = await asyncio.to_thread(
                model.generate_content,
                req.message,
                generation_config=generation_config
            )
            return SingleBatchResponse(
                response=response.text,
                model_used=req.model,
                success=True,
                error=None
            )
        except Exception as e:
            logging.error(f"Error in batch processing for model {req.model}, message '{req.message[:50]}...': {e}", exc_info=True)
            return SingleBatchResponse(
                response=None,
                model_used=req.model,
                success=False,
                error=str(e)
            )

    try:
        # Run all requests concurrently using asyncio.gather
        responses = await asyncio.gather(*[process_single_request(req) for req in requests])
        
        return BatchChatResponse(responses=responses)
    
    except Exception as e:
        logging.error(f"Overall error in batch processing: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error in batch processing: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    
    # The warning logic is now handled during genai.configure check above
    
    uvicorn.run(app, host="0.0.0.0", port=8000)