import uvicorn
import datetime
import logging
import os
from pydantic import BaseModel
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from routers import image_uploader
from routers import audio_uploader
from routers import chat_manager
from routers import models
from routers import lego
from routers.chat_manager import get_chat_manager, set_vlm_agent, router as chat_router
from routers.models import set_vlm_agent as set_models_vlm_agent
from routers.lego import set_vlm_agent as set_lego_state_vlm_agent

from dotenv import load_dotenv
from services.vlm_service import ChatHandler
from services.ai_agent_refactored import LegoAssemblyAgent

load_dotenv()



logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

chat_manager = get_chat_manager()

agent = None
vlm_agent = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global agent, vlm_agent
    print("Initializing AI agent asynchronously...")
    agent = await LegoAssemblyAgent.create(
        instructions_db_path="./lego_instructions"
    )
    vlm_agent = ChatHandler(chat_manager=chat_manager, agent=agent)
    set_vlm_agent(vlm_agent)
    set_models_vlm_agent(vlm_agent)
    set_lego_state_vlm_agent(vlm_agent)
    audio_uploader.set_chat_handler(vlm_agent)
    print("AI agent initialized successfully!")
    yield
    print("Shutting down...")


app = FastAPI(
    lifespan=lifespan,
    title="AR LEGO Assembly - Complete Server", 
    description="FastAPI server with image upload, VLM chat, manual crawling, and dataset preprocessing for LEGO building guidance",
    version="4.0.0",
    
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.datetime.now().isoformat(),
        "server": "AR LEGO Assembly Server",
        "version": "4.0.0",
        "message": "Server is running and accepting connections"
    }
    
    
app.include_router(image_uploader.router, tags=["upload"])
app.include_router(audio_uploader.router, tags=["upload"])
app.include_router(chat_router, tags=["websocket"])
app.include_router(models.router, tags=["models"])
app.include_router(lego.router, tags=["lego_state"])






if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )