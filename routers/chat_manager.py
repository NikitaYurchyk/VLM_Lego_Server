import json
import logging
import io
import tempfile
from typing import Dict, Union
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, UploadFile, File
from pathlib import Path
from datetime import datetime
from services.vlm_service import ChatHandler

logger = logging.getLogger(__name__)


router = APIRouter()
vlm_agent = None

def set_vlm_agent(agent):
    global vlm_agent
    vlm_agent = agent
    
class ChatManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
    
    async def connect(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        self.active_connections[client_id] = websocket
        logger.info(f"WebSocket client connected: {client_id}")
        
        await self.send_message(client_id, {
            "type": "system",
            "echo": f"Connected to AR LEGO Assembly Server! Your ID: {client_id}\n\nTIP: For best results, take 2-3 photos from different angles (front, side, top) when checking your assembly progress. Many steps have multiple reference images for detailed comparison!\n\nSay 'check' to analyze your assembly or ask for help with navigation commands.",
            "timestamp": datetime.now().isoformat(),
            "model": vlm_agent.get_current_model(),
            "provider": vlm_agent.get_current_provider()
        })
    
    def disconnect(self, client_id: str):
        if client_id in self.active_connections:
            del self.active_connections[client_id]
        logger.info(f"WebSocket client disconnected: {client_id}")
    
    async def send_message(self, client_id: str, message: dict):
        if client_id in self.active_connections:
            try:
                await self.active_connections[client_id].send_text(json.dumps(message))
                return True
            except Exception as e:
                logger.error(f"Error sending message to {client_id}: {e}")
                self.disconnect(client_id)
                return False
        return False


chat_manager = ChatManager()

@router.websocket("/chat/{client_id}")
async def chat_websocket(websocket: WebSocket, client_id: str):
    await chat_manager.connect(websocket, client_id)
    try:
        while True:
            data = await websocket.receive_text()

            try:
                message_data = json.loads(data)
                logger.info(f"Received message from {client_id}: {message_data}")                
                message_content = message_data.get("message", "")
                message_type = message_data.get("type", "chat")
                
                if message_type == "button":
                    await vlm_agent.handle_button_message(client_id, message_content)
                else:
                    await vlm_agent.handle_chat_message(client_id, message_content)
                                    
            except json.JSONDecodeError:
                logger.info(f"Received plain text from {client_id}: {data}")
                if vlm_agent:
                    await vlm_agent.handle_chat_message(client_id, data)
            
    except WebSocketDisconnect:
        chat_manager.disconnect(client_id)
    except Exception as e:
        logger.error(f"WebSocket error for {client_id}: {e}")
        chat_manager.disconnect(client_id)
        


def get_chat_manager():
    return chat_manager