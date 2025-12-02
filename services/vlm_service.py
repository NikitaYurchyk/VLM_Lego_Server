import asyncio
import json
import logging
from typing import Dict, Optional
import aiohttp
import os
import re
from datetime import datetime
from aiopath import AsyncPath
import aiofiles
import base64
from dotenv import load_dotenv

# 
logger = logging.getLogger(__name__)

class ChatHandler:
    def __init__(self, chat_manager, agent):
        self.chat_manager = chat_manager
        self.agent = agent
    
    def get_current_model(self):
        """Get current model from the agent"""
        return self.agent.get_current_model() if self.agent else "unknown"
    
    def get_current_provider(self):
        """Get current provider from the agent"""
        return self.agent.get_current_provider() if self.agent else "unknown"

    async def handle_chat_message(self, client_id: str, message: str):
        try:
            if not self.agent:
                response = {
                    "type": "system",
                    "echo": "VLM agent not available. Please check configuration.",
                    "timestamp": datetime.now().isoformat(),
                    "client_id": client_id,
                    "model_name": self.agent.get_current_model(),
                    "provider": self.agent.get_current_provider()
                }
        
                
                await self.chat_manager.send_message(client_id, response)
                return
            
            
            vlm_response = await self.agent.process_message(message)
            
            response = {
                "type": "chat",
                "message": {
                    "step": vlm_response.get("step", 1),
                    "feedback": vlm_response.get("feedback", ""),
                    "is_complete": vlm_response.get("is_complete", False),
                    "detected_pieces": vlm_response.get("detected_pieces", []),
                    "message_type": vlm_response.get("message_type", "unknown"),
                    "next_step": vlm_response.get("next_step", 1)
                },
                "original_message": message,
                "timestamp": datetime.now().isoformat(),
                "client_id": client_id,
                "model_name": self.agent.get_current_model(),
                "provider": self.agent.get_current_provider()
            }
            
            await self.chat_manager.send_message(client_id, response)
            logger.info(f"Sent VLM chat response to {client_id}")
            
        except TimeoutError as e:
            logger.error(f"Timeout error handling VLM chat message: {e}")
            timeout_response = {
                "type": "error", 
                "echo": f"Timeout: {str(e)}",
                "timestamp": datetime.now().isoformat(),
                "client_id": client_id,
                "model_name": self.get_current_model(),
                "provider": self.get_current_provider()
            }
            await self.chat_manager.send_message(client_id, timeout_response)
        except Exception as e:
            logger.error(f"Error handling VLM chat message: {e}")
            error_response = {
                "type": "error", 
                "echo": f"Error processing message: {str(e)}",
                "timestamp": datetime.now().isoformat(),
                "client_id": client_id,
                "model_name": self.get_current_model(),
                "provider": self.get_current_provider()
            }
            await self.chat_manager.send_message(client_id, error_response)

    async def handle_button_message(self, client_id: str, message: str):
        try:
            if not self.agent:
                response = {
                    "type": "system",
                    "echo": "VLM agent not available. Please check configuration.",
                    "timestamp": datetime.now().isoformat(),
                    "client_id": client_id,
                    "model_name": self.get_current_model(),
                    "provider": self.get_current_provider()
                }
                await self.chat_manager.send_message(client_id, response)
                return
            
            vlm_response = await self.agent.process_message(message)
            
            response = {
                "type": "button",
                "message": {
                    "step": vlm_response.get("step", 1),
                    "feedback": vlm_response.get("feedback", ""),
                    "is_complete": vlm_response.get("is_complete", False),
                    "detected_pieces": vlm_response.get("detected_pieces", []),
                    "message_type": vlm_response.get("message_type", "unknown"),
                    "next_step": vlm_response.get("next_step", 1)
                },
                "original_message": message,
                "timestamp": datetime.now().isoformat(),
                "client_id": client_id,
                "model_name": self.agent.get_current_model(),
                "provider": self.agent.get_current_provider()
            }
            
            await self.chat_manager.send_message(client_id, response)
            logger.info(f"Sent VLM button response to {client_id}")
            
        except TimeoutError as e:
            logger.error(f"Timeout error handling VLM button message: {e}")
            timeout_response = {
                "type": "error", 
                "echo": f"Timeout: {str(e)}",
                "timestamp": datetime.now().isoformat(),
                "client_id": client_id,
                "model_name": self.get_current_model(),
                "provider": self.get_current_provider()
            }
            await self.chat_manager.send_message(client_id, timeout_response)
        except Exception as e:
            logger.error(f"Error handling VLM button message: {e}")
            error_response = {
                "type": "error", 
                "echo": f"Error processing button message: {str(e)}",
                "timestamp": datetime.now().isoformat(),
                "client_id": client_id,
                "model_name": self.get_current_model(),
                "provider": self.get_current_provider()
            }
            await self.chat_manager.send_message(client_id, error_response)

   
    def load_image(self, image_path):
        try:
            with open(image_path, 'rb') as img_file:
                content = img_file.read()
                return base64.b64encode(content).decode('utf-8')
        except FileNotFoundError:
            print(f"Warning: Image file not found: {image_path}")
            return None

    


