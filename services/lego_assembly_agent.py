"""
Main LEGO Assembly Agent

This module contains the refactored LegoAssemblyAgent that coordinates
all the modular components for LEGO assembly guidance.
"""

import asyncio
import os
from typing import Dict, Any
from dotenv import load_dotenv

from .models import AssemblyState, MessageClassifier, RegularInstructionClassifier
from .vlm_providers import VLMProviderManager
from .workflow import WorkflowManager  
from .database_service import DatabaseService

load_dotenv()


class LegoAssemblyAgent:
    """Main agent that coordinates LEGO assembly guidance"""
    
    @classmethod
    async def create(cls, instructions_db_path: str):
        """Async factory method to create LegoAssemblyAgent instance"""
        instance = cls.__new__(cls)
        await instance._async_init(instructions_db_path)
        return instance
    
    def __init__(self):
        pass
    
    async def _async_init(self, instructions_db_path: str):
        """Initialize the agent with all modular components"""
        self.current_step = 12  
        
        self.llm_manager = VLMProviderManager()
        
        default_model = os.getenv("DEFAULT_MODEL")
        default_provider = os.getenv("DEFAULT_PROVIDER")

        
        self.llm = await self.llm_manager.create_vlm(default_model, default_provider, temperature=0.1)
        self.llm_manager.llm = self.llm
        self.llm_manager.current_model = default_model
        self.llm_manager.current_provider = self.llm_manager.detect_provider(default_model)
        
        self.classify_msg_llm = self.llm
        self.classify_regular_msg_llm = self.llm
        
        self.database_service = DatabaseService(instructions_db_path)
        await self.database_service.initialize()
        
        self.instruction_collection = self.database_service.get_instruction_collection()        
        self.workflow_manager = WorkflowManager(self)
        self.workflow = self.workflow_manager.workflow
    
    def _create_structured_output_llm(self, schema):
        return self.llm.with_structured_output(schema)
    
    
    async def switch_vlm(self, model_name: str, provider: str):
        success = await self.llm_manager.switch_vlm(model_name, provider)
        if success:
            self.llm = self.llm_manager.llm
            self.classify_msg_llm = self.llm
            self.classify_regular_msg_llm = self.llm
        return success
    
    def get_current_model(self):
        """Get current model"""
        return self.llm_manager.get_current_model()
    
    def get_current_provider(self):
        """Get current provider"""
        return self.llm_manager.get_current_provider()
    
    def get_provider_status(self):
        """Get provider status"""
        return self.llm_manager.get_provider_status()
    
    async def test_provider_connection(self, provider: str) -> bool:
        """Test provider connection"""
        return await self.llm_manager.test_provider_connection(provider)
    
    async def process_message(self, user_message: str) -> Dict[str, Any]:
        """Main method to process user messages and return feedback"""
        print("process_message started")

        initial_state = AssemblyState(
            user_message=user_message,
            current_step=self.current_step
        )
        
        result = await self.workflow.ainvoke(initial_state)
        
        self.current_step = result.get("current_step", self.current_step)
        
        return {
            "step": result.get("current_step", 1),
            "feedback": result.get("feedback", "No feedback available"),
            "is_complete": result.get("is_step_complete", False),
            "detected_pieces": result.get("detected_pieces", []),
            "message_type": result.get("message_type", "unknown"),
            "next_step": result.get("current_step", 1) + 1 if result.get("is_step_complete", False) else result.get("current_step", 1)
        }

    # Utility Methods
    async def generate_workflow_graph(self, output_file_path: str = "workflow_graph.png"):
        """Generate workflow visualization"""
        try:
            self.workflow.get_graph().draw_mermaid_png(output_file_path=output_file_path)
            return output_file_path
        except Exception as e:
            print(f"Error generating workflow graph: {e}")
            return None