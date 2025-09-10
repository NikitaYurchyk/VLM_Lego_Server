"""
LEGO Assembly Agent - Refactored Entry Point

This file maintains backward compatibility while using the new modular structure.
Import this instead of the old ai_agent.py file.
"""

from .lego_assembly_agent import LegoAssemblyAgent
from .models import AssemblyState, MessageClassifier, RegularInstructionClassifier, Instruction
from .vlm_providers import LLMProvider, VLMProviderManager
from .workflow import WorkflowManager
from .database_service import DatabaseService


__all__ = [
    'LegoAssemblyAgent',
    'AssemblyState', 
    'MessageClassifier',
    'RegularInstructionClassifier',
    'Instruction',
    'LLMProvider',
    'VLMProviderManager',
    'WorkflowManager',
    'DatabaseService'
]