"""
LLM Provider Management Module

This module handles different LLM providers including cloud-based (OpenRouter, Anthropic, OpenAI)
and local providers (MLX-VLM) with automatic provider detection and switching capabilities.
"""

import asyncio
import os
import sys
import subprocess
import base64
import tempfile
import json
import re
from enum import Enum
from typing import Optional, Dict, Any
import requests
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage


class LLMProvider(Enum):
    OPENROUTER = "openrouter"
    MLX_VLM = "mlx_vlm"
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    LM_STUDIO = "lm_studio"


class ModelConfig:
    def __init__(self, provider: LLMProvider, model_id: str, base_url: Optional[str] = None, **kwargs):
        self.provider = provider
        self.model_id = model_id
        self.base_url = base_url
        self.config = kwargs




class VLMProviderManager:
    
    def __init__(self):
        self.current_provider = None
        self.current_model = None
        self.llm = None
        
    def detect_provider(self, model: str) -> LLMProvider:
        if model is None:
            return LLMProvider.LM_STUDIO  # Default fallback
        model_lower = model.lower()
        
        if model_lower.startswith("lm-studio/"):
            return LLMProvider.LM_STUDIO
        
        if "claude" in model_lower:
            if "anthropic/" in model:
                return LLMProvider.OPENROUTER  
            else:
                return LLMProvider.ANTHROPIC 
        
        if "/" in model:
            if "lm-studio/" in model:
                return LLMProvider.LM_STUDIO
            elif "mlx-community/" in model:
                return LLMProvider.LM_STUDIO 
            elif "openai/" in model or "gpt" in model_lower:
                return LLMProvider.OPENROUTER 
            elif "meta/" in model or "llama" in model_lower:
                return LLMProvider.OPENROUTER  
            elif "google/" in model or "gemini" in model_lower:
                return LLMProvider.OPENROUTER  
            else:
                return LLMProvider.OPENROUTER
        else:
            local_patterns = ['qwen', 'llava', 'pixtral', 'phi', 'mistral', 'gemma', 'llama']
            if any(pattern in model_lower for pattern in local_patterns):
                return LLMProvider.LM_STUDIO
            return LLMProvider.OPENROUTER
    
    def get_available_providers(self) -> Dict[str, bool]:
        lm_studio_available = False
        try:
            response = requests.get("http://localhost:1234/v1/models", timeout=2)
            lm_studio_available = response.status_code == 200
        except:
            lm_studio_available = False
        
        return {
            "openrouter": bool(os.getenv("OPENROUTER_API_KEY")),
            "lm_studio": lm_studio_available,
            "anthropic": bool(os.getenv("ANTHROPIC_API_KEY")),
            "openai": bool(os.getenv("OPENAI_API_KEY"))
        }

        
        
    async def create_vlm(self, model: str, provider:str, temperature: float = 0.1, **kwargs):
        try:
            if isinstance(provider, str):
                provider_enum = LLMProvider(provider)
            else:
                provider_enum = provider
            if provider_enum == LLMProvider.OPENROUTER:                
                llm = ChatOpenAI(
                    api_key=os.getenv("OPENROUTER_API_KEY"),
                    base_url=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
                    model=model,
                    temperature=temperature,
                    default_headers={
                        "HTTP-Referer": os.getenv("YOUR_SITE_URL", "http://localhost:8000"),
                        "X-Title": os.getenv("YOUR_SITE_NAME", "AR LEGO Assembly"),
                    },
                    **kwargs
                )
                llm._provider_type = "openrouter"
                return llm
            
            elif provider_enum == LLMProvider.LM_STUDIO:
                llm = ChatOpenAI(
                    base_url="http://localhost:1234/v1",
                    api_key="lm-studio",  
                    model="google/gemma-3-4b",
                    temperature=temperature,
                    max_tokens=kwargs.get("max_tokens", 1024),
                )
                return llm
                
            elif provider_enum == LLMProvider.ANTHROPIC:
                llm = ChatAnthropic(
                    api_key=os.getenv("ANTHROPIC_API_KEY"),
                    model=model,
                    temperature=temperature,
                    **kwargs
                )
                llm._provider_type = "anthropic"
                return llm
            
            else:
                raise ValueError(f"Unsupported provider: {provider_enum}")
                
        except Exception as e:
            print(f"Error creating LLM for provider {provider}: {e}")
            return None
            
    
    async def switch_vlm(self, _model_name:str, _provider: str):
        try:
            new_llm = await self.create_vlm(model=_model_name, provider=_provider)
            if new_llm is None:
                print(f"Failed to create LLM for {_model_name} with provider {_provider}")
                return False
                
            self.llm = new_llm
            self.current_model = _model_name
            self.current_provider = LLMProvider(_provider) if isinstance(_provider, str) else _provider
            print(f"VLM set to {_model_name} provider {self.current_provider}")
            return True
        except Exception as e:
            print(f"Error switching model: {e}")
            return False            

    
    def get_current_model(self):
        return self.current_model
    
    def get_current_provider(self):
        return self.current_provider.value if self.current_provider else None
    
    def get_provider_status(self):
        return {
            "current": {
                "provider": self.get_current_provider(),
                "model": self.current_model
            },
            "available_providers": self.get_available_providers(),
            "provider_urls": {
                "openrouter": os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
                "mlx_vlm": "local",
                "anthropic": "https://api.anthropic.com",
                "openai": "https://api.openai.com/v1"
            }
        }
    
    