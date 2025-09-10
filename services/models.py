"""
Data Models and Schemas Module

This module contains all the Pydantic models, TypedDicts, and dataclasses
used throughout the LEGO assembly application.
"""

from typing import Dict, List, Optional, Any, TypedDict, Literal
from pydantic import BaseModel, Field
from dataclasses import dataclass


class Instruction(TypedDict):
    instruction_id: int
    instruction_text: str
    reference_image_base64: str


@dataclass
class AssemblyState:
    current_step: int = 12
    max_steps: int = 32
    current_images: Optional[List[str]] = None  # Multiple images for comprehensive analysis
    instruction_text: Optional[str] = None
    reference_image: Optional[str] = None  # Primary reference image
    reference_images: Optional[List[str]] = None  # Additional reference images
    has_multiple_images: bool = False  # Flag indicating if multiple reference images are available
    comparison_result: Optional[Dict] = None
    feedback: Optional[str] = None
    is_step_complete: bool = False
    assembly_history: List[Dict] = None
    detected_pieces: List[str] = None
    message_type: Optional[str] = None
    user_message: Optional[str] = None
    
    def __post_init__(self):
        if self.assembly_history is None:
            self.assembly_history = []
        if self.detected_pieces is None:
            self.detected_pieces = []
        if self.current_images is None:
            self.current_images = []
        if self.reference_images is None:
            self.reference_images = []


class MessageClassifier(BaseModel):
    message_type: Literal["regular", "check"] = Field(
        ...,
        description="Classify the message. If the message contains next, next step, previous, prev, reset, zoom in, zoom out or show image, they should be classified it as 'regular'. If the message contains word check, it should be classified as 'check'.",
    )


class RegularInstructionClassifier(BaseModel):
    message_type: Literal["next", "prev", "reset", "zoom in", "zoom out", "show image", "repeat", "current step"] = Field(
        ...,
        description=
            "Classify the message."
            + "If the message contains words next or next step classify it as 'Next'"
            + "If the message contains words previous or prev, classify it as 'Prev"
            + "If the message contains words reset or start over, classify it as 'Reset"
            + "If the message contains words zoom in or scale up, classify it as 'ZoomIn'"
            + "If the message contains words zoom out or scale down, classify it as 'ZoomOut'"
            + "If the message contains words picture or image, classify it as 'ShowPicture'"
            + "If the message contains words current step or what step, classify it as 'CurrentStep'"
    )