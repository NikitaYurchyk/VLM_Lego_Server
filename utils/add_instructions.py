import asyncio
import os
import base64
import aiofiles
from typing import List


async def add_instructions(self, list_of_instructions: List[dict]):
    tasks = [self.add_instruction(
        instruction["step_number"], 
        instruction["instruction_text"], 
        instruction.get("reference_image_base64"),
        instruction.get("reference_images_base64", [])
    ) for instruction in list_of_instructions]
    await asyncio.gather(*tasks)
        
async def add_instruction(self, step_number: int, instruction_text: str, reference_image_base64: str = None, reference_images_base64: List[str] = None):
    """Add instruction to the database with support for multiple reference images"""
    await self._add_instruction_async(step_number, instruction_text, reference_image_base64, reference_images_base64)

async def _add_instruction_async(self, step_number: int, instruction_text: str, reference_image_base64: str = None, reference_images_base64: List[str] = None):
    """Async version of add instruction with support for multiple reference images"""
    if reference_images_base64 is None:
        reference_images_base64 = []
    
    await asyncio.to_thread(
        self.instruction_collection.add,
        documents=[instruction_text],
        metadatas=[{
            "step": step_number,
            "reference_image": reference_image_base64,
            "reference_images": reference_images_base64,
            "has_multiple_images": len(reference_images_base64) > 0
        }],
        ids=[f"step_{step_number}"]
    )

async def load_multiple_reference_images(step_number: int, images_path: str = "./instructions_with_more_ref_imgs/images") -> List[str]:
    """Load multiple reference images for a given step from the instructions_with_more_ref_imgs directory"""
    step_images_path = os.path.join(images_path, str(step_number))
    reference_images_base64 = []
    
    if os.path.exists(step_images_path) and os.path.isdir(step_images_path):
        try:
            # Get all image files in the step directory
            image_files = [f for f in os.listdir(step_images_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]
            image_files.sort()  # Sort for consistent ordering
            
            for image_file in image_files:
                image_path = os.path.join(step_images_path, image_file)
                try:
                    async with aiofiles.open(image_path, 'rb') as img_file:
                        content = await img_file.read()
                        img_base64 = base64.b64encode(content).decode('utf-8')
                        reference_images_base64.append(img_base64)
                except Exception as e:
                    print(f"Warning: Could not load image {image_path}: {e}")
                    continue
                    
        except Exception as e:
            print(f"Warning: Could not access directory {step_images_path}: {e}")
    
    return reference_images_base64
    
