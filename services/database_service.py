"""
Database and RAG Service Module

This module handles ChromaDB operations, instruction storage and retrieval,
and embedding functionality for the LEGO assembly system.
"""

import asyncio
import os
import json
import base64
import aiofiles
from typing import Optional
from sentence_transformers import SentenceTransformer
import chromadb
from .models import Instruction


class DatabaseService:    
    def __init__(self, instructions_db_path: str = "./lego_instructions"):
        self.instructions_db_path = instructions_db_path
        self.chroma_client = None
        self.instruction_collection = None
        self.embedding_model = None
    
    async def initialize(self):
        """Initialize the database service"""
        print("Loading SentenceTransformer model...")
        self.embedding_model = await asyncio.to_thread(SentenceTransformer, 'all-MiniLM-L6-v2')
        self.embedding_model.encode = self._encode_without_progress
        print("SentenceTransformer model loaded successfully!")
    
        self.chroma_client = chromadb.PersistentClient(path=self.instructions_db_path)
        
        class CustomEmbeddingFunction:
            def __init__(self, model):
                self.model = model
            
            def __call__(self, input):
                return self.model.encode(input).tolist()
            
            def name(self):
                return "sentence_transformer"  
            
            def dimension(self):
                return 384  
        
        try:
            self.instruction_collection = await asyncio.to_thread(
                self.chroma_client.get_or_create_collection,
                name="lego_instructions",
                embedding_function=CustomEmbeddingFunction(self.embedding_model)
            )
        except Exception as e:
            print(f"Collection conflict, recreating: {e}")
            try:
                await asyncio.to_thread(self.chroma_client.delete_collection, name="lego_instructions")
            except:
                pass
            self.instruction_collection = await asyncio.to_thread(
                self.chroma_client.create_collection,
                name="lego_instructions",
                embedding_function=CustomEmbeddingFunction(self.embedding_model)
            )        
        
        await self._load_instructions_if_needed()
    
    def _encode_without_progress(self, sentences, **kwargs):
        kwargs['show_progress_bar'] = False
        return SentenceTransformer.encode(self.embedding_model, sentences, **kwargs)
    
    async def _encode_without_progress_async(self, sentences, **kwargs):
        kwargs['show_progress_bar'] = False
        return await asyncio.to_thread(SentenceTransformer.encode, self.embedding_model, sentences, **kwargs)
    
    async def _load_instructions_if_needed(self):
        """Load instructions from JSON if database is empty"""
        try:
            existing_count = await asyncio.to_thread(self.instruction_collection.count)
            print(f"Found {existing_count} existing instructions in database")
        except:
            existing_count = 0
            
        if existing_count == 0:
            print("Database is empty. Loading instructions from JSON...")
            
            # First try to load from instructions_with_more_ref_imgs, fallback to regular instructions
            json_files = [
                './instructions_with_more_ref_imgs/lego-60399-green-race-car-readscr.json',
                './instructions/lego-60399-green-race-car-readscr.json'
            ]
            
            data = None
            json_file_used = None
            for json_file in json_files:
                if os.path.exists(json_file):
                    async with aiofiles.open(json_file, 'r') as file:
                        content = await file.read()
                        data = json.loads(content)
                        json_file_used = json_file
                        print(f"Loading instructions from: {json_file}")
                        break
            
            if data is None:
                print("Error: No instruction JSON files found!")
                return
                
            instructions = []
            for instruction in data.get("instructions"):
                processed = await self._process_instruction(instruction, json_file_used)
                if processed is not None:
                    instructions.append(processed)
            
            print(f"Processed {len(instructions)} instructions:")
            for i, instruction in enumerate(instructions):
                print(f"  {i+1}: Step {instruction['step_number']} - {instruction['instruction_text'][:50]}...")
            
            print(f"Adding {len(instructions)} instructions to database...")
            for instruction in instructions:
                await self.add_instruction(
                    instruction['step_number'], 
                    instruction['instruction_text'], 
                    instruction['reference_image_base64'],
                    instruction.get('reference_images_base64', [])
                )
            print("All instructions added to database!")
        else:
            print("Database already contains instructions. Skipping JSON processing and initialization.")
    
    async def add_instruction(self, step_number: int, instruction_text: str, reference_image_base64: str = None, reference_images_base64: list = None):
        if reference_images_base64 is None:
            reference_images_base64 = []
        
        # ChromaDB doesn't support storing lists directly in metadata
        # We'll store multiple images as separate indexed fields and count
        metadata = {
            "step": step_number,
            "reference_image": reference_image_base64,
            "num_reference_images": len(reference_images_base64),
            "has_multiple_images": len(reference_images_base64) > 0
        }
        
        # Store up to 10 reference images as separate fields
        for i, img_base64 in enumerate(reference_images_base64[:10]):
            metadata[f"reference_image_{i}"] = img_base64
        
        await asyncio.to_thread(
            self.instruction_collection.add,
            documents=[instruction_text],
            metadatas=[metadata],
            ids=[f"step_{step_number}"]
        )

    async def _process_instruction(self, instruction, json_file_path=None):
        step_number = instruction["step"]
        instruction_text = " ".join(instruction["instructions"]) if isinstance(instruction["instructions"], list) else instruction["instructions"]
        
        # Handle primary assembly image
        reference_image_base64 = None
        reference_images_base64 = []
        
        # Try to get primary image
        img_path = None
        if 'assembly_img' in instruction:
            img_path = instruction['assembly_img']
        elif 'assembly_imgs' in instruction:
            # Handle new format with multiple images path
            imgs_path = instruction['assembly_imgs']
            if imgs_path.startswith('/images/'):
                step_dir = imgs_path.replace('/images/', '')
                img_path = f"./instructions_with_more_ref_imgs/images/{step_dir}"
        
        if img_path:
            if img_path.startswith('./instructions_with_more_ref_imgs/') or 'instructions_with_more_ref_imgs' in (json_file_path or ''):
                # Handle multiple reference images from instructions_with_more_ref_imgs
                if instruction.get('assembly_imgs'):
                    imgs_path = instruction['assembly_imgs']
                    if imgs_path.startswith('/images/'):
                        step_dir = imgs_path.replace('/images/', '')
                        step_images_dir = f"./instructions_with_more_ref_imgs/images/{step_dir}"
                        
                        # Load all images from the step directory
                        if os.path.exists(step_images_dir) and os.path.isdir(step_images_dir):
                            try:
                                image_files = [f for f in os.listdir(step_images_dir) 
                                             if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]
                                image_files.sort()
                                
                                for image_file in image_files:
                                    image_path = os.path.join(step_images_dir, image_file)
                                    try:
                                        async with aiofiles.open(image_path, 'rb') as img_file:
                                            content = await img_file.read()
                                            img_base64 = base64.b64encode(content).decode('utf-8')
                                            reference_images_base64.append(img_base64)
                                            
                                            # Use first image as primary reference image
                                            if reference_image_base64 is None:
                                                reference_image_base64 = img_base64
                                                
                                    except Exception as e:
                                        print(f"Warning: Could not load image {image_path}: {e}")
                                        continue
                                        
                            except Exception as e:
                                print(f"Warning: Could not access directory {step_images_dir}: {e}")
                
                # Also try to load original single image if available
                if not reference_image_base64 and 'assembly_img' in instruction:
                    filename = instruction['assembly_img'].split('/')[-1]
                    single_img_path = f"./instructions/images/{filename}"
                    try:
                        async with aiofiles.open(single_img_path, 'rb') as img_file:
                            content = await img_file.read()
                            reference_image_base64 = base64.b64encode(content).decode('utf-8')
                    except FileNotFoundError:
                        print(f"Warning: Image file not found: {single_img_path}")
            else:
                # Handle single reference image from regular instructions
                filename = img_path.split('/')[-1]  
                single_img_path = f"./instructions/images/{filename}"
                
                try:
                    async with aiofiles.open(single_img_path, 'rb') as img_file:
                        content = await img_file.read()
                        reference_image_base64 = base64.b64encode(content).decode('utf-8')
                except FileNotFoundError:
                    print(f"Warning: Image file not found: {single_img_path}")
        
        return {
            'step_number': step_number,
            'instruction_text': instruction_text,
            'reference_image_base64': reference_image_base64,
            'reference_images_base64': reference_images_base64
        }
    
    async def query_instructions(self, step_number: int, n_results: int = 1):
        try:
            results = await asyncio.to_thread(
                self.instruction_collection.query,
                query_texts=[f"step {step_number}"],
                n_results=n_results,
                include=["documents", "metadatas"],
                where={"step": step_number}
            )
            
            # Reconstruct reference_images list from separate fields
            if results.get("metadatas"):
                for metadata_list in results["metadatas"]:
                    for metadata in metadata_list:
                        # Reconstruct the reference images list
                        reference_images = []
                        num_images = metadata.get("num_reference_images", 0)
                        
                        for i in range(num_images):
                            img_key = f"reference_image_{i}"
                            if img_key in metadata:
                                reference_images.append(metadata[img_key])
                        
                        metadata["reference_images"] = reference_images
                        
                        # Ensure has_multiple_images field exists
                        if "has_multiple_images" not in metadata:
                            metadata["has_multiple_images"] = len(reference_images) > 0
            
            return results
        except Exception as e:
            print(f"Error querying instructions for step {step_number}: {e}")
            return {"documents": [], "metadatas": []}
    
    def get_instruction_collection(self):
        return self.instruction_collection