import asyncio
import aiohttp
import aiofiles
from bs4 import BeautifulSoup
import json
import os
from pathlib import Path
import re
from typing import Optional, Dict, List, Any
from dotenv import load_dotenv

load_dotenv()


class LegoDatasetProcessor:
    def __init__(self):
        self.manual_id = os.getenv("LEGO_SET_NAME", "lego-60399-green-race-car-readscr")
        self.set_number = self._extract_set_number(self.manual_id)
        self.set_name = os.getenv("LEGO_SET_DISPLAY_NAME", "Green Race Car")
        self.base_url = os.getenv("LEGO_BASE_URL", 
            "https://legoaudioinstructions.com/wp-content/themes/mtt-wordpress-theme/assets/manual/manual-images")
        self.output_folder = os.getenv("LEGO_OUTPUT_FOLDER", "json_file_updated")
        self.max_concurrent_downloads = int(os.getenv("MAX_CONCURRENT_DOWNLOADS", "5"))
        
        # Input directory patterns from env or defaults
        self.input_directories = self._get_input_directories()

    def _extract_set_number(self, manual_id: str) -> str:
        """Extract set number from manual ID"""
        match = re.search(r'(\d+)', manual_id)
        return match.group(1) if match else "unknown"

    def _get_input_directories(self) -> List[str]:
        """Get input directories from environment or use defaults"""
        env_dirs = os.getenv("LEGO_INPUT_DIRECTORIES")
        if env_dirs:
            return [dir.strip() for dir in env_dirs.split(",")]
        
        return [
            f'../data/vox_arta_dataset/manuals/lego/{self.manual_id}',
            f'J.Pei/data/vox_arta_dataset/manuals/lego/{self.manual_id}',
            f'ARTA_LEGO/lego/{self.manual_id}',
            f'lego_data/{self.manual_id}',
            f'data/{self.manual_id}',
            self.manual_id
        ]

    async def download_single_image(self, session: aiohttp.ClientSession, img_url: str, 
                                  directory: str) -> bool:
        """Download a single image asynchronously"""
        try:
            async with session.get(img_url) as response:
                if response.status == 200:
                    images_folder = Path(directory) / 'images'
                    images_folder.mkdir(exist_ok=True)
                    
                    # Extract filename from URL
                    img_name = os.path.basename(img_url.split('?')[0])
                    if not img_name.lower().endswith(('.jpg', '.jpeg', '.png', '.gif')):
                        img_name += '.jpg'
                    
                    save_path = images_folder / img_name
                    content = await response.read()
                    
                    async with aiofiles.open(save_path, 'wb') as img_file:
                        await img_file.write(content)
                    
                    print(f"Downloaded {img_url} to {save_path}")
                    return True
        except Exception as e:
            print(f"Error downloading {img_url}: {e}")
        return False

    async def download_lego_images(self, directory: str) -> None:
        """Download all LEGO images asynchronously"""
        json_file_path = Path(directory) / f"{self.manual_id}.json"
        if not json_file_path.exists():
            print(f"JSON file not found: {json_file_path}")
            return
        
        async with aiofiles.open(json_file_path, 'r', encoding='utf-8') as f:
            content = await f.read()
            json_data = json.loads(content)
            
        instructions = json_data.get("instructions", [])
        
        # Collect all image URLs
        image_urls = []
        for instruction in instructions:
            assembly_img = instruction.get('assembly_img')
            if assembly_img and assembly_img != 'No image found':
                image_urls.append(assembly_img)
            
            parts_img = instruction.get('parts_img')
            if parts_img and parts_img != 'No image found':
                image_urls.append(parts_img)
        
        # Download images concurrently
        if image_urls:
            connector = aiohttp.TCPConnector(limit=self.max_concurrent_downloads)
            async with aiohttp.ClientSession(connector=connector) as session:
                semaphore = asyncio.Semaphore(self.max_concurrent_downloads)
                
                async def download_with_semaphore(url):
                    async with semaphore:
                        return await self.download_single_image(session, url, directory)
                
                tasks = [download_with_semaphore(url) for url in image_urls]
                results = await asyncio.gather(*tasks)
                
                successful = sum(1 for r in results if r)
                print(f"Downloaded {successful}/{len(image_urls)} images successfully")

    def extract_step_number(self, img_filename: str) -> Optional[int]:
        """Extract step number from image filename"""
        # Look for pattern like "0000_step" or "0006_step"
        match = re.search(r'(\d{4})_(?:step|eop)', img_filename)
        if match:
            return int(match.group(1))
        return None

    def is_parts_image(self, img_filename: str) -> bool:
        """Determine if image is a parts image based on filename"""
        return '_eop_' in img_filename

    def is_assembly_image(self, img_filename: str) -> bool:
        """Determine if image is an assembly image based on filename"""
        return '_step_' in img_filename

    def determine_step_type(self, step_data: Dict, original_step_number: int, 
                          assigned_step_number: int) -> str:
        """Determine the step type based on the step data and context"""
        # Check for "beg" type: step was reassigned due to gap
        if original_step_number != assigned_step_number:
            return "beg"
        
        # Check for "end" type: no parts_img (regardless of instruction count)
        has_parts_img = 'parts_img' in step_data and step_data['parts_img']
        
        if not has_parts_img:
            return "end"
        
        # Default to "assembly"
        return "assembly"

    async def process_html_file(self, html_file_path: str) -> Optional[Dict[str, Any]]:
        """Process HTML file for LEGO instructions with updated JSON structure"""
        async with aiofiles.open(html_file_path, 'r', encoding='utf-8') as file:
            html_content = await file.read()

        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Look for instruction rows
        rows = (soup.find_all(class_='row') or 
                soup.find_all('div', class_='instruction-step') or 
                soup.find_all('div', class_='step'))
        
        if not rows:
            # Try finding by other common patterns
            rows = soup.find_all('div', recursive=True)
            rows = [row for row in rows if row.find('img') and row.get_text().strip()]
        
        # Group instructions by step
        step_groups = {}
        
        for row in rows:
            # Extract image
            img = row.find('img')
            if not img or not img.has_attr('src'):
                continue
            
            img_src = img['src']
            
            # Handle relative URLs
            if img_src.startswith('./'):
                img_src = img_src.replace("./", f"{self.base_url}/{self.set_number}/")
            
            # URL encode special characters
            img_src = img_src.replace("#", "%23").replace(" ", "%20")
            
            # Extract step number from image filename
            img_filename = os.path.basename(img_src.split('?')[0])
            step_number = self.extract_step_number(img_filename)
            
            if step_number is None:
                # Handle special case for start image
                if 'Start' in img_src:
                    step_number = 0
                else:
                    continue
            
            # Initialize step group if not exists
            if step_number not in step_groups:
                step_groups[step_number] = {
                    'step': step_number,
                    'instructions': [],
                    'assembly_img': '',
                    'parts_img': ''
                }
            
            # Extract text content
            text_content = row.get_text().strip()
            if text_content:
                step_groups[step_number]['instructions'].append(text_content)
            
            # Set local image path
            img_filename_clean = os.path.basename(img_src.split('?')[0])
            if not img_filename_clean.lower().endswith(('.jpg', '.jpeg', '.png', '.gif')):
                img_filename_clean += '.jpg'
            
            local_img_path = f'{self.manual_id}/images/{img_filename_clean}'
            
            # Categorize image based on filename pattern
            if self.is_parts_image(img_filename):
                step_groups[step_number]['parts_img'] = local_img_path
            elif self.is_assembly_image(img_filename) or 'Start' in img_src:
                step_groups[step_number]['assembly_img'] = local_img_path

        # Convert to list format and add step_type with reassigned step numbers
        instructions_list = []
        assigned_step = 0
        
        for original_step_num in sorted(step_groups.keys()):
            step_data = step_groups[original_step_num]
            
            # Assign sequential step numbers
            step_data['step'] = assigned_step
            
            # Determine step type based on whether step was reassigned
            step_type = self.determine_step_type(step_data, original_step_num, assigned_step)
            step_data['step_type'] = step_type
            
            # Log reassigned steps
            if original_step_num != assigned_step:
                print(f"Step {original_step_num} reassigned to {assigned_step} and marked as 'beg'")
            
            # Remove empty parts_img if not set
            if not step_data['parts_img']:
                del step_data['parts_img']
            
            instructions_list.append(step_data)
            assigned_step += 1

        # Create final JSON structure
        return {
            "manual_id": self.manual_id,
            "manual_type": "lego",
            "set_number": self.set_number,
            "set_name": self.set_name,
            "instructions": instructions_list
        }

    async def process_text_files(self, directory: str) -> Optional[Dict[str, Any]]:
        """Process text files with updated structure"""
        data = []
        
        # Look for numbered text files
        text_files = []
        for i in range(100):  
            text_file = Path(directory) / f"{self.manual_id}_{i}.txt"
            if text_file.exists():
                text_files.append(text_file)
        
        step_number = 0
        for text_file in text_files:
            async with aiofiles.open(text_file, 'r', encoding='utf-8') as f:
                content = await f.read()
                content = content.strip()
                
                if content:
                    step_data = {
                        'step': step_number,
                        'step_type': 'assembly',
                        'instructions': [content],
                        'assembly_img': f'{self.manual_id}/images/',
                    }
                    
                    data.append(step_data)
                    step_number += 1
        
        if data:
            return {
                "manual_id": self.manual_id,
                "manual_type": "lego",
                "set_number": self.set_number,
                "set_name": self.set_name,
                "instructions": data
            }
        
        return None

    async def process_existing_json(self, json_file_path: str) -> Optional[Dict[str, Any]]:
        """Process existing JSON file and convert to new structure"""
        async with aiofiles.open(json_file_path, 'r', encoding='utf-8') as f:
            content = await f.read()
            original_data = json.loads(content)
        
        # Group instructions by step
        step_groups = {}
        
        for instruction in original_data['instructions']:
            img_path = instruction.get('img', '')
            if not img_path or img_path == 'No image found':
                continue
            
            # Extract step number from image path
            step_number = self.extract_step_number(img_path)
            
            if step_number is None:
                # Handle special case for start image
                if 'Start' in img_path:
                    step_number = 0
                else:
                    continue
            
            # Initialize step group if not exists
            if step_number not in step_groups:
                step_groups[step_number] = {
                    'step': step_number,
                    'instructions': [],
                    'assembly_img': '',
                    'parts_img': ''
                }
            
            # Add instruction text
            if instruction.get('text') and isinstance(instruction['text'], list):
                for text in instruction['text']:
                    if text.strip():
                        step_groups[step_number]['instructions'].append(text.strip())
            
            # Set image path based on type
            local_img_path = instruction.get('VLM', {}).get('img_path', '')
            
            if self.is_parts_image(img_path):
                step_groups[step_number]['parts_img'] = local_img_path
            elif self.is_assembly_image(img_path) or 'Start' in img_path:
                step_groups[step_number]['assembly_img'] = local_img_path

        # Convert to list format and add step_type with reassigned step numbers
        instructions_list = []
        assigned_step = 0
        
        for original_step_num in sorted(step_groups.keys()):
            step_data = step_groups[original_step_num]
            
            # Assign sequential step numbers
            step_data['step'] = assigned_step
            
            # Determine step type based on whether step was reassigned
            step_type = self.determine_step_type(step_data, original_step_num, assigned_step)
            step_data['step_type'] = step_type
            
            # Log reassigned steps
            if original_step_num != assigned_step:
                print(f"Step {original_step_num} reassigned to {assigned_step} and marked as 'beg'")
            
            # Remove empty parts_img if not set
            if not step_data['parts_img']:
                del step_data['parts_img']
            
            instructions_list.append(step_data)
            assigned_step += 1

        # Create final JSON structure
        return {
            "manual_id": self.manual_id,
            "manual_type": "lego",
            "set_number": self.set_number,
            "set_name": self.set_name,
            "instructions": instructions_list
        }

    async def save_json_data(self, data: Dict[str, Any]) -> str:
        """Save processed JSON data to file"""
        output_folder = Path(self.output_folder)
        output_folder.mkdir(exist_ok=True)
        
        output_file = output_folder / f"{self.manual_id}.json"
        
        async with aiofiles.open(output_file, 'w', encoding='utf-8') as json_file:
            await json_file.write(json.dumps(data, indent=4, ensure_ascii=False))
        
        return str(output_file)

    def print_statistics(self, data: Dict[str, Any]) -> None:
        """Print statistics about processed data"""
        instructions = data['instructions']
        step_types = {}
        
        for step in instructions:
            step_type = step['step_type']
            step_types[step_type] = step_types.get(step_type, 0) + 1
        
        print(f"\nProcessing complete!")
        print(f"Total steps processed: {len(instructions)}")
        print("Step type statistics:")
        for step_type, count in step_types.items():
            print(f"  {step_type}: {count} steps")

    async def process(self) -> Optional[str]:
        """Main processing function"""
        processed_data = None
        
        # Try to find and process the data
        for directory in self.input_directories:
            directory_path = Path(directory)
            if not directory_path.exists():
                continue
                
            print(f"Found directory: {directory}")
            
            # Check for HTML file first
            html_file = directory_path / f"{self.manual_id}.html"
            if html_file.exists():
                print(f"Processing HTML file: {html_file}")
                processed_data = await self.process_html_file(str(html_file))
                break
            
            # Check for existing JSON file
            json_file = directory_path / f"{self.manual_id}.json"
            if json_file.exists():
                print(f"Processing JSON file: {json_file}")
                processed_data = await self.process_existing_json(str(json_file))
                break
            
            # Check for text files
            text_data = await self.process_text_files(str(directory_path))
            if text_data:
                print(f"Processing text files in: {directory}")
                processed_data = text_data
                break
        
        if processed_data:
            # Save the processed JSON
            output_file = await self.save_json_data(processed_data)
            print(f"Successfully created: {output_file}")
            
            # Print statistics
            self.print_statistics(processed_data)
            
            # Download images if JSON was created
            await self.download_lego_images(self.output_folder)
            
            return output_file
        else:
            print(f"No data found for {self.manual_id}")
            print("Please ensure the data exists in one of these directories:")
            for dir_path in self.input_directories:
                print(f"  - {dir_path}")
            print("\nExpected files:")
            print(f"  - {self.manual_id}.html (HTML instruction file)")
            print(f"  - {self.manual_id}.json (existing JSON file)")
            print(f"  - {self.manual_id}_0.txt, {self.manual_id}_1.txt, ... (numbered text files)")
            return None


async def main():
    """Main async function"""
    processor = LegoDatasetProcessor()
    result = await processor.process()
    
    if result:
        print(f"\n✅ Processing completed successfully: {result}")
    else:
        print("\n❌ Processing failed - no data found")


if __name__ == '__main__':
    asyncio.run(main())