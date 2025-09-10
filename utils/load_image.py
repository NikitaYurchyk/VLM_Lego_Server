import aiofiles
import base64


async def load_image(self, image_path):
    try:
        async with aiofiles.open(image_path, 'rb') as img_file:
            content = await img_file.read()
            return base64.b64encode(content).decode('utf-8')
    except FileNotFoundError:
        print(f"Warning: Image file not found: {image_path}")
        return None