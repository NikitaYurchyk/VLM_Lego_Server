import uuid
import datetime
import asyncio
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
import aiofiles
import logging

logger = logging.getLogger(__name__)

router = APIRouter()

UPLOAD_DIR = Path("uploaded_images")
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg"}

UPLOAD_DIR.mkdir(exist_ok=True)

upload_stats = {
    "total_uploads": 0,
    "successful_uploads": 0,
    "failed_uploads": 0,
    "total_size_bytes": 0
}


@router.post("/upload-image")
async def upload_image(
    file: UploadFile = File(...),
    timestamp: Optional[str] = Form(None),
    source: Optional[str] = Form("unknown"),
    image_count: Optional[str] = Form(None),
    client_id: Optional[str] = Form(None)
):
    upload_stats["total_uploads"] += 1
    
    try:
        if not file.filename:
            raise HTTPException(status_code=400, detail="No file provided")
        
        file_extension = Path(file.filename).suffix.lower()
        if file_extension not in ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid file type. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
            )
        
        content = await file.read()
        if len(content) > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=413, 
                detail=f"File too large. Maximum size: {MAX_FILE_SIZE} bytes"
            )
        now = datetime.datetime.now()
        date_str = now.strftime("%Y%m%d")
        time_str = now.strftime("%H%M%S")
        unique_id = str(uuid.uuid4())[:8]
        
        new_filename = f"{date_str}_{time_str}{file_extension}"
        
        date_dir = UPLOAD_DIR
        file_path = date_dir / new_filename
        
        async with aiofiles.open(file_path, 'wb') as f:
            await f.write(content)
        
                
        response_data = {
            "success": True,
            "message": "Image uploaded successfully",
            "file_info": {
                "original_filename": file.filename,
                "saved_filename": new_filename,
                "file_path": str(file_path),
                "file_size_bytes": len(content),
                "upload_timestamp": now.isoformat(),
                "client_timestamp": timestamp,
                "source": source,
                "image_count": image_count
            },
            "upload_id": unique_id
        }
        
        logger.info(f"Successfully uploaded image: {file.filename} -> {new_filename}")
        return JSONResponse(content=response_data, status_code=200)
        
    except HTTPException:
        upload_stats["failed_uploads"] += 1
        raise
    except Exception as e:
        upload_stats["failed_uploads"] += 1
        logger.error(f"Error uploading image: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
