import uuid
import datetime
import asyncio
from pathlib import Path
from typing import Optional
import mlx_whisper

from fastapi import APIRouter, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
import aiofiles
import logging
from services.vlm_service import ChatHandler

logger = logging.getLogger(__name__)

router = APIRouter()

UPLOAD_DIR = Path("uploaded_audio")
MAX_FILE_SIZE = 50 * 1024 * 1024  
ALLOWED_EXTENSIONS = {".wav", ".mp3", ".m4a", ".ogg", ".flac"}

WHISPER_MODEL = "mlx-community/whisper-base.en-mlx-q4"  

UPLOAD_DIR.mkdir(exist_ok=True)

chat_handler: ChatHandler = None


GLOBAL_CLIENT_ID = "unity_ar_client"

upload_stats = {
    "total_uploads": 0,
    "successful_uploads": 0,
    "failed_uploads": 0,
    "total_size_bytes": 0
}


def set_chat_handler(handler: ChatHandler):
    """Set the chat handler instance"""
    global chat_handler
    chat_handler = handler


async def transcribe_audio_with_whisper(audio_file_path: Path) -> dict:
    """Transcribe audio using MLX Whisper"""
    try:
        if not audio_file_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_file_path}")
        
        logger.info(f"Transcribing audio file: {audio_file_path}")
        
        def run_transcription():
            result = mlx_whisper.transcribe(
                str(audio_file_path), 
                path_or_hf_repo=WHISPER_MODEL
            )
            return result
        

        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, run_transcription)
        
        transcription_text = result.get("text", "").strip()
        
        if transcription_text:
            return {
                "success": True,
                "transcription": transcription_text,
                "language": result.get("language", "en"),
                "duration": len(result.get("segments", [])) * 30.0  # Rough estimate
            }
        else:
            return {
                "success": False,
                "error": "No transcription text generated"
            }
                        
    except Exception as e:
        logger.error(f"Error during MLX Whisper transcription: {str(e)}")
        return {
            "success": False,
            "error": f"MLX Whisper transcription error: {str(e)}"
        }


@router.post("/upload_audio")
async def upload_audio(
    audio: UploadFile = File(...),
    recordingId: Optional[str] = Form(None),
    sampleRate: Optional[str] = Form(None),
    channels: Optional[str] = Form(None),
    userId: Optional[str] = Form(None),
    deviceType: Optional[str] = Form("unknown"),
    sdkVersion: Optional[str] = Form(None),
    hasPassthrough: Optional[str] = Form("false"),
    hasMRContext: Optional[str] = Form("false"),
    timestamp: Optional[str] = Form(None),
    source: Optional[str] = Form("Quest3"),
    clientId: Optional[str] = Form(None)
):
    upload_stats["total_uploads"] += 1
    
    try:
        if not audio.filename:
            raise HTTPException(status_code=400, detail="No audio file provided")
        
        file_extension = Path(audio.filename).suffix.lower()
        if file_extension not in ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid file type. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
            )
        
        content = await audio.read()
        if len(content) > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=413, 
                detail=f"File too large. Maximum size: {MAX_FILE_SIZE} bytes"
            )
        
        upload_stats["total_size_bytes"] += len(content)
        
        now = datetime.datetime.now()
        date_str = now.strftime("%Y%m%d")
        time_str = now.strftime("%H%M%S")
        unique_id = str(uuid.uuid4())[:8]
        
        recording_id = recordingId if recordingId else f"rec_{unique_id}"
        new_filename = f"{date_str}_{time_str}_{recording_id}{file_extension}"
        
        date_dir = UPLOAD_DIR
        file_path = date_dir / new_filename
        
        async with aiofiles.open(file_path, 'wb') as f:
            await f.write(content)
        
        upload_stats["successful_uploads"] += 1
        
        sample_rate_int = int(sampleRate) if sampleRate and sampleRate.isdigit() else None
        channels_int = int(channels) if channels and channels.isdigit() else None
        has_passthrough = hasPassthrough.lower() == "true" if hasPassthrough else False
        has_mr_context = hasMRContext.lower() == "true" if hasMRContext else False
        
        response_data = {
            "success": True,
            "message": "Audio uploaded successfully",
            "file_info": {
                "original_filename": audio.filename,
                "saved_filename": new_filename,
                "file_path": str(file_path),
                "file_size_bytes": len(content),
                "upload_timestamp": now.isoformat(),
                "recording_id": recording_id
            },
            "metadata": {
                "sample_rate": sample_rate_int,
                "channels": channels_int,
                "user_id": userId,
                "device_type": deviceType,
                "sdk_version": sdkVersion,
                "has_passthrough": has_passthrough,
                "has_mr_context": has_mr_context,
                "client_timestamp": timestamp,
                "source": source
            },
            "upload_id": unique_id,
            "transcription_ready": False  # Will be set to True when transcription is complete
        }
        
        logger.info(f"Successfully uploaded audio: {audio.filename} -> {new_filename}")
        logger.info(f"Audio metadata - Sample Rate: {sample_rate_int}Hz, Channels: {channels_int}, Device: {deviceType}")
        
        # Try to transcribe audio using Whisper (optional)
        logger.info("Starting Whisper transcription...")
        transcription_result = await transcribe_audio_with_whisper(file_path)
        
        # Add transcription to response
        response_data["transcription"] = transcription_result
        
        if transcription_result.get("success"):
            response_data["transcription_ready"] = True
            transcription_text = transcription_result.get("transcription", "")
            logger.info(f"Transcription successful: {transcription_text[:100]}...")
            
            # Send transcription to ChatHandler using global client ID
            if chat_handler and transcription_text.strip():
                try:
                    logger.info(f"Sending transcription to ChatHandler for global client: {GLOBAL_CLIENT_ID}")
                    await chat_handler.handle_chat_message(GLOBAL_CLIENT_ID, transcription_text)
                    response_data["chat_processed"] = True
                    response_data["client_id"] = GLOBAL_CLIENT_ID
                except Exception as e:
                    logger.error(f"Error sending to ChatHandler: {str(e)}")
                    response_data["chat_processed"] = False
                    response_data["chat_error"] = str(e)
            else:
                if not chat_handler:
                    logger.warning("ChatHandler not available")
                response_data["chat_processed"] = False
        else:
            logger.warning(f"Transcription failed: {transcription_result.get('error', 'Unknown error')}")
            logger.info("Audio uploaded successfully, but transcription unavailable. Check if LM Studio Whisper is running on port 1234.")
            response_data["transcription_ready"] = False
        
        return JSONResponse(content=response_data, status_code=200)
        
    except HTTPException:
        upload_stats["failed_uploads"] += 1
        raise
    except Exception as e:
        upload_stats["failed_uploads"] += 1
        logger.error(f"Error uploading audio: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.get("/upload-stats")
async def get_upload_stats():
    """Get audio upload statistics"""
    return {
        "stats": upload_stats,
        "upload_directory": str(UPLOAD_DIR.absolute()),
        "max_file_size_mb": MAX_FILE_SIZE / (1024 * 1024),
        "allowed_extensions": list(ALLOWED_EXTENSIONS)
    }


@router.get("/audio-files")
async def list_audio_files():
    """List all uploaded audio files"""
    try:
        audio_files = []
        for file_path in UPLOAD_DIR.glob("*"):
            if file_path.is_file() and file_path.suffix.lower() in ALLOWED_EXTENSIONS:
                stat = file_path.stat()
                audio_files.append({
                    "filename": file_path.name,
                    "size_bytes": stat.st_size,
                    "created": datetime.datetime.fromtimestamp(stat.st_ctime).isoformat(),
                    "modified": datetime.datetime.fromtimestamp(stat.st_mtime).isoformat()
                })
        
        return {
            "success": True,
            "audio_files": sorted(audio_files, key=lambda x: x["created"], reverse=True),
            "total_files": len(audio_files)
        }
    except Exception as e:
        logger.error(f"Error listing audio files: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error listing files: {str(e)}")