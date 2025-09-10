import datetime
from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()

class VLMModel(BaseModel):
    name: str
    provider: str

vlm_agent = None

def set_vlm_agent(agent):
    global vlm_agent
    vlm_agent = agent

@router.get("/providers/status")
async def get_provider_status():
    return vlm_agent.agent.get_provider_status()


@router.put("/model/")
async def change_model(vlm_model: VLMModel):
    print(f"Changed model to {vlm_model.name}")
    success = await vlm_agent.agent.switch_vlm(vlm_model.name, vlm_model.provider)
    return {"success": success}  


@router.get("/current_model")
async def get_current_model():
    return {
        "model": vlm_agent.agent.get_current_model(),
        "provider": vlm_agent.agent.get_current_provider(),
        "timestamp": datetime.datetime.now().isoformat()
    }
