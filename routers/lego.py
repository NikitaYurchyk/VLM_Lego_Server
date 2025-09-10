import datetime
from fastapi import APIRouter

router = APIRouter()

vlm_agent = None

def set_vlm_agent(agent):
    global vlm_agent
    vlm_agent = agent

@router.get("/current_step")
async def get_curr_step():
    return {
        "current_step": vlm_agent.agent.current_step,
        "timestamp": datetime.datetime.now().isoformat()
    }
    
@router.get("/increment_step")
async def increment_step():
    vlm_agent.agent.current_step += 1
    return {
        "message": f"Updated current step to {vlm_agent.agent.current_step}",
        "timestamp": datetime.datetime.now().isoformat()
    }


@router.get("/decrement_step")
async def decrement_step():
    vlm_agent.agent.current_step -= 1
    return {
        "message": f"Updated current step to {vlm_agent.agent.current_step}",
        "timestamp": datetime.datetime.now().isoformat()
    }

@router.get("/reset_step")
async def reset_step():
    vlm_agent.agent.current_step = 12
    return {
        "message": "Reset current step to 1",
        "timestamp": datetime.datetime.now().isoformat()
    }