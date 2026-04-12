from fastapi import FastAPI, Request
from env.ration_env import RationGuardEnv

app = FastAPI()
env = RationGuardEnv()

@app.get("/")
def root():
    return {"message": "RationGuardEnv running"}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/reset")
def reset_get(level: str = "easy"):
    obs = env.reset(level=level)
    return {"observation": obs}


@app.post("/reset")
@app.post("/reset/")
async def reset_post(request: Request):
    level = "easy"
    try:
        payload = await request.json()
        if isinstance(payload, dict):
            level = str(payload.get("level", "easy"))
    except Exception:
        # Empty body is valid for reset.
        pass

    obs = env.reset(level=level)
    return {"observation": obs}


@app.post("/step")
@app.post("/step/")
async def step(request: Request):
    action = {}
    try:
        payload = await request.json()
        if isinstance(payload, dict):
            action = payload
    except Exception:
        # Missing/invalid body is forwarded as invalid action to env.step.
        action = {}

    obs, reward, done, info = env.step(action)
    return {
        "observation": obs,
        "reward": reward,
        "done": done,
        "info": info
    }


@app.get("/state")
def state():
    return {"state": env.state()}