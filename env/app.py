from fastapi import FastAPI
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
def reset_post(payload: dict | None = None):
    payload = payload or {}
    level = payload.get("level", "easy")
    obs = env.reset(level=level)
    return {"observation": obs}

@app.post("/step")
def step(action: dict):
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