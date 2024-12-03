from fastapi import FastAPI
from pydantic import BaseModel, Field, conlist
from main import main as advisor


class RequestBody(BaseModel):
    codes: conlist(str, max_length=300)  # list of str, max length 300
    risk_level: int = Field(..., ge=1, le=5)  # 1 <= risk_level <= 5
    investor_goal: int = Field(..., ge=1, le=4)  # 1 <= investor_goal <= 4


app = FastAPI()


@app.get("/healthcheck")
def read_root():
    return {"message": "Hello, World!"}


@app.post("/advisor")
def execute_roboadvisor(body: RequestBody):
    return advisor(body.codes, body.risk_level, body.investor_goal)
