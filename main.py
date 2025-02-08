from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

app = FastAPI()

class AIRequest(BaseModel) :
    data : list # spring 에서 받은 예제 데이터

class AIResponse(BaseModel) :
    result : str # 예측 결과

@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/hello/{name}")
async def say_hello(name: str):
    return {"message": f"Hello {name}"}




if __name__ == "__main__":

    uvicorn.run(app, host= "127.0.0.1", port=8000)