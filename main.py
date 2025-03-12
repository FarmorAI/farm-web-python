import uvicorn
from fastapi import FastAPI
from src.api.analyze_routes import router as analyze_router
from src.api.distribution_routes import router as distribution_router

app = FastAPI()
app.include_router(analyze_router)
app.include_router(distribution_router)

@app.get("/")
async def root():
    return {"message": "Hello World"}

# uvicorn 실행 환경 설정
if __name__ == "__main__":
    uvicorn.run("main:app", host= "0.0.0.0", port=9090, reload=True)
