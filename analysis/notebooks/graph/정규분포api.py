from fastapi import FastAPI
import numpy as np
import pandas as pd
import scipy.stats as stats
import mysql.connector
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/fit_distribution")
def fit_distribution():
    df = get_data()
    
    response_data = []
    for grade in [0, 1, 2]:
        data = df[df["cate3"] == grade]["weight"]
        mu, std = np.mean(data), np.std(data)
        x = np.linspace(mu - 3*std, mu + 3*std, 100)
        y = stats.norm.pdf(x, mu, std)
        
        # ✅ 무게(weight) 기준 상위 25% 값 계산
        threshold_75 = np.percentile(data, 75)
        mean_value = mu  # ✅ 평균값 추가

        response_data.append({
            "grade": grade,
            "mu": mu,
            "std": std,
            "mean": mean_value,  # ✅ 평균값 추가
            "x": x.tolist(),
            "y": y.tolist(),
            "threshold_75": threshold_75  # ✅ 무게 기준 상위 25% 값 추가
        })
    
    return {"distribution": response_data}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=9090)
