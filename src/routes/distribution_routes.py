import numpy as np
import scipy.stats as stats
from fastapi import APIRouter
from src.repository.MySqlRepository import mysql_repository

router = APIRouter()

@router.get("/fit_distribution")
async def fit_distribution():
   df = mysql_repository.get_data()
   
   response_data = []
   for grade in [0, 1, 2]:
      data = df[df["cate3"] == grade]["weight"]
      mu, std = np.mean(data), np.std(data)
      x = np.linspace(mu - 3*std, mu + 3*std, 100)
      y = stats.norm.pdf(x, mu, std)
      
      # 무게(weight) 기준 상위 25% 값 계산
      threshold_75 = np.percentile(data, 75)
      mean_value = mu

      response_data.append({
            "grade": grade,
            "mu": mu,
            "std": std,
            "mean": mean_value,   # 평균값 추가
            "x": x.tolist(),
            "y": y.tolist(),
            "threshold_75": threshold_75  # 무게 기준 상위 25% 값 추가
      })
   
   return {"distribution": response_data}