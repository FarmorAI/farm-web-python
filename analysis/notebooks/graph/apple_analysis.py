from fastapi import FastAPI
import mysql.connector
import pandas as pd
import numpy as np
from fastapi.middleware.cors import CORSMiddleware

# FastAPI 앱 초기화
app = FastAPI()

# CORS 설정 (React 및 다른 클라이언트에서 API 호출 가능하도록 허용)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 특정 프론트엔드 URL을 지정 가능
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# MySQL에서 데이터 가져오는 함수
def get_apple_data():
    conn = mysql.connector.connect(
        host="192.168.0.4",     # MySQL 서버 주소
        user="farmorai_admin",  # MySQL 사용자명
        password="farmorai12345", # MySQL 비밀번호
        database="farmdb"  # 사용할 데이터베이스 이름
    )
    
    cursor = conn.cursor()
    cursor.execute("SELECT cate3, width, height, weight FROM apple_label_dataset")
    rows = cursor.fetchall()
    
    # DataFrame 생성
    df = pd.DataFrame(rows, columns=['cate3', 'width', 'height', 'weight'])
    
    # 등급을 숫자로 변환
    grade_map = {'특': 2, '상': 1, '보통': 0}
    df['cate3'] = df['cate3'].map(grade_map)

    # 부피 계산
    df["volume"] = (4/3) * np.pi * (df["width"] / 2) ** 2 * (df["height"] / 2)
    
    conn.close()  # 연결 종료
    return df

# FastAPI 엔드포인트: 데이터 제공
@app.get("/data")
def read_data():
    df = get_apple_data()
    return df.to_dict(orient="records")  # JSON 형태로 변환하여 반환

