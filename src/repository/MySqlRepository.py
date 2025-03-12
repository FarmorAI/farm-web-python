from fastapi import HTTPException
import pandas as pd
import mysql.connector
import os
from dotenv import load_dotenv
load_dotenv()
class MySqlRepository:
    def __init__(self):
        self.pool = mysql.connector.pooling.MySQLConnectionPool(
            pool_name="mypool",
            pool_size=10,
            host = os.getenv("DB_HOST"),
            user = os.getenv("DB_USER"),
            password = os.getenv("DB_PASSWORD"),
            database = os.getenv("DB_NAME")
        )
    def get_data(self):
        try:
            conn = self.pool.get_connection()
            cursor = conn.cursor(dictionary=True)
            query = "SELECT cate3, weight FROM apple_label_dataset"
            cursor.execute(query)
            data = cursor.fetchall()
            cursor.close()
            conn.close()
            df = pd.DataFrame(data)
            if df.empty:
                raise ValueError("No data retrieved from database")
            grade_map = {"특": 2, "상": 1, "보통": 0}
            df["cate3"] = df["cate3"].map(grade_map)
            return df
        except Exception as e:
            print(f"Database error: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Database error: {str(e)}"
            )
# 싱글톤
mysql_repository = MySqlRepository()