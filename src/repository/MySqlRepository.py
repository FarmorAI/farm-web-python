from fastapi import FastAPI
import numpy as np
import pandas as pd
import scipy.stats as stats
import mysql.connector
from fastapi.middleware.cors import CORSMiddleware

class MySqlRepository:
    def __init__(self, DB_CONFIG):
        self.db_config = DB_CONFIG

    def get_data(self):
        conn = mysql.connector.connect(self.db_config)
        cursor = conn.cursor(dictionary=True)
        
        query = "SELECT cate3, weight FROM apple_label_dataset"
        cursor.execute(query)
        data = cursor.fetchall()

        cursor.close()
        conn.close()

        df = pd.DataFrame(data)
        grade_map = {"특": 2, "상": 1, "보통": 0}
        df["cate3"] = df["cate3"].map(grade_map)

        return df


DB_CONFIG = {
    "host": "192.168.0.4",
    "user": "farmorai_admin",
    "password": "farmorai12345",
    "database": "farmdb"
}

# 싱글톤
mysql_repository = MySqlRepository(DB_CONFIG)