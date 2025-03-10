import mysql.connector  
import pandas as pd  
import seaborn as sns  
import matplotlib.pyplot as plt 


conn = mysql.connector.connect(
    host="192.168.0.4",     # MySQL 서버 주소
    user="farmorai_admin", # MySQL 사용자명
    password="farmorai12345", # MySQL 비밀번호
    database="farmdb"  # 사용할 데이터베이스 이름
)
# 연결 확인
if conn.is_connected():
    print("MySQL 데이터베이스에 성공적으로 연결되었습니다.")

# 연결 종료
conn.close()
