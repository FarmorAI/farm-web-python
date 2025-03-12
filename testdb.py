import mysql.connector

DB_CONFIG = {
    "host": "192.168.0.4",
    "user": "farmorai_admin",
    "password": "farmorai12345",
    "database": "farmdb",
    "connect_timeout": 5  # ⏳ 5초 후 타임아웃
}

try:
    print("✅ MySQL 연결 시도 중...")
    conn = mysql.connector.connect(**DB_CONFIG)
    
    if conn.is_connected():
        print("🎉 MySQL 연결 성공!")
        cursor = conn.cursor()
        cursor.execute("SHOW TABLES;")
        tables = cursor.fetchall()
        print("📌 데이터베이스 테이블 목록:", tables)

        cursor.execute("SELECT COUNT(*) FROM apple_label_dataset;")
        row_count = cursor.fetchone()
        print(f"🍏 apple_label_dataset 테이블의 총 데이터 개수: {row_count[0]}")
        
        cursor.close()
    conn.close()
except mysql.connector.Error as e:
    print(f"❌ MySQL 연결 실패! 오류: {e}")
except Exception as e:
    print(f"🚨 알 수 없는 오류 발생: {e}")
