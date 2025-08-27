import psycopg2

conn = psycopg2.connect(
    dbname="feel_log_db",
    user="feel_log_user",
    password="feel_log_password",
    host="localhost",
    port="5432"
)
cur = conn.cursor()

# 잘못 들어간 문자열 해시를 바이트로 변환
cur.execute("""
    UPDATE auth_tbl
    SET password_hash = decode(substring(password_hash from 3 for length(password_hash)-3), 'escape')
    WHERE password_hash LIKE 'b''%';
""")

conn.commit()
cur.close()
conn.close()

print("잘못된 password_hash가 바이트 형태로 변환되었습니다.")