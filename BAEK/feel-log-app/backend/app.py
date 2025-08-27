# 필요한 모듈 임포트
from flask import Flask, jsonify, request
from flask_cors import CORS
import psycopg2
import psycopg2.extras
import bcrypt
import os
import uuid
import logging
from dotenv import load_dotenv
load_dotenv()  # .env

logging.basicConfig(level=logging.DEBUG)

app = Flask(__name__)
CORS(app)

# 데이터베이스 연결 설정
DB_HOST = os.environ.get('POSTGRES_HOST', 'db')
DB_NAME = os.environ.get('POSTGRES_DB', 'feel_log_db')
DB_USER = os.environ.get('POSTGRES_USER', 'feel_log_user')
DB_PASS = os.environ.get('POSTGRES_PASSWORD', 'feel_log_password')

# 데이터베이스 연결 함수
def get_db_connection():
    try:
        conn = psycopg2.connect(
            host=DB_HOST,
            database=DB_NAME,
            user=DB_USER,
            password=DB_PASS
        )
        psycopg2.extras.register_uuid()
        return conn
    except psycopg2.Error as e:
        logging.error(f"DB 연결 실패: {e}")
        return None

# 회원가입 API
@app.route('/api/signup', methods=['POST'])
def signup():
    data = request.get_json()
    logging.debug(f"POST 데이터: {data}")
    print(f"[DEBUG] 받은 데이터: {data}")  # 디버깅용 로그

    if not data:
        return jsonify({'error': 'JSON 데이터가 없습니다.'}), 400

    nickname = data.get('nickname')
    password = data.get('password')
    reg_type = data.get('reg_type', 'email')
    agree_privacy = data.get('agree_privacy', False)
    agree_alarm = data.get('agree_alarm', False)
    
    if not nickname or not password:
        return jsonify({'error': '닉네임과 비밀번호를 입력하세요.'}), 400

    conn = get_db_connection()
    if not conn:
        return jsonify({'error': '서버 오류가 발생했습니다.'}), 500

    try:
        cursor = conn.cursor()
        
        # 닉네임 중복 확인
        cursor.execute('SELECT user_id FROM user_tbl WHERE user_nickname = %s', (nickname,))
        if cursor.fetchone():
            return jsonify({'error': '이미 사용 중인 닉네임입니다.'}), 400

        # user_tbl에 데이터 삽입
        cursor.execute(
            'INSERT INTO user_tbl (user_nickname, user_reg_type, user_agree_privacy, user_agree_alarm) '
            'VALUES (%s, %s, %s, %s) RETURNING user_id', 
            (nickname, reg_type, agree_privacy, agree_alarm)
        )
        user_id = cursor.fetchone()[0]

        # auth_tbl에 비밀번호 해시 저장
        hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
        cursor.execute(
            'INSERT INTO auth_tbl (user_id, password_hash) VALUES (%s, %s)', 
            (user_id, hashed_password)
        )
        
        conn.commit()
        print(f"[DEBUG] 회원가입 성공: user_id={user_id}")  # 디버깅용 로그
        return jsonify({'message': '회원가입 성공!', 'user_id': str(user_id)}), 201

    except psycopg2.Error as e:
        print(f'[DEBUG] 데이터베이스 오류: {e}')
        if conn:
            conn.rollback()
        return jsonify({'error': '데이터베이스 오류가 발생했습니다.'}), 500
    except Exception as e:
        print(f'[DEBUG] 회원가입 오류: {e}')
        if conn:
            conn.rollback()
        return jsonify({'error': '서버 오류가 발생했습니다.'}), 500
    finally:
        if conn:
            conn.close()            

# 로그인 API
@app.route('/api/login', methods=['POST'])
def login():
    data = request.get_json()
    nickname = data.get('nickname')
    password = data.get('password')

    if not nickname or not password:
        return jsonify({'error': '닉네임과 비밀번호를 입력하세요.'}), 400

    conn = get_db_connection()
    logging.warning(f"db connection 확인: {conn}")
    if not conn:
        return jsonify({'error': '서버 오류가 발생했습니다.'}), 500
    
    try:
        cursor = conn.cursor()
        
        # user_tbl에서 사용자 ID 찾기
        cursor.execute('SELECT user_id FROM user_tbl WHERE user_nickname = %s', (nickname,))
        user = cursor.fetchone()
        if not user:
            return jsonify({'error': '잘못된 닉네임 또는 비밀번호입니다.'}), 401

        user_id = user[0]
        
        # auth_tbl에서 비밀번호 해시 확인
        cursor.execute('SELECT password_hash FROM auth_tbl WHERE user_id = %s', (user_id,))
        auth = cursor.fetchone()
        if not auth:
            return jsonify({'error': '잘못된 닉네임 또는 비밀번호입니다.'}), 401

        # memoryview -> bytes 변환
        stored_hash =  auth[0].tobytes()
        logging.warning(f"stored_hash 확인: {type(stored_hash)}")

        if not bcrypt.checkpw(password.encode('utf-8'), stored_hash):
            return jsonify({'error': '잘못된 닉네임 또는 비밀번호입니다.'}), 401
          
        return jsonify({'message': '로그인 성공!', 'user_id': str(user_id)}), 200

    except Exception as e:
        print(f'로그인 오류: {e}')
        return jsonify({'error': '서버 오류가 발생했습니다.'}), 500
    finally:
        if conn:
            conn.close()

# 유저 전체 조회 API
@app.route('/api/users', methods=['GET'])
def get_users():
    conn = get_db_connection()
    if not conn:
        return jsonify({'error': 'DB 연결 실패'}), 500
    try:
        cursor = conn.cursor()
        cursor.execute('SELECT user_id, user_nickname, user_reg_type, user_create_at FROM user_tbl')
        rows = cursor.fetchall()

        users = []
        for row in rows:
            users.append({
                'user_id': str(row[0]),
                'nickname': row[1],
                'reg_type': row[2],
                'created_at': row[3]
            })

        return jsonify({'users': users}), 200
    except Exception as e:
        print(f'[DEBUG] 유저 조회 오류: {e}')
        return jsonify({'error': '서버 오류'}), 500
    finally:
        conn.close()

# 챗봇 API (chatbot_persona_tbl 기반)
@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.get_json(force=True)
    user_id_str = data.get('user_id')
    chatbot_persona_id_str = data.get('chatbot_persona_id')
    message = data.get('message')

    if not all([user_id_str, chatbot_persona_id_str, message]):
        return jsonify({'error': '필수 정보를 입력하세요: user_id, chatbot_persona_id, message'}), 400

    conn = None
    try:
        user_id = uuid.UUID(user_id_str)
        chatbot_persona_id = uuid.UUID(chatbot_persona_id_str)
        
        conn = get_db_connection()
        if not conn:
            return jsonify({'error': '서버 오류가 발생했습니다.'}), 500

        cursor = conn.cursor()

        # 챗봇 세션 생성 (chatbot_persona_tbl 사용)
        sql = """
            INSERT INTO chat_session_tbl (chat_user_id, chat_chatbot_persona_id)
            VALUES (%s, %s)
            ON CONFLICT (chat_user_id, chat_chatbot_persona_id) DO UPDATE SET chat_session_created_at = NOW()
            RETURNING chat_session_id;
        """
        cursor.execute(sql, (user_id, chatbot_persona_id))
        session_id = cursor.fetchone()[0]

        # 사용자 메시지 저장
        cursor.execute(
            'INSERT INTO message_tbl (message_chat_session_id, message_user_id, message_text) VALUES (%s, %s, %s)',
            (session_id, user_id, message)
        )
        
        # 챗봇 응답 메시지 저장
        response_text = f"챗봇 페르소나 응답: '{message}'에 대한 답변입니다!"
        cursor.execute(
            'INSERT INTO message_tbl (message_chat_session_id, message_user_id, message_text) VALUES (%s, NULL, %s)',
            (session_id, response_text)
        )
        
        conn.commit()
        return jsonify({'message': '대화 저장 성공!', 'response': response_text}), 200
        
    except psycopg2.Error as e:
        print(f'데이터베이스 오류: {e}')
        if conn:
            conn.rollback()
        return jsonify({'error': 'error in DB'}), 500
    except ValueError as e:
        print(f'UUID 변환 오류: {e}')
        return jsonify({'error': 'error in UUID '}), 400
    except Exception as e:
        print(f'알 수 없는 오류: {e}')
        if conn:
            conn.rollback()
        return jsonify({'error': '서버 내부 오류가 발생했습니다.'}), 500
    finally:
        if conn:
            conn.close()

# Flask 서버 실행
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
