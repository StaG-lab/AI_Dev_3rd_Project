# 필요한 모듈 임포트
from flask import Flask, jsonify, request
from flask_cors import CORS
import psycopg2
import psycopg2.extras
import bcrypt
import os
import uuid
import logging
import json

app = Flask(__name__)
CORS(app)

# 데이터베이스 연결 설정
# 환경 변수 이름을 명확하게 지정하고 기본값을 설정합니다.
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
        # psycopg2가 UUID를 올바르게 처리하도록 설정
        psycopg2.extras.register_uuid()
        return conn
    except psycopg2.Error as e:
        print(f"데이터베이스 연결 실패: {e}")
        return None

# 회원가입 API
@app.route('/api/signup', methods=['POST'])
def signup():
    data = request.get_json()
    data = json.loads(data)
    logging.warning(f"데이터 타입은? {type(data)}, {data}")
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
        return jsonify({'message': '회원가입 성공!', 'user_id': str(user_id)}), 201
    except psycopg2.Error as e:
        print(f'데이터베이스 오류: {e}')
        conn.rollback()
        return jsonify({'error': '데이터베이스 오류가 발생했습니다.'}), 500
    except Exception as e:
        print(f'회원가입 오류: {e}')
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
        if not auth or not bcrypt.checkpw(password.encode('utf-8'), auth[0]):
            return jsonify({'error': '잘못된 닉네임 또는 비밀번호입니다.'}), 401
            
        return jsonify({'message': '로그인 성공!', 'user_id': str(user_id)}), 200
    except Exception as e:
        print(f'로그인 오류: {e}')
        return jsonify({'error': '서버 오류가 발생했습니다.'}), 500
    finally:
        if conn:
            conn.close()

# 챗봇 API
@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.get_json(force=True)
    data = json.loads(data)
    user_id_str = data.get('user_id')
    chatbot_id_str = data.get('chatbot_id')
    message = data.get('message')

    if not all([user_id_str, chatbot_id_str, message]):
        return jsonify({'error': '필수 정보를 입력하세요: user_id, chatbot_id, message'}), 400

    conn = None
    try:
        # UUID 객체로 변환
        user_id = uuid.UUID(user_id_str)
        chatbot_id = uuid.UUID(chatbot_id_str)
        
        conn = get_db_connection()
        if not conn:
            return jsonify({'error': '서버 오류가 발생했습니다.'}), 500

        cursor = conn.cursor()

        # 챗봇 응답 생성 (여기에 실제 AI 모델 연동 로직 추가)
        response_text = f"챗봇: '{message}'에 대한 응답입니다!"
        
        # chat_session_tbl에 기존 세션이 있는지 확인 (사용자-챗봇 조합)
        # INSERT INTO ... ON CONFLICT DO UPDATE 구문을 사용해 세션 중복 생성 방지
        sql = """
            INSERT INTO chat_session_tbl (chat_user_id, chat_chatbot_id)
            VALUES (%s, %s)
            ON CONFLICT (chat_user_id, chat_chatbot_id) DO UPDATE SET chat_session_created_at = NOW()
            RETURNING chat_session_id;
        """
        cursor.execute(sql, (user_id, chatbot_id))
        session_id = cursor.fetchone()[0]

        # message_tbl에 사용자 메시지 저장
        cursor.execute(
            'INSERT INTO message_tbl (message_chat_session_id, message_user_id, message_text) VALUES (%s, %s, %s)',
            (session_id, user_id, message)
        )
        
        # message_tbl에 챗봇 응답 메시지 저장
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
        return jsonify({'error': '데이터베이스 오류가 발생했습니다.'}), 500
    except ValueError as e:
        print(f'UUID 변환 오류: {e}')
        return jsonify({'error': 'UUID 형식이 잘못되었습니다.'}), 400
    except E0