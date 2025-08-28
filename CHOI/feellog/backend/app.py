import os
import datetime
import jwt
from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.dialects.postgresql import UUID
import uuid

# --- App & DB Initialization ---
app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY')
db = SQLAlchemy(app)

# --- Custom Hashing Logic ---
from auth_util import hash_email, hash_password, verify_password

# --- SQLAlchemy Models ---
class User(db.Model):
    __tablename__ = 'user_tbl'
    user_id = db.Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_email_hash = db.Column(db.String(128), unique=True, nullable=False)
    user_nickname = db.Column(db.String(50), unique=True, nullable=False)
    user_profile_image_url = db.Column(db.Text, nullable=True)
    user_agree_privacy = db.Column(db.Boolean, nullable=False)
    user_agree_alarm = db.Column(db.Boolean, nullable=False)
    selected_chatbot_id = db.Column(UUID(as_uuid=True), db.ForeignKey('chatbot_persona_tbl.chatbot_id'), nullable=True)
    user_account_created = db.Column(db.DateTime(timezone=True), server_default=db.func.now())
    user_account_updated = db.Column(db.DateTime(timezone=True), server_default=db.func.now(), onupdate=db.func.now())

class Auth(db.Model):
    __tablename__ = 'auth_tbl'
    auth_id = db.Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_email_hash = db.Column(db.String(128), unique=True, nullable=False)
    password_hash = db.Column(db.LargeBinary, nullable=False)

# --- API Endpoints ---
@app.route('/api/auth/signup', methods=['POST'])
def signup():
    data = request.get_json()
    email = data.get('email')
    password = data.get('password')
    nickname = data.get('nickname')
    agree_privacy = data.get('agree_privacy')
    agree_alarm = data.get('agree_alarm')

    if not all([email, password, nickname, agree_privacy is not None, agree_alarm is not None]):
        return jsonify({"error": "모든 필드를 채워주세요."}), 400

    email_hash = hash_email(email)

    # 이메일 또는 닉네임 중복 확인
    if User.query.filter_by(user_email_hash=email_hash).first() or User.query.filter_by(user_nickname=nickname).first():
        return jsonify({"error": "이미 사용 중인 이메일 또는 닉네임입니다."}), 409

    # 사용자 정보 저장
    new_user = User(
        user_email_hash=email_hash,
        user_nickname=nickname,
        user_agree_privacy=agree_privacy,
        user_agree_alarm=agree_alarm
    )
    db.session.add(new_user)

    # 인증 정보 저장
    password_hashed = hash_password(password, email_hash)
    new_auth = Auth(
        user_email_hash=email_hash,
        password_hash=password_hashed
    )
    db.session.add(new_auth)
    
    db.session.commit()

    return jsonify({"message": "회원가입이 완료되었습니다."}), 201

@app.route('/api/auth/login', methods=['POST'])
def login():
    data = request.get_json()
    email = data.get('email')
    password = data.get('password')

    if not email or not password:
        return jsonify({"error": "이메일과 비밀번호를 입력해주세요."}), 400

    email_hash = hash_email(email)
    auth_info = Auth.query.filter_by(user_email_hash=email_hash).first()

    if not auth_info:
        return jsonify({"error": "등록되지 않은 사용자입니다."}), 404

    if verify_password(password, email_hash, auth_info.password_hash):
        user_info = User.query.filter_by(user_email_hash=email_hash).first()
        token = jwt.encode({
            'user_id': str(user_info.user_id),
            'exp': datetime.datetime.utcnow() + datetime.timedelta(hours=24)
        }, app.config['SECRET_KEY'], algorithm="HS256")
        
        return jsonify({
            "message": "로그인 성공",
            "token": token,
            "nickname": user_info.user_nickname
        }), 200
    else:
        return jsonify({"error": "비밀번호가 일치하지 않습니다."}), 401

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
