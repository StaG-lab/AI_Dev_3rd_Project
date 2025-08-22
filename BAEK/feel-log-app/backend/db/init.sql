CREATE TABLE user_tbl (
    user_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_nickname VARCHAR(255) UNIQUE NOT NULL,
    user_reg_type VARCHAR(50) NOT NULL,
    user_agree_privacy BOOLEAN NOT NULL DEFAULT FALSE,
    user_agree_alarm BOOLEAN NOT NULL DEFAULT FALSE,
    user_create_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    user_update_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE auth_tbl (
    user_id UUID REFERENCES user_tbl(user_id) ON DELETE CASCADE,
    password_hash BYTEA NOT NULL,
    PRIMARY KEY (user_id)
);

CREATE TABLE chatbot_persona_tbl (
    chatbot_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    chatbot_name VARCHAR(255) NOT NULL,
    chatbot_description TEXT,
    chatbot_created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE chat_session_tbl (
    chat_session_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    chat_user_id UUID REFERENCES user_tbl(user_id) ON DELETE CASCADE,
    chat_chatbot_id UUID REFERENCES chatbot_persona_tbl(chatbot_id) ON DELETE CASCADE,
    chat_session_created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE message_tbl (
    message_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    message_chat_session_id UUID REFERENCES chat_session_tbl(chat_session_id) ON DELETE CASCADE,
    message_user_id UUID REFERENCES user_tbl(user_id) ON DELETE SET NULL,
    message_text TEXT NOT NULL,
    message_created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- 테스트용 데이터 추가
INSERT INTO user_tbl (user_nickname, user_reg_type) VALUES ('test_user', 'email');
INSERT INTO chatbot_persona_tbl (chatbot_name, chatbot_description) VALUES ('FriendlyBot', 'A friendly chatbot.');