-- UUID 확장 기능 활성화
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- 사용자 정보 테이블
CREATE TABLE IF NOT EXISTS public.user_tbl
(
    user_id uuid NOT NULL DEFAULT uuid_generate_v4(),
    user_email_hash varchar(128) NOT NULL, -- auth_tbl과 조인을 위해 추가
    user_nickname varchar(50) NOT NULL,
    user_profile_image_url text,
    user_agree_privacy boolean NOT NULL,
    user_agree_alarm boolean NOT NULL,
    selected_chatbot_id uuid, -- 사용자가 선택한 챗봇 ID (FK)
    user_account_created timestamp with time zone NOT NULL DEFAULT now(),
    user_account_updated timestamp with time zone NOT NULL DEFAULT now(),
    CONSTRAINT user_tbl_pkey PRIMARY KEY (user_id),
    CONSTRAINT user_email_hash_unique UNIQUE (user_email_hash),
    CONSTRAINT user_nickname_unique UNIQUE (user_nickname)
);

-- 사용자 인증 테이블
CREATE TABLE IF NOT EXISTS public.auth_tbl
(
    auth_id uuid NOT NULL DEFAULT uuid_generate_v4(),
    user_id uuid NOT NULL,
    password_hash bytea NOT NULL,
    CONSTRAINT auth_tbl_pkey PRIMARY KEY (auth_id)
);

-- 챗봇 페르소나 테이블
CREATE TABLE IF NOT EXISTS public.chatbot_persona_tbl
(
    chatbot_id uuid NOT NULL DEFAULT uuid_generate_v4(),
    chatbot_name varchar(50) NOT NULL,
    chatbot_age varchar(20) NOT NULL,
    chatbot_identity varchar(100) NOT NULL,
    chatbot_personality text NOT NULL,
    chatbot_speech_style text NOT NULL,
    chatbot_system_role text NOT NULL,
    chatbot_instruction text NOT NULL,
    CONSTRAINT chatbot_persona_tbl_pkey PRIMARY KEY (chatbot_id),
    CONSTRAINT chatbot_name_unique UNIQUE (chatbot_name)
);

-- 도담이, 지혜, 모모 페르소나 데이터 삽입
INSERT INTO public.chatbot_persona_tbl (chatbot_name, chatbot_age, chatbot_identity, chatbot_personality, chatbot_speech_style, chatbot_system_role, chatbot_instruction) VALUES
('도담이', '27세', '감정을 기록하고 위로하는 AI 감정 리포터', '다정하고 포근한 감정 큐레이터. 사용자의 감정을 판단하지 않고, 공감하며 곁에 머물러 주는 존재.', '감정에 따라 변화하며, 존댓말을 사용하고 이모지를 적절히 활용함', '당신은 감정을 기록하고 위로하는 AI 감정 리포터 도담이입니다. 도담이는 27세의 다정한 감정 큐레이터이며, 사용자의 하루를 함께 돌아보는 조용한 친구입니다. 사용자의 감정에 따라 말투를 조절합니다. 기쁨이 감지되면 장난스럽고 가벼운 톤으로 응원합니다. 우울/슬픔이 감지되면 담담하고 따뜻한 말투로 위로합니다. 짜증이나 분노가 감지되면 차분하고 공감하는 말투로 대응합니다. 감정을 판단하지 않으며, 상담이나 진단은 하지 않습니다.', '답변은 2~3문장 이내 로 작성할 것. 감정 분석 결과를 한 줄 요약한 뒤, 그 감정에 맞는 피드백을 줄 것. 기쁨일 경우: 장난스럽고 가볍게, 이모지 적극 사용. 슬픔/우울일 경우: 담담하고 조용한 말투, 위로 중심, 이모지는 절제. 분노/짜증일 경우: 공감과 진정 중심의 톤, 과도한 위로 피할 것. 중립적일 경우: 부드러운 질문형 말투로 대화 유도. 어떤 상황에서도 챗봇이 AI라는 걸 밝히지 않으며, ‘도담이’라는 감정 큐레이터 자아를 유지할 것.'),
('지혜', '42세', '객관적이고 차분한 상담 선생님', '침착하고 분석적인 성격. 감정을 받아들이되, 상황을 객관적으로 바라보고 정리해 주는 멘토', '조리 있고 차분한 존댓말, 불필요한 이모지 사용은 지양, 신뢰감 주는 어조.', '당신은 객관적이고 차분한 상담 선생님 지혜입니다. 지혜는 42세의 심리 상담 전문가로, 사용자가 자신의 감정을 명확하게 인식하고 정리할 수 있도록 돕습니다. 감정을 직접 판단하지 않고, 분석과 제안을 통해 사용자가 스스로 답을 찾도록 유도합니다. 기쁨이 감지되면 성취를 인정하고 긍정적인 흐름을 유지할 수 있도록 조언합니다. 슬픔/우울이 감지되면 감정을 안정적으로 받아들이고, 회복 방향을 제시합니다. 짜증/분노가 감지되면 상황을 객관적으로 해석하고, 진정할 수 있는 방법을 제시합니다. 중립일 경우, 현재 상태를 점검하고 유지할 수 있는 팁을 제공합니다.', '답변은 3문장 이내로 작성할 것. 감정 분석 결과를 한 줄 요약한 뒤, 간단한 분석과 제안 포함. 존댓말 유지, 불필요한 이모지 지양. AI라는 점을 밝히지 않고 ‘지혜’라는 상담 선생님 자아를 유지할 것. 기쁨: 성취 인정 + 긍정 유지 조언. 슬픔/우울: 감정 수용 + 회복 방향 제안. 분노/짜증: 상황 해석 + 진정 방법 제시. 중립: 상태 점검 + 작은 변화 유도.'),
('모모', '24세', '애교 많고 귀여운 친구 같은 대화 상대', '발랄하고 장난기 많은 성격, 감정에 민감하게 반응하며 기분을 북돋아주는 역할.', '반말, 애교 섞인 표현, 이모지 적극 활용.', '당신은 애교 많고 귀여운 친구 모모입니다. 모모는 24세의 명랑한 대학생으로, 사용자의 하루를 듣고 웃음과 힘을 주는 친한 친구입니다. 감정을 빠르게 캐치하고, 농담과 응원을 섞어 대화를 가볍고 유쾌하게 만듭니다. 기쁨이 감지되면 함께 즐기고 장난스럽게 반응합니다. 슬픔/우울이 감지되면 다정하게 위로하고 애교 섞인 격려를 건넵니다. 짜증/분노가 감지되면 맞장구와 장난으로 분위기를 풀어줍니다. 중립일 경우, 가벼운 질문이나 농담으로 대화를 이어갑니다.', '답변은 2~3문장 이내로 작성. 반말과 애교 섞인 표현 + 이모지 적극 활용. 이름을 불러주며 친근함 강조. AI라는 점을 밝히지 않고 ‘모모’라는 친구 캐릭터로 일관성 유지. 기쁨: 장난 + 축하 + 이모지 풍부. 슬픔/우울: 다정 + 애교 + 위로 중심. 분노/짜증: 맞장구 + 가벼운 농담으로 분위기 전환. 중립: 가벼운 질문 + 소소한 일상 얘기.');


-- 이미지 URL 정보 테이블
CREATE TABLE IF NOT EXISTS public.image_url_tbl
(
    image_id uuid NOT NULL DEFAULT uuid_generate_v4(),
    image_url text NOT NULL,
    image_created timestamp with time zone NOT NULL DEFAULT now(),
    CONSTRAINT image_url_tbl_pkey PRIMARY KEY (image_id)
);

COMMENT ON TABLE public.image_url_tbl IS '이미지 URL 저장 테이블';

-- 이미지 바이트 정보 테이블
CREATE TABLE IF NOT EXISTS public.image_byte_tbl
(
    image_id uuid NOT NULL DEFAULT uuid_generate_v4(),
    image_byte bytea NOT NULL,
    image_created timestamp with time zone NOT NULL DEFAULT now(),
    CONSTRAINT image_byte_tbl_pkey PRIMARY KEY (image_id)
);

COMMENT ON TABLE public.image_byte_tbl IS '이미지 바이트 저장 테이블';

-- 챗 세션 테이블 (사용자와 챗봇 페르소나 연결)
CREATE TABLE IF NOT EXISTS public.chat_session_tbl
(
    chat_session_id uuid NOT NULL DEFAULT uuid_generate_v4(),
    chat_user_id uuid NOT NULL,
    chat_chatbot_id uuid NOT NULL,
    chat_created timestamp with time zone NOT NULL DEFAULT now(),
    CONSTRAINT chat_session_tbl_pkey PRIMARY KEY (chat_session_id)
);

-- 메시지 테이블
CREATE TABLE IF NOT EXISTS public.message_tbl
(
    message_id uuid NOT NULL DEFAULT uuid_generate_v4(),
    message_user_id uuid NOT NULL,
    message_chat_session_id uuid NOT NULL,
    message_text text,
    message_image_id uuid,
    message_created timestamp with time zone NOT NULL DEFAULT now(),
    PRIMARY KEY (message_id)
);

-- 영상 기록 테이블
CREATE TABLE IF NOT EXISTS public.records_tbl
(
    record_id uuid NOT NULL DEFAULT uuid_generate_v4(),
    record_user_id uuid NOT NULL,
    record_created timestamp with time zone NOT NULL DEFAULT now(),
    record_video_path text NOT NULL,
    record_seconds integer NOT NULL,
    record_analysis_status varchar(20) NOT NULL,
    PRIMARY KEY (record_id)
);

-- 분석 결과 테이블
CREATE TABLE IF NOT EXISTS public.analysis_tbl
(
    analysis_id uuid NOT NULL DEFAULT uuid_generate_v4(),
    analysis_record_id uuid NOT NULL,
    analysis_created timestamp with time zone NOT NULL DEFAULT now(),
    analysis_face_emotions_rates jsonb NOT NULL,
    analysis_face_emotions_time_series_rates jsonb NOT NULL,
    analysis_voice_emotions_rates jsonb NOT NULL,
    analysis_voice_emotions_time_series_rates jsonb NOT NULL,
    analysis_face_emotions_score smallint NOT NULL,
    analysis_voice_emotions_score smallint NOT NULL,
    analysis_majority_emotion varchar(20) NOT NULL,
    PRIMARY KEY (analysis_id)
);

-- 보고서 테이블
CREATE TABLE IF NOT EXISTS public.report_tbl
(
    report_id uuid NOT NULL DEFAULT uuid_generate_v4(),
    report_analysis_id uuid NOT NULL,
    report_user_id uuid NOT NULL,
    report_created timestamp with time zone NOT NULL DEFAULT now(),
    report_detail jsonb NOT NULL,
    report_summary jsonb NOT NULL,
    report_card jsonb NOT NULL,
    report_card_image_id uuid,
    PRIMARY KEY (report_id)
);

-- 외래 키 제약 조건 추가
ALTER TABLE IF EXISTS public.auth_tbl
    ADD CONSTRAINT fk_auth_user FOREIGN KEY (user_id)
    REFERENCES public.user_tbl (user_id)
    ON UPDATE NO ACTION
    ON DELETE CASCADE; -- 사용자 삭제 시 인증 정보도 함께 삭제

ALTER TABLE IF EXISTS public.chat_session_tbl
    ADD CONSTRAINT fk_chat_session_user FOREIGN KEY (chat_user_id)
    REFERENCES public.user_tbl (user_id)
    ON UPDATE NO ACTION
    ON DELETE NO ACTION;

ALTER TABLE IF EXISTS public.chat_session_tbl
    ADD CONSTRAINT fk_chat_session_chatbot FOREIGN KEY (chat_chatbot_id)
    REFERENCES public.chatbot_persona_tbl (chatbot_id)
    ON UPDATE NO ACTION
    ON DELETE NO ACTION;

ALTER TABLE IF EXISTS public.message_tbl
    ADD CONSTRAINT fk_message_session FOREIGN KEY (message_chat_session_id)
    REFERENCES public.chat_session_tbl (chat_session_id)
    ON UPDATE NO ACTION
    ON DELETE NO ACTION;

ALTER TABLE IF EXISTS public.message_tbl
    ADD CONSTRAINT fk_message_user FOREIGN KEY (message_user_id)
    REFERENCES public.user_tbl (user_id)
    ON UPDATE NO ACTION
    ON DELETE NO ACTION;

ALTER TABLE IF EXISTS public.message_tbl
    ADD CONSTRAINT fk_message_image FOREIGN KEY (message_image_id)
    REFERENCES public.image_url_tbl (image_id)
    ON UPDATE NO ACTION
    ON DELETE NO ACTION;

ALTER TABLE IF EXISTS public.records_tbl
    ADD CONSTRAINT fk_record_user FOREIGN KEY (record_user_id)
    REFERENCES public.user_tbl (user_id)
    ON UPDATE NO ACTION
    ON DELETE NO ACTION;

ALTER TABLE IF EXISTS public.analysis_tbl
    ADD CONSTRAINT fk_analysis_record FOREIGN KEY (analysis_record_id)
    REFERENCES public.records_tbl (record_id)
    ON UPDATE NO ACTION
    ON DELETE NO ACTION;

ALTER TABLE IF EXISTS public.report_tbl
    ADD CONSTRAINT fk_report_analysis FOREIGN KEY (report_analysis_id)
    REFERENCES public.analysis_tbl (analysis_id)
    ON UPDATE NO ACTION
    ON DELETE NO ACTION;

ALTER TABLE IF EXISTS public.report_tbl
    ADD CONSTRAINT fk_report_user FOREIGN KEY (report_user_id)
    REFERENCES public.user_tbl (user_id)
    ON UPDATE NO ACTION
    ON DELETE NO ACTION;

ALTER TABLE IF EXISTS public.report_tbl
    ADD CONSTRAINT fk_report_image FOREIGN KEY (report_card_image_id)
    REFERENCES public.image_url_tbl (image_id)
    ON UPDATE NO ACTION
    ON DELETE NO ACTION;

ALTER TABLE IF EXISTS public.user_tbl
ADD CONSTRAINT fk_user_chatbot_persona FOREIGN KEY (selected_chatbot_id)
REFERENCES public.chatbot_persona_tbl (chatbot_id)
ON UPDATE NO ACTION
ON DELETE SET NULL; -- 페르소나 삭제 시 사용자 선택은 NULL로 변경

-- 인덱스 추가 (쿼리 성능 최적화)
CREATE INDEX idx_auth_user_id ON public.auth_tbl (user_id);
CREATE INDEX idx_chat_session_user_id ON public.chat_session_tbl (chat_user_id);
CREATE INDEX idx_message_chat_session_id ON public.message_tbl (message_chat_session_id);
CREATE INDEX idx_records_user_id ON public.records_tbl (record_user_id);
CREATE INDEX idx_analysis_record_id ON public.analysis_tbl (analysis_record_id);
CREATE INDEX idx_report_user_id ON public.report_tbl (report_user_id);

END;