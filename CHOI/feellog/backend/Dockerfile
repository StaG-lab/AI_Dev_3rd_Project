#backend/Dockerfile

# --- 1단계: 빌더(Builder) ---
FROM python:3.9 as builder

# 빌드에 필요한 C 컴파일러(build-essential)와 uv 설치 도구(curl) 설치
RUN apt-get update && apt-get install -y curl build-essential python3-dev

# 최신 버전의 uv 설치
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# uv를 사용해 가상환경 생성
# 이 가상환경은 최종 이미지에 그대로 복사됩니다.
RUN /root/.local/bin/uv venv /opt/venv

# requirements.txt 파일을 복사
COPY requirements.txt .

# uv를 사용하여 가상환경에 패키지 설치 (경로 수정)
# 이 단계에서 C 컴파일러를 사용하여 uWSGI가 성공적으로 빌드됩니다.
RUN /root/.local/bin/uv pip install --no-cache-dir -r requirements.txt --python /opt/venv/bin/python

# --- 2단계: 최종 이미지 ---
FROM python:3.9

# 작업 디렉토리 설정
WORKDIR /home/app

# 1단계(builder)에서 패키지가 모두 설치된 가상환경을 통째로 복사
COPY --from=builder /opt/venv /opt/venv

# 환경 변수 PATH에 가상환경의 bin 디렉토리를 추가하여,
# 'uwsgi', 'python' 등의 명령어를 바로 사용할 수 있도록 설정
ENV PATH="/opt/venv/bin:$PATH"

# 애플리케이션 소스 코드를 복사
COPY . .

# 5000 포트 노출
EXPOSE 5000

# 가상환경에 설치된 uWSGI를 사용하여 애플리케이션 실행
CMD ["uwsgi", "--ini", "uwsgi.ini"]