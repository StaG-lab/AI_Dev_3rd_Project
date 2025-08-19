FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY docker/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 이전에 ./app 폴더를 복사한 단계가 성공했으니 그대로 둡니다.

# CMD 명령어를 올바른 파일 이름으로 수정합니다.
CMD ["uwsgi", "--ini", "./app/uwsgi.ini"]

# Flask 앱을 실행하는 커맨드 추가
CMD ["uwsgi", "--ini", "uwsgi.ini"]