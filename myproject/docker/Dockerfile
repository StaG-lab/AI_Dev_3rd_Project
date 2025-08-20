# Python 3.10 slim 버전을 기본 이미지로 사용합니다.
FROM python:3.10-slim

# 컨테이너의 작업 디렉토리를 /app으로 설정합니다.
WORKDIR /app

# uWSGI 설치에 필요한 시스템 라이브러리들을 설치합니다.
RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# 빌드 컨텍스트(myproject)에 있는 requirements.txt 파일을 컨테이너에 복사합니다.
# 이 파일은 docker-compose.yml의 'build context' 설정에 따라 찾습니다.
COPY requirements.txt .

# requirements.txt에 명시된 모든 Python 패키지를 설치합니다.
RUN pip install --no-cache-dir -r requirements.txt

# uWSGI 서버를 실행하는 최종 명령입니다.
# app 폴더는 docker-compose.yml의 volumes 설정으로 컨테이너에 마운트됩니다.
# uwsgi.ini 파일은 마운트된 /app 폴더 안에 있으므로,
# WORKDIR을 기준으로 'uwsgi.ini'를 찾을 수 있습니다.
CMD ["uwsgi", "--ini", "uwsgi.ini"]
