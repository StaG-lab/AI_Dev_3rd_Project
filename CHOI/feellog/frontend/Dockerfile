# frontend/Dockerfile

# 1. 베이스 이미지 설정
FROM node:22-alpine

# 2. 작업 폴더 지정
WORKDIR /home/frontend

# 3. package.json 파일을 먼저 복사하여 의존성 캐싱 활용
# 'frontend/' 경로를 명시하여 올바른 위치에서 파일을 복사
COPY package*.json .

# 4. 의존성 설치
RUN npm install

# 5. 소스 코드 복사
# 'frontend/' 경로를 명시하여 올바른 소스 코드를 복사
COPY . ./

# 6. 컨테이너 실행 시 Vite 개발 서버 시작
CMD ["npm", "run", "dev", "--", "--host"]