# nginx/Dockerfile
FROM nginx:1.23-alpine

# Nginx 설정 파일은 docker-compose.yml에서 볼륨 마운트되므로 여기서는 복사할 필요 없음
# 프론트엔드 빌드 결과물도 docker-compose.yml에서 볼륨 마운트되므로 여기서는 복사할 필요 없음

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]