import { fileURLToPath, URL } from 'node:url'

import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'
import vueDevTools from 'vite-plugin-vue-devtools'

// https://vite.dev/config/
export default defineConfig({
  plugins: [vue(), vueDevTools()],
  server: {
    host: true, // 0.0.0.0으로 설정하여 외부에서 접근 가능하도록 함
    watch: {
      usePolling: true, // 파일 변경 감지를 위해 폴링 방식 사용
      interval: 1000, // 1초마다 파일 변경 확인 (필요에 따라 조정)
    },
  },
  resolve: {
    alias: {
      '@': fileURLToPath(new URL('./src', import.meta.url)),
    },
  },
})
