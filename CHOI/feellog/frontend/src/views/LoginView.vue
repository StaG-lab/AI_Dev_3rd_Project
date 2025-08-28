<template>
  <div class="p-8 bg-white flex-grow flex flex-col justify-center">
    <h2 class="text-3xl font-bold text-gray-800 mb-2 text-center">환영합니다!</h2>
    <p class="text-gray-500 text-center mb-12">로그인하고 모든 기능을 이용해보세요.</p>

    <div class="space-y-4">
      <button @click="startGoogleLogin" class="w-full flex items-center justify-center bg-gray-100 font-semibold py-3 px-4 rounded-xl hover:bg-gray-200">
        <svg class="w-6 h-6 mr-3" viewBox="0 0 48 48">
          <path fill="#FFC107" d="M43.611,20.083H42V20H24v8h11.303c-1.649,4.657-6.08,8-11.303,8c-6.627,0-12-5.373-12-12s5.373-12,12-12c3.059,0,5.842,1.154,7.961,3.039l5.657-5.657C34.046,6.053,29.268,4,24,4C12.955,4,4,12.955,4,24s8.955,20,20,20s20-8.955,20-20c0-1.341-.138-2.65-.389-3.917z" />
          <path fill="#FF3D00" d="M6.306,14.691l6.571,4.819C14.655,15.108,18.961,12,24,12c3.059,0,5.842,1.154,7.961,3.039l5.657-5.657C34.046,6.053,29.268,4,24,4C16.318,4,9.656,8.337,6.306,14.691z" />
          <path fill="#4CAF50" d="M24,44c5.166,0,9.86-1.977,13.409-5.192l-6.19-5.238C29.211,35.091,26.715,36,24,36c-5.202,0-9.619-3.317-11.283-7.946l-6.522,5.025C9.505,39.556,16.227,44,24,44z" />
          <path fill="#1976D2" d="M43.611,20.083H42V20H24v8h11.303c-.792,2.237-2.231,4.166-4.087,5.571l6.19,5.238C42.022,35.244,44,30.036,44,24c0-1.341-.138-2.65-.389-3.917z" />
        </svg>
        Google 계정으로 시작하기
      </button>
      <button @click="goToEmailLogin" class="w-full bg-gray-800 text-white font-semibold py-3 px-4 rounded-xl hover:bg-gray-900 flex items-center justify-center">
        <MailIcon class="w-5 h-5 mr-3" />
        이메일로 시작하기
      </button>
    </div>

    <div class="mt-8 text-center">
      <button @click="browseAsGuest" class="text-gray-500 hover:text-indigo-600 font-semibold">게스트로 둘러보기</button>
    </div>

    <div class="mt-4 text-center text-sm">
      <a class="text-gray-500 hover:text-indigo-600" href="#">비밀번호 찾기</a>
      <span class="text-gray-300 mx-2">|</span>
      <button @click="goToSignup" class="text-gray-500 hover:text-indigo-600 font-semibold">회원 가입</button>
    </div>
  </div>
</template>



<script setup>
import { useRouter } from 'vue-router';
import { MailIcon } from 'lucide-vue-next';
const router = useRouter();

const goToEmailLogin = () => router.push('/signin');
const goToSignup = () => router.push('/signup');
const browseAsGuest = () => router.push('/home'); // 다음 단계에서 만들 홈으로 이동

const startGoogleLogin = async () => {
  try {
    const response = await api.get('/auth/google/login');
    // 백엔드가 알려준 구글 인증 페이지로 이동
    window.location.href = response.data.redirect_uri;
  } catch (error) {
    console.error("Google login error", error);
    alert('구글 로그인에 실패했습니다.');
  }
};
</script>
