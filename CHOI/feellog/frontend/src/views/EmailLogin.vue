<template>
  <button @click="handleLogin">로그인</button>
</template>

<script setup>
// ... (ref, useRouter, axios 임포트) ...
const handleLogin = async () => {
  try {
    const response = await axios.post('/api/auth/login', {
      email: email.value,
      password: password.value,
    });
    // 로그인 성공 시 토큰을 localStorage에 저장
    localStorage.setItem('authToken', response.data.token);
    localStorage.setItem('userNickname', response.data.nickname);
    alert(response.data.message);
    router.push('/home'); // 로그인 성공 후 홈으로
  } catch (error) {
    alert(error.response?.data?.error || '로그인 중 오류가 발생했습니다.');
  }
};
</script>
