<template>
  <button @click="openAgreementModal">가입완료</button>

  <div v-if="showModal" class="modal-overlay">
    <div class="modal-content">
      <h3 class="text-xl font-bold mb-4">약관 동의</h3>
      <div class="space-y-3">
        <label class="flex items-center">
          <input type="checkbox" v-model="agreePrivacy" class="h-5 w-5 rounded border-gray-300 text-indigo-600 focus:ring-indigo-500">
          <span class="ml-2 text-gray-700">[필수] 개인정보 사용 동의</span>
        </label>
        <label class="flex items-center">
          <input type="checkbox" v-model="agreeAlarm" class="h-5 w-5 rounded border-gray-300 text-indigo-600 focus:ring-indigo-500">
          <span class="ml-2 text-gray-700">[선택] 알람 받기 동의</span>
        </label>
      </div>
      <div class="mt-6 flex justify-end space-x-3">
        <button @click="showModal = false">취소</button>
        <button @click="handleSignup" :disabled="!agreePrivacy">동의</button>
      </div>
    </div>
  </div>
</template>

<script setup>
// ... (ref, useRouter, axios 임포트) ...
const handleSignup = async () => {
  if (!agreePrivacy.value) {
    alert('개인정보 사용에 동의해주세요.');
    return;
  }
  try {
    const response = await axios.post('/api/auth/signup', {
      email: email.value,
      password: password.value,
      nickname: nickname.value,
      agree_privacy: agreePrivacy.value,
      agree_alarm: agreeAlarm.value,
    });
    alert(response.data.message);
    router.push('/'); // 회원가입 성공 후 로그인 선택 화면으로
  } catch (error) {
    alert(error.response?.data?.error || '회원가입 중 오류가 발생했습니다.');
  }
};
</script>

<style>
/* ... (모달 스타일) ... */
</style>
