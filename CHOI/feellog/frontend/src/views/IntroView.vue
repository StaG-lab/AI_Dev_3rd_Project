<!-- app/frontend/src/views/IntroView.vue -->
<script setup>
import { onMounted } from 'vue';
import { useRouter } from 'vue-router';

const router = useRouter();

onMounted(() => {
    setTimeout(() => {
        router.push({ name: 'onboarding' });
    }, 3000); // 3초 후 온보딩 화면으로 이동
});
</script>

<template>
    <div id="screen-loading" class="screen active flex-col items-center justify-center bg-indigo-600 p-8">
        <h1 class="font-montserrat text-7xl font-bold text-white text-center">Feel-Log</h1>
        <p class="mt-4 text-xl text-indigo-200 text-center">당신의 감정을 기록하세요</p>
        <p class="mt-12 text-sm text-indigo-300">Team Second</p>
    </div>
</template>

<style scoped>
.screen {
    display: flex !important; /* Force display for this initial screen */
}
</style>
