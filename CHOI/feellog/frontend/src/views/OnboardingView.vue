<script setup>
import { onMounted } from 'vue';
import { useRouter } from 'vue-router';
const router = useRouter();
import { ref, computed } from 'vue';
import { VideoIcon, BotIcon, TrendingUpIcon } from 'lucide-vue-next';

const currentSlide = ref(0);
const totalSlides = 3; // Assuming 3 slides based on the HTML

const buttonText = computed(() => {
  return currentSlide.value === totalSlides - 1 ? '시작하기' : '다음';
});

const showSlide = (index) => {
  currentSlide.value = index;
};

const nextSlide = () => {
  if (currentSlide.value < totalSlides - 1) {
    currentSlide.value++;
  } else {
    router.push({ name: 'login' });
  }
};

// The second script block in the original HTML contained global navigation logic.
// In a Vue.js application, this logic would typically be handled by Vue Router
// or by emitting custom events to a parent component for navigation.
// For now, it's omitted as it's outside the scope of a single component conversion.
// If needed, it should be integrated into the main application routing.
</script>
<template>
  <div class="container">
    <main class="screen flex flex-col items-center justify-center text-center px-8">
      <div class="w-full flex-1 flex items-center justify-center" id="slides">
        <div :class="{ 'hidden': currentSlide !== 0 }" class="slide space-y-6">
          <VideoIcon class="w-24 h-24 text-indigo-500 mx-auto" />
          <h2 class="text-3xl font-extrabold text-slate-900">영상으로 기록</h2>
          <p class="text-gray-600 leading-relaxed">하루의 감정을 영상으로 쉽고<br/>편리하게 기록해보세요.</p>
        </div>
        <div :class="{ 'hidden': currentSlide !== 1 }" class="slide space-y-6">
          <BotIcon class="w-24 h-24 text-indigo-500 mx-auto" />
          <h2 class="text-3xl font-extrabold text-slate-900">AI 감정 분석</h2>
          <p class="text-gray-600 leading-relaxed">AI 챗봇이 당신의 영상을 분석하고<br/>감정 리포트를 제공해줘요.</p>
        </div>
        <div :class="{ 'hidden': currentSlide !== 2 }" class="slide space-y-6">
          <TrendingUpIcon class="w-24 h-24 text-indigo-500 mx-auto" />
          <h2 class="text-3xl font-extrabold text-slate-900">감정 트렌드 확인</h2>
          <p class="text-gray-600 leading-relaxed">캘린더와 차트를 통해<br/>감정의 변화를 한눈에 파악하세요.</p>
        </div>
      </div>
      <div class="flex items-center gap-2 my-6" id="dots">
        <div
          v-for="(dot, index) in totalSlides"
          :key="index"
          :class="{ 'active': currentSlide === index }"
          class="dot"
          @click="showSlide(index)"
        ></div>
      </div>
      <div class="w-full px-6 pb-10">
        <button
          class="w-full py-4 rounded-2xl bg-indigo-600 text-white text-lg font-semibold shadow-lg shadow-indigo-500/20"
          @click="nextSlide"
        >
          {{ buttonText }}
        </button>
      </div>
    </main>
  </div>
</template>

<style scoped>
/* Tailwind CSS is used, so most styling is via classes. */
/* Custom styles from the original <style> tag */
.container {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  height: 100vh; /* 화면 전체 높이 */
}
.screen {
  display: flex !important; /* Force display for this initial screen */
}
.dot {
  width: 10px;
  height: 10px;
  border-radius: 9999px;
  background: #cbd5e1;
}
.dot.active {
  background: #4f46e5;
}
</style>
