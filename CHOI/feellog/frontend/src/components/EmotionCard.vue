<!-- /frontend/src/components/EmotionCard.vue -->
<script setup>
const props = defineProps({
  sentimentScore: { type: Number, default: 50 },
  overallEmotionMessage: { type: String, default: '현재 감정은 중립적입니다.' },
  overallEmotionIcon: { type: String, default: '☀️' },
  emotionDistribution: {
    type: Array,
    default: () => [
      { icon: '😐', emotion: '중립', percentage: '100%' },
      { icon: 'N/A', emotion: 'N/A', percentage: '0%' },
      { icon: 'N/A', emotion: 'N/A', percentage: '0%' }
    ]
  }
});
</script>

<template>
  <div class="weather flex min-h-[10em] min-w-[20em] flex-col items-center justify-center gap-[0.5em] rounded-[1.5em] bg-[#FFDAE9] px-4 py-3 font-nunito text-[#F471A6] shadow-[0px_4px_16px_0px_#222]">
      <div class="flex h-fit w-full items-center justify-center gap-4">
          <!-- SVG Icons for weather/overall mood - Placeholder as static, should be dynamic based on overallEmotionIcon -->
          <!-- For now, using the original SVG from card_template_01.html. This will need to be refactored to be dynamic or use simple emojis -->
          <!-- Let's replace the large static SVG with a simple emoji for now to keep the component light -->
          <span class="text-5xl">{{ overallEmotionIcon }}</span>
          <span class="h-16 w-[1px] rounded-full bg-[hsla(336,86%,70%,0.5)]"></span>
          <div class="flex flex-col items-start justify-center">
              <p class="text-xs font-light">현재 당신의 감정온도는?!</p>
              <p class="text-2xl font-semibold">{{ sentimentScore }}°C</p>
              <div class="flex items-center justify-center gap-1">
                  <i data-lucide="clock" class="w-3 h-3 text-[#F471A6]"></i> <!-- Changed to lucide icon for consistency -->
                  <p class="text-[0.625rem] font-light">{{ overallEmotionMessage }}</p>
              </div>
          </div>
      </div>
      <div class="h-[0.5px] w-full rounded-full bg-[hsla(336,86%,70%,0.5)]"></div>
      <div class="flex h-fit w-full items-center justify-between">
          <div class="flex h-fit w-full flex-col items-start justify-between text-xs">
              <div v-for="(item, index) in emotionDistribution" :key="index" class="flex flex-row items-center justify-center gap-2 p-1">
                  <span>{{ item.icon }}</span>
                  <span class="h-2 w-[1px] rounded-full bg-[hsla(336,86%,70%,0.5)]"></span>
                  <p class="w-10">{{ item.emotion }}</p>
                  <span class="h-2 w-[1px] rounded-full bg-[hsla(336,86%,70%,0.5)]"></span>
                  <p>{{ item.percentage }}</p>
              </div>
          </div>
          <div class="flex h-full w-24 flex-col items-center py-1 text-[0.625rem]">
              <div class="flex h-10 w-10 items-center justify-center rounded-full bg-white text-lg shadow-[0px_4px_16px_0px_#222]">
                  {{ overallEmotionIcon }}
              </div>
          </div>
      </div>
  </div>
</template>
