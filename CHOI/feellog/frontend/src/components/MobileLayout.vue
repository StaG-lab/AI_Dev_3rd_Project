<!-- frontend/src/components/MobileLayout.vue -->
<script setup>
import { ref, watch } from 'vue';
import { useRouter, useRoute } from 'vue-router';
import { HomeIcon, ChartLineIcon, BotIcon, SettingsIcon } from 'lucide-vue-next';

const props = defineProps({
  showNavbar: {
    type: Boolean,
    default: false
  }
});

const router = useRouter();
const route = useRoute();
const currentActiveLink = ref(route.name);

watch(
  () => route.name,
  (newName) => {
    currentActiveLink.value = newName;
  }
);

const navigate = (routeName) => {
  router.push({ name: routeName });
};
</script>

<template>
  <div class="mobile-container">
    <main class="flex-grow overflow-hidden flex flex-col">
      <slot></slot>
    </main>

    <nav v-if="showNavbar" class="absolute bottom-0 left-0 right-0 h-20 bg-white/80 backdrop-blur-sm border-t border-gray-200 flex justify-around items-center">
        <button
            :class="['flex flex-col items-center nav-link', currentActiveLink === 'home' ? 'text-indigo-600' : 'text-gray-500']"
            @click="navigate('home')"
        >
            <HomeIcon class="cta-icon" aria-hidden="true" />
            <i data-lucide="home"></i><span class="text-xs mt-1">홈</span>
        </button>
        <button
            :class="['flex flex-col items-center nav-link', currentActiveLink === 'trends' ? 'text-indigo-600' : 'text-gray-500']"
            @click="navigate('trends')"
        >
            <ChartLineIcon class="cta-icon" aria-hidden="true" />
            <i data-lucide="trending-up"></i><span class="text-xs mt-1">트렌드</span>
        </button>
        <button
            :class="['flex flex-col items-center nav-link', currentActiveLink === 'chatbot' ? 'text-indigo-600' : 'text-gray-500']"
            @click="navigate('chatbot')"
        >
            <BotIcon class="cta-icon" aria-hidden="true" />
            <i data-lucide="message-circle"></i><span class="text-xs mt-1">챗봇</span>
        </button>
        <button
            :class="['flex flex-col items-center nav-link', currentActiveLink === 'settings' ? 'text-indigo-600' : 'text-gray-500']"
            @click="navigate('settings')"
        >
            <SettingsIcon class="cta-icon" aria-hidden="true" />
            <i data-lucide="settings"></i><span class="text-xs mt-1">설정</span>
        </button>
    </nav>
  </div>
</template>
