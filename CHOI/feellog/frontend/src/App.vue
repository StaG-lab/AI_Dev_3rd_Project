<!-- app/frontend/src/App.vue -->
<script setup>
import { useRoute } from 'vue-router';
import MobileLayout from './components/MobileLayout.vue';
</script>

<template>
  <MobileLayout :show-navbar="$route.meta.showNavbar">
    <router-view />
  </MobileLayout>
</template>

<style>
/* Global styles or specific app styles if needed */
body {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}
</style>
