import { createRouter, createWebHistory } from 'vue-router';
import IntroView from '../views/IntroView.vue';
import OnboardingView from '../views/OnboardingView.vue';
import LoginView from '../views/LoginView.vue';
//import EmailSignupView from '../views/EmailSignupView.vue';
//import EmailLoginView from '../views/EmailLoginView.vue';
import HomeView from '../views/HomeView.vue';
import RecordView from '../views/RecordView.vue';
//import TrendsView from '../views/TrendsView.vue';
//import ChatbotView from '../views/ChatbotView.vue';
//import ReportView from '../views/ReportView.vue';
//import SettingsView from '../views/SettingsView.vue';
//import SettingPersonaView from '../views/SettingPersonaView.vue';

const routes = [
  { path: '/onboarding', name: 'onboarding', component: OnboardingView },
  { path: '/login', name: 'login', component: LoginView },
//  { path: '/signup/email', name: 'email-signup', component: EmailSignupView },
//  { path: '/login/email', name: 'email-login', component: EmailLoginView },
  { path: '/home', name: 'home', component: HomeView, meta: { showNavbar: true } },
  { path: '/record', name: 'record', component: RecordView },
//  { path: '/trends', name: 'trends', component: TrendsView, meta: { showNavbar: true } },
//  { path: '/chatbot', name: 'chatbot', component: ChatbotView, meta: { showNavbar: true } },
//  { path: '/report', name: 'report', component: ReportView },
//  { path: '/settings', name: 'settings', component: SettingsView, meta: { showNavbar: true } },
//  { path: '/settings/persona', name: 'setting-persona', component: SettingPersonaView }
  { path: '/', name: 'intro', component: IntroView }
];

const router = createRouter({
  history: createWebHistory(),
  routes,
});

// Navigation guards (can be expanded later for auth)
router.beforeEach((to, from, next) => {
    // Example: If not logged in, redirect to login, except for public pages
    // This will be implemented fully in a later step
    next();
});

export default router;
