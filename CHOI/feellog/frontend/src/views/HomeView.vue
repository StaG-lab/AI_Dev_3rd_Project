<template>
  <div class="container">
    <main class="screen">
      <header class="px pt">
        <div class="brand">Feel-Log</div>
        <div class="date">{{ currentDate }}</div>
        <p class="subtitle">오늘 하루도 기록하며 자신을 알아가세요!</p>
      </header>

      <div class="cta-wrap">
        <button @click="goToRecord" class="cta-card" aria-label="오늘의 감정 기록하기">
          <CameraIcon class="cta-icon" aria-hidden="true" />
          <div class="cta-text">오늘의 감정 기록하기</div>
        </button>
      </div>

      <h3 class="section-title">최근 감정 요약</h3>

      <div class="cards-wrapper">
        <button class="scroll-btn left" @click="scrollLeft">&lt;</button>
        <section class="cards" ref="cards" aria-label="최근 감정 요약 카드 목록">
          <article class="card">
            <div class="row">
              <div class="emoji" aria-hidden="true">😊</div>
              <div>
                <p class="card-title">어제는 <span class="emphasis">긍정 감정</span>이 70%였어요!</p>
                <p class="card-desc">가장 많이 나타난 감정은 '행복'이네요!</p>
              </div>
            </div>
          </article>
          <article class="card">
            <div class="row">
              <div class="emoji" aria-hidden="true">😌</div>
              <div>
                <p class="card-title">일주일 평균은 <span class="emphasis">안정</span> 상태</p>
                <p class="card-desc">피크 스트레스 시간대는 오후 3시</p>
              </div>
            </div>
          </article>
        </section>
        <button class="scroll-btn right" @click="scrollRight">&gt;</button>
      </div>

    </main>
  </div>
</template>

<script>
export default {
  name: 'home',
  data() {
    return {
      currentDate: ''
    };
  },
  mounted() {
    const today = new Date();
    const year = today.getFullYear();
    const month = today.getMonth() + 1;
    const day = today.getDate();
    const weekday = today.toLocaleDateString('ko-KR', { weekday: 'long' });
    this.currentDate = `${year}년 ${month}월 ${day}일 ${weekday}`;
  },
  methods: {
    scrollLeft() {
      const container = this.$refs.cards;
      const scrollAmount = 316; // 300px card + 16px gap
      container.scrollTo({
        left: container.scrollLeft - scrollAmount,
        behavior: 'smooth'
      });
    },
    scrollRight() {
      const container = this.$refs.cards;
      const scrollAmount = 316;
      container.scrollTo({
        left: container.scrollLeft + scrollAmount,
        behavior: 'smooth'
      });
    }
  }
}
</script>

<script setup>
import { useRouter } from 'vue-router';
import { CameraIcon } from 'lucide-vue-next';
//import api from '../services/api'; // api.js 모듈 임포트
const router = useRouter();
const goToRecord = () => {
  router.push('/record');
};
</script>

<style scoped>
:root{
  --indigo:#4f46e5;
  --indigo-700:#4338ca;
  --text:#0f172a;
  --muted:#64748b;
  --card:#ffffff;
  --shell:#0b1220;
  --bg:#f1f5f9;
}
*{box-sizing:border-box}
body{
  margin:0;
  background:var(--bg);
  font-family: 'Inter','Noto Sans KR', ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, 'Apple SD Gothic Neo', 'Noto Sans KR', sans-serif;
  color:var(--text);
}
.container {
  display: flex;
  flex-direction: column;
  height: 100vh; /* 화면 전체 높이 */
}
.screen{
  background:#fff;
  border-radius:10px;
  overflow:hidden;
  box-shadow:0 10px 30px rgba(2,6,23,.06);
  display:flex;
  flex-direction:column;
}
/* content paddings */
.px{padding-left:24px; padding-right:24px;}
.pt{padding-top:24px;}
.pb{padding-bottom:16px;}
.brand{
  font-family:'Montserrat', sans-serif; font-weight:700; color:var(--indigo);
  font-size:28px; letter-spacing:.3px;
}
.date{ color:var(--muted); margin-top:8px; font-size:20px; }
.subtitle{ margin-top:8px; font-size:18px; line-height:1.6; color:#1f2937; }
/* Big CTA card */
.cta-wrap{ margin-top:24px; padding:0 24px; }
.cta-card{
  width:100%; height:280px; border-radius:28px; background:#4f46e5;
  display:flex; flex-direction:column; align-items:center; justify-content:center; gap:14px;
  color:#ffffff; text-decoration:none; box-shadow: 0 18px 30px rgba(79,70,229,.32);
}
.cta-icon{ width:120px; height:120px; display:block; }
.cta-text{ font-size:20px; font-weight:700; }
/* Section title */
.section-title{ font-weight:700; color:#1f2937; margin-top:28px; margin-bottom:12px; padding:0 24px; }
/* Horizontal cards */
.cards-wrapper {
  position: relative;
  margin: 0 8px; /* Optional adjustment for button placement */
}
.cards{
  padding:0 16px 8px 16px;
  overflow-x: scroll;
  display:flex; gap:16px; scroll-snap-type:x mandatory;
  scrollbar-width: none; /* Firefox */
  -ms-overflow-style: none; /* IE and Edge */
}
.cards::-webkit-scrollbar {
  display: none; /* Chrome, Safari, Opera */
}
.card{
  min-width:300px; max-width:300px; background:var(--card); border-radius:18px;
  box-shadow:0 8px 24px rgba(2,6,23,.08);
  padding:20px; scroll-snap-align:start;
}
.card .row{ display:flex; gap:14px; }
.emoji{ font-size:32px; line-height:1; }
.card-title{ margin:0 0 4px 0; font-weight:700; }
.card-desc{ margin:0; color:#6b7280; }
.emphasis{ color:#16a34a; font-weight:700; }
/* Scroll buttons */
.scroll-btn {
  position: absolute;
  top: 80%;
  transform: translateY(-50%);
  background: white;
  border: 1px solid #ccc;
  border-radius: 50%;
  width: 32px;
  height: 32px;
  display: flex;
  align-items: center;
  justify-content: center;
  cursor: pointer;
  z-index: 10;
  font-size: 18px;
  color: #333;
}
.left {
  left: 16px;
}
.right {
  right: 16px;
}
.sr-only{ position:absolute; width:1px; height:1px; padding:0; margin:-1px; overflow:hidden; clip:rect(0,0,0,0); border:0;}
</style>
