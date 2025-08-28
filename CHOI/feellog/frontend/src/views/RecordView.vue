<template>
  <div class="screen bg-gray-900 text-white p-6 active">
    <header class="flex items-center mb-4">
      <button class="p-2 rounded-full hover:bg-gray-700" @click="goBack">
        <ArrowLeftIcon class="cta-icon" aria-hidden="true" />
      </button>
      <h2 class="text-xl font-semibold mx-auto">감정 기록</h2>
    </header>
    <div class="flex-grow flex flex-col items-center justify-center">
      <div class="w-full aspect-square bg-black rounded-2xl flex items-center justify-center mb-4">
        <video ref="videoElement" class="w-full h-full rounded-2xl" autoplay playsinline></video>
        <p v-if="!isRecording" class="text-gray-400 camera-placeholder">당신의 감정을 기록하세요</p>
      </div>
      <p class="text-lg mb-6">{{ formatTime(timer) }} / {{ formatTime(maxDuration) }}</p>
    </div>
    <div class="text-center pb-8 flex items-center justify-center space-x-4">
      <input type="file" ref="fileInput" @change="handleFileChange" accept="video/*" class="hidden">
      <button
        class="w-16 h-16 bg-gray-500 rounded-full border-4 border-white shadow-lg focus:outline-none focus:ring-4 focus:ring-gray-300 flex items-center justify-center"
        @click="openFilePicker"
      >
        <i data-lucide="upload" class="w-8 h-8 text-white"></i>
      </button>

      <button
        class="w-20 h-20 rounded-full border-4 border-white shadow-lg focus:outline-none focus:ring-4"
        :class="isRecording ? 'bg-red-500 focus:ring-red-300' : 'bg-gray-400 focus:ring-gray-300'"
        @click="toggleRecording"
      >
      </button>

      <button
        v-if="videoFile"
        class="w-16 h-16 bg-indigo-500 rounded-full border-4 border-white shadow-lg focus:outline-none focus:ring-4 focus:ring-indigo-300 flex items-center justify-center"
        @click="analyzeVideo"
      >
        <i data-lucide="flask-conical" class="w-8 h-8 text-white"></i>
      </button>
    </div>
  </div>
</template>

<script>
import { onMounted, ref, onUnmounted } from 'vue';
import { useRouter } from 'vue-router';
import axios from 'axios';
import { ArrowLeftIcon } from 'lucide-vue-next';

export default {
  setup() {
    const router = useRouter();
    const videoElement = ref(null);
    const fileInput = ref(null);
    const isRecording = ref(false);
    const mediaRecorder = ref(null);
    const recordedChunks = ref([]);
    const videoFile = ref(null);
    const timer = ref(0);
    const timerInterval = ref(null);
    const maxDuration = 120; // 2분 제한

    const goBack = () => {
      router.push({ name: 'home' });
    };

    const formatTime = (seconds) => {
      const minutes = Math.floor(seconds / 60);
      const remainingSeconds = Math.floor(seconds % 60);
      return `${String(minutes).padStart(2, '0')}:${String(remainingSeconds).padStart(2, '0')}`;
    };

    const startRecording = () => {
      navigator.mediaDevices.getUserMedia({ video: true, audio: true })
        .then(stream => {
          videoElement.value.srcObject = stream;
          isRecording.value = true;
          recordedChunks.value = [];

          mediaRecorder.value = new MediaRecorder(stream);
          mediaRecorder.value.ondataavailable = (event) => {
            if (event.data.size > 0) {
              recordedChunks.value.push(event.data);
            }
          };

          mediaRecorder.value.onstop = () => {
            const blob = new Blob(recordedChunks.value, { type: 'video/mp4' });
            videoFile.value = new File([blob], `recorded_video_${Date.now()}.mp4`, { type: 'video/mp4' });
            stream.getTracks().forEach(track => track.stop());
            videoElement.value.srcObject = null;
          };

          mediaRecorder.value.start();
          timer.value = 0;
          timerInterval.value = setInterval(() => {
            timer.value += 1;
            if (timer.value >= maxDuration) {
              stopRecording();
            }
          }, 1000);
        })
        .catch(error => {
          console.error("카메라 접근 오류:", error);
          isRecording.value = false;
        });
    };

    const stopRecording = () => {
      if (mediaRecorder.value && isRecording.value) {
        mediaRecorder.value.stop();
        isRecording.value = false;
        clearInterval(timerInterval.value);
      }
    };

    const toggleRecording = () => {
      if (isRecording.value) {
        stopRecording();
      } else {
        startRecording();
      }
    };

    const openFilePicker = () => {
      fileInput.value.click();
    };

    const handleFileChange = (event) => {
      const file = event.target.files[0];
      if (file) {
        videoFile.value = file;
      }
    };

    const analyzeVideo = async () => {
      if (!videoFile.value) {
        alert("분석할 영상 파일이 없습니다.");
        return;
      }

      const formData = new FormData();
      formData.append('video', videoFile.value);

      try {
        const response = await axios.post('http://localhost:5000/api/analyze_video', formData, {
          withCredentials: true,
          headers: {
            'Content-Type': 'multipart/form-data'
          }
        });
        alert(response.data.message);
        router.push({ name: 'chatbot' }); // 분석 요청 후 챗봇 화면으로 이동
      } catch (error) {
        console.error("영상 분석 요청 실패:", error);
        alert("영상 분석 요청에 실패했습니다.");
      }
    };

    onUnmounted(() => {
      clearInterval(timerInterval.value);
      if (videoElement.value && videoElement.value.srcObject) {
        videoElement.value.srcObject.getTracks().forEach(track => track.stop());
      }
    });

    return {
      goBack,
      isRecording,
      toggleRecording,
      videoElement,
      videoFile,
      timer,
      maxDuration,
      formatTime,
      openFilePicker,
      handleFileChange,
      fileInput,
      analyzeVideo
    };
  },
};
</script>
<style scoped>
.camera-placeholder {
  position: absolute;
  top: 50%;
  left: 50%;
  transform: translate(-50%, -50%);
  font-size: 18px;
  color: #aaa;
  text-align: center;
  z-index: 10;
}
.cta-icon{ width:20px; height:20px; color: #ff0202; display:block; }
</style>
