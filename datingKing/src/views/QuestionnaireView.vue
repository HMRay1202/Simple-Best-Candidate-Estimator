<template>
  <div class="questionnaire-view">
    <QuestionnaireForm />
  </div>
</template>

<script setup lang="ts">
import QuestionnaireForm from '@/components/QuestionnaireForm.vue'
</script>

<style scoped>
.questionnaire-view {
  min-height: 100vh;
  background-color: #f5f7fa;
  padding: 20px 0;
}
</style> 