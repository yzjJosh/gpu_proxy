<template>
  <div class="plugin-ui">
    <div class="plugin-ui-title">{{ title }}</div>
    <!-- Plugin HTML is trusted (admin-deployed) -->
    <div class="plugin-ui-content" v-html="html"></div>
  </div>
</template>

<script setup>
defineProps({
  title: { type: String, required: true },
  html: { type: String, required: true },
});
</script>

<style scoped>
.plugin-ui {
  padding: 8px 12px;
  margin-bottom: 8px;
  background: #fafafa;
  border-radius: 4px;
  border: 1px solid #e0e0e0;
}
.plugin-ui-title {
  font-size: 11px;
  font-weight: 600;
  color: #888;
  text-transform: uppercase;
  margin-bottom: 4px;
}
.plugin-ui-content {
  font-size: 13px;
  color: #333;
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 8px;
}
</style>
