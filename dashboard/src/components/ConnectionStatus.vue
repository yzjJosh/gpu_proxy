<template>
  <div class="connection-status">
    <span class="connection-dot" :class="connectionStatus"></span>
    <span>{{ statusText }}</span>
    <span v-if="lastUpdate">| Last update: {{ formatDate(lastUpdate) }}</span>
  </div>
</template>

<script setup>
import { computed } from 'vue';

const props = defineProps({
  connectionStatus: {
    type: String,
    required: true,
    validator: (v) => ['connected', 'connecting', 'disconnected'].includes(v),
  },
  lastUpdate: {
    type: Date,
    default: null,
  },
});

const statusText = computed(() => {
  switch (props.connectionStatus) {
    case 'connected':
      return 'Connected';
    case 'connecting':
      return 'Connecting...';
    case 'disconnected':
      return 'Disconnected';
    default:
      return 'Unknown';
  }
});

function formatDate(date) {
  if (!date) return '';
  const d = new Date(date);
  const pad = (n) => String(n).padStart(2, '0');
  return `${pad(d.getMonth() + 1)}/${pad(d.getDate())} ${pad(d.getHours())}:${pad(d.getMinutes())}:${pad(d.getSeconds())}`;
}
</script>
