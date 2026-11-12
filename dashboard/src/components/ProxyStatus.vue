<template>
  <div class="card">
    <h2>Proxy Status</h2>
    <div class="stats-grid">
      <div class="stat-item">
        <div class="stat-value">{{ formatUptime(proxyData.uptime_s) }}</div>
        <div class="stat-label">Uptime</div>
      </div>
      <div class="stat-item">
        <div class="stat-value" :class="budgetClass">{{ gpuData.effective_budget_mib }}</div>
        <div class="stat-label">Effective Budget (MiB)</div>
      </div>
    </div>
    <p style="font-size: 0.875rem; color: #999;">
      Last GPU Poll: {{ formatTimestamp(proxyData.last_poll) }}
    </p>
  </div>
</template>

<script setup>
import { computed } from 'vue';

const props = defineProps({
  proxyData: {
    type: Object,
    required: true,
  },
  gpuData: {
    type: Object,
    required: true,
  },
});

const budgetClass = computed(() => {
  const total = props.gpuData.total_vram_mib;
  if (!total) return 'gpu-metric-low';
  const pct = (props.gpuData.effective_budget_mib / total) * 100;
  if (pct >= 50) return 'gpu-metric-low';
  if (pct >= 20) return 'gpu-metric-medium';
  return 'gpu-metric-high';
});

function formatUptime(seconds) {
  if (!seconds && seconds !== 0) return '0s';
  const d = Math.floor(seconds / 86400);
  const h = Math.floor((seconds % 86400) / 3600);
  const m = Math.floor((seconds % 3600) / 60);
  const s = seconds % 60;
  const parts = [];
  if (d > 0) parts.push(d + 'd');
  if (h > 0) parts.push(h + 'h');
  if (m > 0) parts.push(m + 'm');
  parts.push(s + 's');
  return parts.join(' ');
}

function formatTimestamp(ts) {
  if (!ts) return 'N/A';
  const d = new Date(ts);
  const pad = (n) => String(n).padStart(2, '0');
  return `${pad(d.getMonth() + 1)}/${pad(d.getDate())} ${pad(d.getHours())}:${pad(d.getMinutes())}:${pad(d.getSeconds())}`;
}
</script>
