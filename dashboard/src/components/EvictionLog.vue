<template>
  <div class="card">
    <div class="card-header">
      <h2>Eviction Event Log</h2>
      <button class="verbose-toggle" :class="{ active: verbose }" @click="verbose = !verbose">
        <span class="toggle-dot" :class="{ filled: verbose }"></span>
        {{ verbose ? 'Verbose' : 'Quiet' }}
      </button>
    </div>
    <div class="table-wrapper">
      <table>
        <thead>
          <tr>
            <th>Timestamp</th>
            <th>Event</th>
            <th>Service</th>
            <th>Details</th>
          </tr>
        </thead>
        <tbody>
          <tr v-if="recentLogs.length === 0">
            <td colspan="4" style="text-align:center; color:#999; padding:20px;">
              {{ verbose ? 'No events' : 'No significant events' }}
            </td>
          </tr>
          <tr v-else v-for="(entry, index) in recentLogs" :key="index">
            <td>{{ formatTime(entry.timestamp) }}</td>
            <td>{{ entry.event }}</td>
            <td>{{ entry.service || '' }}</td>
            <td>{{ entry.details }}</td>
          </tr>
        </tbody>
      </table>
    </div>
  </div>
</template>

<script setup>
import { computed, ref } from 'vue';

const props = defineProps({
  evictionLog: { type: Array, required: true },
  quietEvictionLog: { type: Array, default: () => [] },
});

const verbose = ref(false);
const LOG_LIMIT = 50;

const recentLogs = computed(() => {
  const source = verbose.value ? props.evictionLog : props.quietEvictionLog;
  return (source || []).slice(-LOG_LIMIT);
});

function formatTime(ts) {
  if (!ts) return '';
  const d = new Date(ts);
  const pad = (n) => String(n).padStart(2, '0');
  return `${pad(d.getMonth() + 1)}/${pad(d.getDate())} ${pad(d.getHours())}:${pad(d.getMinutes())}:${pad(d.getSeconds())}`;
}
</script>

<style scoped>
.card-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 12px;
}

.card-header h2 {
  margin: 0;
}

.verbose-toggle {
  display: flex;
  align-items: center;
  gap: 6px;
  background: none;
  border: 1px solid #d0d0d0;
  border-radius: 16px;
  padding: 4px 12px;
  font-size: 0.8125rem;
  color: #107c10;
  cursor: pointer;
  font-family: inherit;
}

.verbose-toggle:hover {
  background: #f7f7f7;
}

.verbose-toggle.active {
  color: #b8860b;
  border-color: #b8860b;
}

.toggle-dot {
  display: inline-block;
  width: 8px;
  height: 8px;
  border-radius: 50%;
  border: 1.5px solid currentColor;
}

.toggle-dot.filled {
  background: currentColor;
}
</style>
