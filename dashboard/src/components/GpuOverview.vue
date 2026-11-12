<template>
  <div class="card">
    <h2>GPU Overview</h2>
    <div class="stats-grid">
      <div class="stat-item">
        <div class="stat-value" :class="gpuUtilClass">{{ gpuData.gpu_util_pct }}%</div>
        <div class="stat-label">GPU Util</div>
      </div>
      <div class="stat-item">
        <div class="stat-value" :class="grClockClass">{{ grClockDisplay }}</div>
        <div class="stat-label">Graphics Clock</div>
      </div>
      <div class="stat-item">
        <div class="stat-value" :class="memClockClass">{{ memClockDisplay }}</div>
        <div class="stat-label">VRAM Clock</div>
      </div>
      <div class="stat-item">
        <div class="stat-value" :class="pstateClass">{{ gpuData.pstate || 'N/A' }}</div>
        <div class="stat-label">P-State</div>
      </div>
      <div class="stat-item">
        <div class="stat-value" :class="powerClass">{{ gpuData.power_usage_w }}W</div>
        <div class="stat-label">Power</div>
      </div>
      <div class="stat-item">
        <div class="stat-value">{{ gpuData.power_limit_w }}W</div>
        <div class="stat-label">Power Limit</div>
      </div>
      <div class="stat-item">
        <div class="stat-value" :class="tempClass">{{ gpuData.temperature_c }}°C</div>
        <div class="stat-label">Temp</div>
      </div>
      <div class="stat-item">
        <div class="stat-value">{{ gpuData.total_vram_mib }}</div>
        <div class="stat-label">Total VRAM (MiB)</div>
      </div>
      <div class="stat-item">
        <div class="stat-value" :class="vramClass">{{ gpuData.available_vram_mib }}</div>
        <div class="stat-label">Available VRAM (MiB)</div>
      </div>
      <div class="stat-item">
        <div class="stat-value" :class="dockerVramClass">{{ gpuData.docker_vram_mib }}</div>
        <div class="stat-label">Docker VRAM (MiB)</div>
      </div>
      <div class="stat-item">
        <div class="stat-value" :class="systemVramClass">{{ gpuData.system_vram_mib }}</div>
        <div class="stat-label">System VRAM (MiB)</div>
      </div>
    </div>
    <h2>Containers</h2>
    <div class="table-wrapper">
      <table>
        <thead>
          <tr>
            <th>Container</th>
            <th>VRAM (MiB)</th>
            <th>GPU Util</th>
          </tr>
        </thead>
        <tbody>
          <tr v-for="(container, name) in gpuData.containers" :key="name">
            <td>{{ name }}</td>
            <td :class="getContainerVramClass(container.vram_mib)">{{ container.vram_mib }}</td>
            <td>{{ container.gpu_util_pct }}%</td>
          </tr>
        </tbody>
      </table>
    </div>
  </div>
</template>

<script setup>
import { computed } from 'vue';

const props = defineProps({
  gpuData: {
    type: Object,
    required: true,
  },
});

const gpuUtilClass = computed(() => getMetricClass(props.gpuData.gpu_util_pct, 'gpu_util'));
const powerClass = computed(() => getMetricClass(props.gpuData.power_usage_w, 'power'));
const tempClass = computed(() => getMetricClass(props.gpuData.temperature_c, 'temperature'));
const vramClass = computed(() => {
  const total = props.gpuData.total_vram_mib;
  const available = props.gpuData.available_vram_mib;
  const pct = total > 0 ? ((total - available) / total) * 100 : 0;
  return getMetricClass(pct, 'vram');
});

const grClockDisplay = computed(() => {
  const v = props.gpuData.gr_clock_mhz;
  return v && v > 0 ? `${v} MHz` : 'N/A';
});

const memClockDisplay = computed(() => {
  const v = props.gpuData.mem_clock_mhz;
  return v && v > 0 ? `${v} MHz` : 'N/A';
});

const grClockClass = computed(() => {
  const v = props.gpuData.gr_clock_mhz;
  if (!v || v <= 0) return '';
  if (v >= 1400) return 'gpu-metric-high';
  if (v >= 900) return 'gpu-metric-medium';
  return 'gpu-metric-low';
});

const memClockClass = computed(() => {
  const v = props.gpuData.mem_clock_mhz;
  if (!v || v <= 0) return '';
  if (v >= 7000) return 'gpu-metric-high';
  if (v >= 4000) return 'gpu-metric-medium';
  return 'gpu-metric-low';
});

const pstateClass = computed(() => {
  const ps = props.gpuData.pstate;
  if (!ps) return '';
  const num = parseInt(ps.replace('P', ''), 10);
  if (isNaN(num)) return '';
  if (num <= 1) return 'gpu-metric-high';
  if (num <= 4) return 'gpu-metric-medium';
  return 'gpu-metric-low';
});

const dockerVramClass = computed(() => {
  const total = props.gpuData.total_vram_mib;
  if (!total) return 'gpu-metric-low';
  const pct = (props.gpuData.docker_vram_mib / total) * 100;
  if (pct < 25) return 'gpu-metric-low';
  if (pct <= 60) return 'gpu-metric-medium';
  return 'gpu-metric-high';
});

const systemVramClass = computed(() => {
  const total = props.gpuData.total_vram_mib;
  if (!total) return 'gpu-metric-low';
  const pct = (props.gpuData.system_vram_mib / total) * 100;
  if (pct < 25) return 'gpu-metric-low';
  if (pct <= 60) return 'gpu-metric-medium';
  return 'gpu-metric-high';
});

function getContainerVramClass(vram) {
  const total = props.gpuData.total_vram_mib;
  if (!total) return 'gpu-metric-low';
  const pct = (vram / total) * 100;
  if (pct < 25) return 'gpu-metric-low';
  if (pct <= 60) return 'gpu-metric-medium';
  return 'gpu-metric-high';
}

function getMetricClass(value, type) {
  if (type === 'gpu_util') {
    if (value < 60) return 'gpu-metric-low';
    if (value <= 80) return 'gpu-metric-medium';
    return 'gpu-metric-high';
  }
  if (type === 'power') {
    if (value < 180) return 'gpu-metric-low';
    if (value <= 280) return 'gpu-metric-medium';
    return 'gpu-metric-high';
  }
  if (type === 'temperature') {
    if (value < 60) return 'gpu-metric-low';
    if (value <= 75) return 'gpu-metric-medium';
    return 'gpu-metric-high';
  }
  if (type === 'vram') {
    if (value < 50) return 'gpu-metric-low';
    if (value <= 80) return 'gpu-metric-medium';
    return 'gpu-metric-high';
  }
  return 'gpu-metric-low';
}
</script>
