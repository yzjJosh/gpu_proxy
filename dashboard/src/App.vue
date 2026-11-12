<template>
  <div id="app">
    <h1>GPU Priority Proxy</h1>
    
    <ConnectionStatus
      :connection-status="connectionStatus"
      :last-update="lastUpdate"
    />
    
    <ProxyStatus
      v-if="dashboardData"
      :proxy-data="dashboardData.proxy"
      :gpu-data="dashboardData.gpu"
    />
    
    <GpuOverview
      v-if="dashboardData"
      :gpu-data="dashboardData.gpu"
    />

    <ServicesList
      v-if="dashboardData"
      :services-data="servicesData"
      :plugin-uis="pluginUis"
    />
    
    <EvictionLog
      v-if="dashboardData"
      :eviction-log="dashboardData.eviction_log"
      :quiet-eviction-log="dashboardData.quiet_eviction_log || []"
    />
    
    <RoutingStats
      v-if="dashboardData"
      :routing-stats="dashboardData.routing_stats"
    />
  </div>
</template>

<script setup>
import { computed } from 'vue';
import { useDashboard } from './composables/useDashboard';
import ConnectionStatus from './components/ConnectionStatus.vue';
import ProxyStatus from './components/ProxyStatus.vue';
import GpuOverview from './components/GpuOverview.vue';
import ServicesList from './components/ServicesList.vue';
import EvictionLog from './components/EvictionLog.vue';
import RoutingStats from './components/RoutingStats.vue';

const { dashboardData, connectionStatus, lastUpdate } = useDashboard();

const pluginUis = computed(() => {
  if (!dashboardData.value) return {};
  return dashboardData.value.plugin_uis || {};
});

const servicesData = computed(() => {
  if (!dashboardData.value) return {};
  const services = {};
  for (const [name, service] of Object.entries(dashboardData.value.services)) {
    services[name] = {
      ...service,
      vram_used: service.vram_used_mib ?? 0,
    };
  }
  return services;
});
</script>
