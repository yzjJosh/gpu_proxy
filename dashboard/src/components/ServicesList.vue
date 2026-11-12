<template>
  <div class="card">
    <h2>Services</h2>
    <div class="table-wrapper desktop-table">
      <table>
        <thead>
          <tr>
            <th>Name</th>
            <th>Priority</th>
            <th>Health</th>
            <th>State</th>
            <th>VRAM (Used / Budget)</th>
            <th>In-Flight</th>
            <th>Target</th>
            <th>Backup</th>
          </tr>
        </thead>
        <tbody>
          <template v-for="(service, name) in servicesData" :key="name">
            <tr>
              <td>
                <template v-if="hasPlugins(name)">
                  <button class="service-toggle" @click="toggleService(name)">
                    {{ expandedServices.has(name) ? '▼' : '▶' }}
                  </button>
                </template>
                {{ name }}{{ service.external ? ' (External)' : '' }}
              </td>
              <td>{{ service.external ? '-' : service.priority }}</td>
              <td :class="service.is_healthy ? 'health-healthy' : 'health-unhealthy'">
                {{ service.is_healthy ? 'Healthy' : 'Unhealthy' }}
              </td>
              <td :class="service.external ? '' : 'state-' + service.state">{{ service.external ? '-' : service.state }}</td>
              <td :class="service.external ? '' : getVramClass(service.vram_used, service.vram_budget_mib)">{{ service.external ? '-' : (service.vram_used + ' / ' + service.vram_budget_mib + ' MiB') }}</td>
              <td>{{ service.external ? '-' : service.in_flight }}</td>
              <td>{{ service.routing.target }}</td>
              <td>{{ service.routing.backup }}</td>
            </tr>
            <tr v-if="hasPlugins(name) && expandedServices.has(name)" class="plugin-ui-row">
              <td :colspan="8" class="plugin-ui-cell">
                <PluginUI v-for="ui in pluginUis[name]" :key="ui.title" :title="ui.title" :html="ui.html" />
              </td>
            </tr>
          </template>
        </tbody>
      </table>
    </div>
    <div class="mobile-cards">
      <template v-for="(service, name) in servicesData" :key="name">
        <div class="mobile-card">
          <div class="mobile-card-header">
            <template v-if="hasPlugins(name)">
              <button class="service-toggle" @click="toggleService(name)">
                {{ expandedServices.has(name) ? '▼' : '▶' }}
              </button>
            </template>
            {{ name }}{{ service.external ? ' (External)' : '' }}
          </div>
          <div class="mobile-card-row">
            <span class="mobile-card-label">Priority</span>
            <span>{{ service.external ? '-' : service.priority }}</span>
          </div>
          <div class="mobile-card-row">
            <span class="mobile-card-label">Health</span>
            <span :class="service.is_healthy ? 'health-healthy' : 'health-unhealthy'">
              {{ service.is_healthy ? 'Healthy' : 'Unhealthy' }}
            </span>
          </div>
          <div class="mobile-card-row">
            <span class="mobile-card-label">State</span>
            <span :class="service.external ? '' : 'state-' + service.state">{{ service.external ? '-' : service.state }}</span>
          </div>
          <div class="mobile-card-row">
            <span class="mobile-card-label">VRAM</span>
            <span :class="service.external ? '' : getVramClass(service.vram_used, service.vram_budget_mib)">{{ service.external ? '-' : (service.vram_used + ' / ' + service.vram_budget_mib + ' MiB') }}</span>
          </div>
          <div class="mobile-card-row">
            <span class="mobile-card-label">In-Flight</span>
            <span>{{ service.external ? '-' : service.in_flight }}</span>
          </div>
          <div class="mobile-card-row">
            <span class="mobile-card-label">Target/Backup</span>
            <span>{{ service.routing.target }}/{{ service.routing.backup }}</span>
          </div>
        </div>
        <div v-if="hasPlugins(name) && expandedServices.has(name)" class="mobile-plugin-section">
          <PluginUI v-for="ui in pluginUis[name]" :key="ui.title" :title="ui.title" :html="ui.html" />
        </div>
      </template>
    </div>
  </div>
</template>

<script setup>
import { ref } from 'vue';
import PluginUI from './PluginUI.vue';

const props = defineProps({
  servicesData: {
    type: Object,
    required: true,
  },
  pluginUis: {
    type: Object,
    default: () => ({}),
  },
});

const expandedServices = ref(new Set());

function hasPlugins(name) {
  const list = props.pluginUis?.[name];
  return Array.isArray(list) && list.length > 0;
}

function toggleService(name) {
  if (expandedServices.value.has(name)) {
    expandedServices.value.delete(name);
  } else {
    expandedServices.value.add(name);
  }
}

function getVramClass(used, budget) {
  if (!budget) return 'gpu-metric-low';
  const pct = (used / budget) * 100;
  if (pct < 50) return 'gpu-metric-low';
  if (pct <= 80) return 'gpu-metric-medium';
  return 'gpu-metric-high';
}
</script>

<style scoped>
.service-toggle {
  display: inline-block;
  background: transparent;
  border: none;
  cursor: pointer;
  font-size: 10px;
  padding: 0;
  margin-right: 4px;
  color: #888;
  line-height: 1;
}
.service-toggle:hover {
  color: #333;
}
.plugin-ui-row td {
  background: #fafafa;
  padding: 8px 12px;
}
.mobile-plugin-section {
  padding: 0 12px 12px;
}
</style>
