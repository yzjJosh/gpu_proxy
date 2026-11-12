<template>
  <div class="card">
    <h2>Routing Stats</h2>
    <div class="table-wrapper">
      <table>
        <thead>
          <tr>
            <th>Service</th>
            <th>Target</th>
            <th>Backup</th>
            <th>Evictions</th>
          </tr>
        </thead>
        <tbody>
          <tr v-for="(stats, name) in routingStats" :key="name">
            <td>{{ name }}</td>
            <td>{{ stats.target }}</td>
            <td>{{ stats.backup }}</td>
            <td>{{ stats.evictions }}</td>
          </tr>
        </tbody>
      </table>
    </div>
  </div>
</template>

<script setup>
defineProps({
  routingStats: {
    type: Object,
    required: true,
  },
});
</script>
