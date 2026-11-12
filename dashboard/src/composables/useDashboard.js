import { ref, onMounted, onUnmounted } from 'vue';

export function useDashboard() {
  const dashboardData = ref(null);
  const connectionStatus = ref('disconnected');
  const lastUpdate = ref(null);
  const reconnectAttempts = ref(0);
  
  let websocket = null;
  let reconnectTimer = null;
  
  const BACKOFF_DELAYS = [1000, 2000, 4000, 8000, 30000];
  
  function getConnectionUrl() {
    const baseUrl = window.location.origin;
    return `${baseUrl}/ws`;
  }
  
  async function fetchInitialData() {
    try {
      const response = await fetch('/dashboard?format=json');
      if (response.ok) {
        dashboardData.value = await response.json();
      }
    } catch (error) {
      console.error('Failed to fetch initial dashboard data:', error);
    }
  }
  
  function connect() {
    if (connectionStatus.value === 'connecting') {
      return;
    }
    
    connectionStatus.value = 'connecting';
    reconnectAttempts.value = 0;
    
    try {
      websocket = new WebSocket(getConnectionUrl());
      
      websocket.onopen = () => {
        connectionStatus.value = 'connected';
        reconnectAttempts.value = 0;
      };
      
      websocket.onmessage = (event) => {
        const message = JSON.parse(event.data);
        if (message.type === 'initial' || message.type === 'update') {
          dashboardData.value = message.data;
          lastUpdate.value = new Date();
        }
      };
      
      websocket.onerror = () => {
        console.error('WebSocket error');
        connectionStatus.value = 'disconnected';
      };
      
      websocket.onclose = () => {
        connectionStatus.value = 'disconnected';
        scheduleReconnect();
      };
    } catch (error) {
      console.error('Failed to create WebSocket:', error);
      connectionStatus.value = 'disconnected';
      scheduleReconnect();
    }
  }
  
  function scheduleReconnect() {
    const delay = BACKOFF_DELAYS[Math.min(reconnectAttempts.value, BACKOFF_DELAYS.length - 1)];
    reconnectAttempts.value++;
    
    reconnectTimer = setTimeout(() => {
      connect();
    }, delay);
  }
  
  function disconnect() {
    if (reconnectTimer) {
      clearTimeout(reconnectTimer);
      reconnectTimer = null;
    }
    if (websocket) {
      websocket.close();
      websocket = null;
    }
  }
  
  onMounted(() => {
    fetchInitialData();
    connect();
  });
  
  onUnmounted(() => {
    disconnect();
  });
  
  return {
    dashboardData,
    connectionStatus,
    lastUpdate,
    reconnectAttempts,
  };
}
