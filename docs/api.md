# Titan Dashboard API Documentation

The Titan Dashboard provides a REST API and WebSocket interface for managing and monitoring the agent swarm.

## Quick Start

Start the dashboard server:

```bash
uvicorn dashboard.app:create_app --factory --host 0.0.0.0 --port 8080
```

Access documentation:
- Swagger UI: http://localhost:8080/docs
- ReDoc: http://localhost:8080/redoc
- OpenAPI JSON: http://localhost:8080/openapi.json

## REST API Endpoints

### System Status

#### GET /api/status

Returns current system health and status.

**Response:**
```json
{
  "status": "healthy",
  "active_agents": 3,
  "current_topology": "swarm",
  "timestamp": "2024-01-19T12:00:00Z"
}
```

### Agents

#### GET /api/agents

List all active agents.

**Response:**
```json
[
  {
    "id": "agent_001",
    "name": "researcher",
    "role": "peer",
    "state": "running",
    "joined_at": "2024-01-19T12:00:00Z",
    "capabilities": ["web_search", "summarization"]
  }
]
```

#### GET /api/agents/{agent_id}

Get details for a specific agent.

**Parameters:**
- `agent_id` (path): Agent identifier

**Response:** Same as single agent in list response.

**Errors:**
- `404`: Agent not found

#### POST /api/agents/{agent_id}/cancel

Cancel a running agent.

**Parameters:**
- `agent_id` (path): Agent identifier

**Response:**
```json
{
  "status": "cancelled",
  "agent_id": "agent_001"
}
```

**Errors:**
- `404`: Agent not found

### Topology

#### GET /api/topology

Get current topology state.

**Response:**
```json
{
  "current": "swarm",
  "agents": [
    {"id": "agent_001", "role": "peer"},
    {"id": "agent_002", "role": "peer"}
  ],
  "history": [
    {
      "from": "pipeline",
      "to": "swarm",
      "timestamp": "2024-01-19T12:00:00Z"
    }
  ]
}
```

#### POST /api/topology/switch/{topology_type}

Switch to a different topology.

**Parameters:**
- `topology_type` (path): One of `swarm`, `hierarchy`, `pipeline`, `mesh`, `ring`, `star`

**Response:**
```json
{
  "status": "success",
  "new_topology": "pipeline",
  "agent_count": 3,
  "duration_ms": 42.5
}
```

**Errors:**
- `400`: Invalid topology type
- `500`: Switch failed

### Events

#### GET /api/events

Get recent event history.

**Parameters:**
- `limit` (query, optional): Maximum events to return (default: 50)

**Response:**
```json
[
  {
    "type": "agent_joined",
    "timestamp": "2024-01-19T12:00:00Z",
    "payload": {
      "agent_id": "agent_001",
      "name": "researcher"
    }
  }
]
```

### Learning

#### GET /api/learning/stats

Get episodic learning statistics.

**Response:**
```json
{
  "total_episodes": 42,
  "topologies": {
    "swarm": {
      "episodes": 20,
      "avg_score": 0.75,
      "success_rate": 0.85
    },
    "pipeline": {
      "episodes": 22,
      "avg_score": 0.82,
      "success_rate": 0.91
    }
  }
}
```

### Metrics

#### GET /api/metrics

Get Prometheus-formatted metrics.

**Response:** Plain text in Prometheus exposition format.

```
# HELP titan_agents_spawned_total Total number of agents spawned
# TYPE titan_agents_spawned_total counter
titan_agents_spawned_total{archetype="researcher"} 15
titan_agents_spawned_total{archetype="coder"} 12
...
```

## WebSocket API

Connect to the WebSocket endpoint for real-time updates:

```
ws://localhost:8080/ws
```

### Protocol

All messages are JSON-encoded. The WebSocket supports bidirectional communication.

### Client Messages

#### ping

Keep-alive ping message.

```json
{"type": "ping"}
```

**Response:**
```json
{"type": "pong"}
```

#### subscribe

Subscribe to specific event topics.

```json
{
  "type": "subscribe",
  "topics": ["agents", "topology"]
}
```

Available topics:
- `agents`: Agent join/leave events
- `topology`: Topology change events
- `tasks`: Task progress updates
- `metrics`: Periodic metric updates

### Server Messages

#### pong

Response to client ping.

```json
{"type": "pong"}
```

#### agent_joined

Notification when an agent joins the swarm.

```json
{
  "type": "agent_joined",
  "agent_id": "agent_001",
  "name": "researcher"
}
```

#### agent_left

Notification when an agent leaves the swarm.

```json
{
  "type": "agent_left",
  "agent_id": "agent_001",
  "reason": "task_completed"
}
```

#### topology_changed

Notification when topology changes.

```json
{
  "type": "topology_changed",
  "old_type": "swarm",
  "new_type": "pipeline"
}
```

### Example: JavaScript Client

```javascript
const ws = new WebSocket('ws://localhost:8080/ws');

ws.onopen = () => {
  console.log('Connected to Titan Dashboard');

  // Subscribe to events
  ws.send(JSON.stringify({
    type: 'subscribe',
    topics: ['agents', 'topology']
  }));

  // Start keepalive
  setInterval(() => {
    ws.send(JSON.stringify({type: 'ping'}));
  }, 30000);
};

ws.onmessage = (event) => {
  const message = JSON.parse(event.data);

  switch (message.type) {
    case 'pong':
      // Keepalive acknowledged
      break;
    case 'agent_joined':
      console.log(`Agent joined: ${message.name} (${message.agent_id})`);
      break;
    case 'agent_left':
      console.log(`Agent left: ${message.agent_id}`);
      break;
    case 'topology_changed':
      console.log(`Topology: ${message.old_type} -> ${message.new_type}`);
      break;
  }
};

ws.onerror = (error) => {
  console.error('WebSocket error:', error);
};

ws.onclose = () => {
  console.log('Disconnected from Titan Dashboard');
};
```

### Example: Python Client

```python
import asyncio
import json
import websockets

async def connect_to_dashboard():
    async with websockets.connect('ws://localhost:8080/ws') as ws:
        # Subscribe to events
        await ws.send(json.dumps({
            'type': 'subscribe',
            'topics': ['agents', 'topology']
        }))

        async for message in ws:
            data = json.loads(message)

            if data['type'] == 'agent_joined':
                print(f"Agent joined: {data['name']}")
            elif data['type'] == 'topology_changed':
                print(f"Topology: {data['old_type']} -> {data['new_type']}")

asyncio.run(connect_to_dashboard())
```

## Error Handling

All errors follow a consistent format:

```json
{
  "detail": "Error message describing what went wrong",
  "code": "ERROR_CODE"
}
```

### Common Error Codes

| Code | HTTP Status | Description |
|------|-------------|-------------|
| `AGENT_NOT_FOUND` | 404 | Specified agent does not exist |
| `INVALID_TOPOLOGY` | 400 | Invalid topology type specified |
| `TOPOLOGY_SWITCH_FAILED` | 500 | Topology switch operation failed |
| `INTERNAL_ERROR` | 500 | Unexpected server error |

## Rate Limiting

The API does not currently implement rate limiting. For production deployments, consider placing a reverse proxy (nginx, traefik) in front of the dashboard.

## Authentication

The dashboard does not currently implement authentication. For production deployments, consider:

1. Adding API key authentication
2. Using OAuth2/OIDC
3. Placing behind an authenticating proxy
