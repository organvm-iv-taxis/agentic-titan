# Claude Code Integration Guide

This guide explains how to integrate Agentic Titan with Claude Code or Claude Desktop using the Model Context Protocol (MCP).

## Prerequisites

- Python 3.11 or later
- Claude Code CLI or Claude Desktop application
- Titan package installed with MCP support

### Installation

```bash
# Clone the repository
git clone https://github.com/your-org/agentic-titan.git
cd agentic-titan

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install with all dependencies
pip install -e ".[dashboard,metrics]"
```

## Quick Setup

### For Claude Code CLI

Add the following to your Claude Code configuration file (`~/.claude/config.json` or project-level `.claude/config.json`):

```json
{
  "mcpServers": {
    "titan": {
      "command": "python",
      "args": ["-m", "mcp.server"],
      "cwd": "/path/to/agentic-titan",
      "env": {
        "ANTHROPIC_API_KEY": "${env:ANTHROPIC_API_KEY}"
      }
    }
  }
}
```

### For Claude Desktop

1. Open Claude Desktop settings
2. Navigate to "Developer" > "MCP Servers"
3. Add new server with the configuration above

Or edit the config file directly at:
- macOS: `~/Library/Application Support/Claude/claude_desktop_config.json`
- Windows: `%APPDATA%\Claude\claude_desktop_config.json`
- Linux: `~/.config/Claude/claude_desktop_config.json`

See `examples/claude_desktop_config.json` for a complete example.

## Available MCP Tools

Titan exposes 5 tools through the MCP protocol:

### spawn_agent

Create and run an agent to perform a task.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `agent_type` | string | Yes | Type of agent: `researcher`, `coder`, `reviewer`, `orchestrator` |
| `task` | string | Yes | Task description for the agent |

**Returns:** `{ session_id: string, status: string }`

**Example:**
```
Use the spawn_agent tool to create a researcher agent to investigate best practices for Python async programming.
```

### agent_status

Check the status of a running or completed agent session.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `session_id` | string | Yes | Session ID returned by spawn_agent |

**Returns:** `{ status: string, agent_type: string, error?: string }`

**Example:**
```
Check the status of agent session sess_abc12345.
```

### agent_result

Get the final result from a completed agent session.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `session_id` | string | Yes | Session ID returned by spawn_agent |

**Returns:** `{ result: any, status: string }`

**Example:**
```
Get the result from agent session sess_abc12345.
```

### list_agents

List all active and recent agent sessions.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| (none) | - | - | - |

**Returns:** `{ sessions: Array<{ id, agent_type, status, created_at }> }`

**Example:**
```
List all active Titan agents.
```

### cancel_agent

Cancel a running agent session.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `session_id` | string | Yes | Session ID to cancel |

**Returns:** `{ cancelled: boolean }`

**Example:**
```
Cancel agent session sess_abc12345.
```

## Available MCP Resources

### titan://agents/types

Returns a list of available agent archetypes with their descriptions and capabilities.

### titan://system/status

Returns current system status including active agents and topology information.

## Usage Examples

### Research Task

```
Spawn a researcher agent to find information about Kubernetes autoscaling patterns.
```

Claude will:
1. Call `spawn_agent` with type "researcher" and the task
2. Return the session ID
3. You can then ask to check status or get results

### Code Generation

```
Create a coder agent to implement a rate limiter class in Python with sliding window algorithm.
```

Claude will:
1. Spawn a coder agent
2. The agent will generate code based on patterns from the codebase
3. Results include generated code and test files

### Multi-Agent Workflow

```
1. First, spawn a researcher to investigate OAuth 2.0 best practices
2. Then spawn a coder to implement the authentication flow based on the research
3. Finally, spawn a reviewer to check the implementation
```

Claude can orchestrate multiple agents in sequence, using results from one as input to the next.

### Check Progress

```
What's the status of my researcher agent?
```

or

```
List all my active agents and show me any completed results.
```

## Environment Variables

The MCP server supports these environment variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `ANTHROPIC_API_KEY` | API key for Claude-based agents | Required |
| `OPENAI_API_KEY` | API key for OpenAI-based agents | Optional |
| `OLLAMA_HOST` | Host for local Ollama models | `http://localhost:11434` |
| `TITAN_LOG_LEVEL` | Logging verbosity | `INFO` |
| `TITAN_METRICS_PORT` | Prometheus metrics port | `9100` |

## Troubleshooting

### Server Not Starting

**Symptom:** Claude cannot connect to Titan MCP server.

**Solutions:**
1. Verify Python is in your PATH: `which python`
2. Check the `cwd` path in your config points to the agentic-titan directory
3. Ensure dependencies are installed: `pip install -e .`
4. Check server logs: run `python -m mcp.server` manually to see errors

### Agent Hangs

**Symptom:** Agent status stays "running" indefinitely.

**Solutions:**
1. Check if LLM API keys are configured correctly
2. Verify network connectivity to LLM providers
3. Cancel the agent and try again with a simpler task
4. Check logs for rate limiting errors

### No Results Returned

**Symptom:** Agent completes but result is empty.

**Solutions:**
1. Verify the task description is clear and specific
2. Check agent_status for any error messages
3. Ensure the agent type matches the task (e.g., don't use coder for research)

### Permission Errors

**Symptom:** "Permission denied" or sandbox errors.

**Solutions:**
1. Ensure the MCP server has read/write access to its working directory
2. If using Docker, verify volume mounts are correct
3. Check file permissions on the agentic-titan directory

## Advanced Configuration

### Custom Agent Settings

Pass additional settings via the MCP server environment:

```json
{
  "mcpServers": {
    "titan": {
      "command": "python",
      "args": ["-m", "mcp.server"],
      "cwd": "/path/to/agentic-titan",
      "env": {
        "ANTHROPIC_API_KEY": "${env:ANTHROPIC_API_KEY}",
        "TITAN_DEFAULT_MODEL": "claude-3-5-sonnet",
        "TITAN_MAX_TURNS": "20",
        "TITAN_TIMEOUT_SECONDS": "300"
      }
    }
  }
}
```

### Running with Monitoring

To enable Prometheus metrics collection:

```json
{
  "env": {
    "TITAN_METRICS_ENABLED": "true",
    "TITAN_METRICS_PORT": "9100"
  }
}
```

Then access metrics at `http://localhost:9100/metrics`.

### Using with Redis

For multi-instance deployments with shared state:

```json
{
  "env": {
    "REDIS_URL": "redis://localhost:6379/0"
  }
}
```

## Security Notes

- Never commit API keys to version control
- Use environment variable references (`${env:VAR_NAME}`) in configs
- The MCP server runs with the same permissions as the Claude application
- Agent-generated code is not automatically executed unless explicitly requested

## Getting Help

- Check the [Titan documentation](./README.md)
- Review [MCP protocol specification](https://spec.modelcontextprotocol.io/)
- Open an issue on the repository for bugs or feature requests
