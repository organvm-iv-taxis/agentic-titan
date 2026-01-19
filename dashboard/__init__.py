"""
Agentic Titan - Web Dashboard

A FastAPI-based web dashboard for managing and monitoring the agent swarm.

Features:
- Real-time agent status
- Topology visualization
- Task management
- Metrics dashboard
- Learning insights

Start the dashboard:
    titan dashboard serve --port 8080
"""

from dashboard.app import create_app, TitanDashboard

__all__ = ["create_app", "TitanDashboard"]
