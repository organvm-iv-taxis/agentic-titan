# Phase 4: Operational Hardening - Chaos Engineering

## Objective
Verify the resilience of the `FissionFusionTopology` by simulating agent "death" and observing cluster rebalancing.

## Experiment Design
1.  **Setup**: Initialize a `FissionFusionTopology` with 10 agents in 2 clusters.
2.  **Steady State**: Verify all agents are connected and responsive.
3.  **Chaos Injection**: Terminate 3 random agents (simulate crash).
4.  **Observation**: 
    - Monitor `CLUSTER_COUNT` and `NEIGHBOR_COUNT` metrics.
    - Check if remaining agents re-establish connectivity.
5.  **Recovery**: Spawn 3 new replacement agents.
6.  **Validation**: Confirm system returns to healthy state.

## Implementation (tests/chaos/test_fission_recovery.py)
```python
import pytest
import asyncio
from hive.topology import TopologyEngine, TopologyType

@pytest.mark.asyncio
async def test_fission_cluster_recovery():
    engine = TopologyEngine()
    topo = engine.create_topology(TopologyType.FISSION_FUSION)
    
    # 1. Setup
    agents = [f"agent-{i}" for i in range(10)]
    for i, aid in enumerate(agents):
        cluster = "A" if i < 5 else "B"
        topo.add_agent(aid, aid, ["worker"], cluster_id=cluster)
        
    assert len(topo.nodes) == 10
    
    # 2. Chaos
    import random
    victims = random.sample(agents, 3)
    print(f"Killing agents: {victims}")
    
    for v in victims:
        topo.remove_agent(v)
        
    assert len(topo.nodes) == 7
    
    # 3. Validation
    # Check if remaining agents in affected clusters still have neighbors
    for node in topo.nodes.values():
        # A node should have neighbors if it's not the last one in cluster
        cluster_peers = [n for n in topo.nodes.values() 
                        if n.metadata["cluster_id"] == node.metadata["cluster_id"] 
                        and n.agent_id != node.agent_id]
        
        if cluster_peers:
            assert len(node.neighbors) > 0
            
    print("Cluster connectivity maintained after node failure.")
```
