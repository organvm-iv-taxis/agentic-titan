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

if __name__ == "__main__":
    import sys
    pytest.main(sys.argv)
