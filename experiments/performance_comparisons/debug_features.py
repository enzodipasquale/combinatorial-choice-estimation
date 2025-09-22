#!/usr/bin/env python3
"""
Debug the features oracle validation issue
"""

import numpy as np

def features_oracle(i_id, B_j, data):
    """Compute features for a given agent and bundle(s)."""
    print(f"DEBUG: i_id={i_id}, B_j.shape={B_j.shape}, B_j={B_j}")
    modular_agent = data["agent_data"]["modular"][i_id]
    modular_agent = np.atleast_2d(modular_agent)
    print(f"DEBUG: modular_agent.shape={modular_agent.shape}, modular_agent={modular_agent}")
    
    single_bundle = False
    if B_j.ndim == 1:
        B_j = B_j[:, None]
        single_bundle = True
        print(f"DEBUG: After conversion, B_j.shape={B_j.shape}")
    
    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
        agent_sum = modular_agent @ B_j  # (1, num_items) @ (num_items, num_bundles) = (1, num_bundles)
    print(f"DEBUG: agent_sum.shape={agent_sum.shape}, agent_sum={agent_sum}")
    neg_sq = -np.sum(B_j, axis=0, keepdims=True) ** 2
    print(f"DEBUG: neg_sq.shape={neg_sq.shape}, neg_sq={neg_sq}")
    
    features = np.vstack((agent_sum, neg_sq))
    print(f"DEBUG: features.shape={features.shape}, features={features}")
    if single_bundle:
        result = features[:, 0]
        print(f"DEBUG: single bundle result.shape={result.shape}, result={result}")
        return result
    return features

# Test the features oracle
num_agents = 5
num_items = 5
modular = np.random.normal(0, 1, (num_agents, num_items))
errors = np.random.normal(0, 0.1, (num_agents, num_items))
agent_data = {"modular": modular}
input_data = {"agent_data": agent_data, "errors": errors}

print("Testing features oracle with validation-like input:")
test_bundle = np.ones(num_items)
print(f"test_bundle.shape={test_bundle.shape}, test_bundle={test_bundle}")

result = features_oracle(0, test_bundle, input_data)
print(f"Final result.shape={result.shape}, result={result}")
