import torch
import sys
import networkx as nx


def solve_QSNetwork(self, _pricing_pb, local_id, lambda_k, p_j):
    local_id = torch.arange(self.num_local_agents) if local_id is None else local_id
    n_local_id = len(local_id)

    error_i_j = self.torch_local_errors
    modular_j_k = self.torch_item_data.get("modular", None)
    modular_i_j_k = self.torch_local_agent_data.get("modular", None)
    quadratic_j_j_k = self.torch_item_data.get("quadratic", None)
    quadratic_i_j_j_k = self.torch_local_agent_data.get("quadratic", None)

    if quadratic_j_j_k is not None:
        assert torch.all(quadratic_j_j_k.diagonal(dim1 = 0, dim2 = 1) == 0)
    if quadratic_i_j_j_k is not None:
        assert torch.all(quadratic_i_j_j_k.diagonal(dim1 = 1, dim2 = 2) == 0)

    device = error_i_j.device
    precision = error_i_j.dtype
    lambda_k = torch.tensor(lambda_k, device=device, dtype=precision)
    if p_j is not None:
        p_j = torch.tensor(p_j, device=device, dtype=precision)

    num_mod = modular_j_k.shape[-1] if modular_j_k is not None else 0
    num_mod_agent = modular_i_j_k.shape[-1] if modular_i_j_k is not None else 0
    num_quad = quadratic_j_j_k.shape[-1] if quadratic_j_j_k is not None else 0
    num_quad_agent = quadratic_i_j_j_k.shape[-1] if quadratic_i_j_j_k is not None else 0

    offset = 0
    lambda_mod = lambda_k[offset:offset + num_mod]; offset += num_mod
    lambda_mod_agent = lambda_k[offset:offset + num_mod_agent]; offset += num_mod_agent
    lambda_quad = lambda_k[offset:offset + num_quad]; offset += num_quad
    lambda_quad_agent = lambda_k[offset:]

    P_i_j_j = torch.zeros((n_local_id, self.num_items, self.num_items), device=device, dtype=precision)

    if quadratic_j_j_k is not None:
        P_i_j_j += (quadratic_j_j_k @ lambda_quad).unsqueeze(0)
    if quadratic_i_j_j_k is not None:
        P_i_j_j += quadratic_i_j_j_k[local_id] @ lambda_quad_agent

    P_i_j_j = (P_i_j_j + P_i_j_j.transpose(1, 2))


    modular_components = error_i_j[local_id]
    if modular_j_k is not None:
        modular_components += modular_j_k @ lambda_mod
    if modular_i_j_k is not None:
        modular_components += modular_i_j_k[local_id] @ lambda_mod_agent
    if p_j is not None:
        modular_components -= p_j.unsqueeze(0)
    
    P_i_j_j.diagonal(dim1=1, dim2=2).copy_(modular_components)
 
    constraint_i_j = torch.isinf(error_i_j[local_id])
    assert torch.all(modular_components[constraint_i_j] == -float('inf'))
    assert torch.all(modular_components[~constraint_i_j] != -float('inf'))
    

    def build_graph(a_j_j, a_j, P, nodes):
        n = a_j_j.shape[0]
        G = nx.DiGraph()

        for i in nodes:
            for j in nodes:
                if j > i: 
                    G.add_edge(i, j, capacity=a_j_j[i, j].item())
        G.add_node('s')
        G.add_node('t')

        for i in nodes:
            # assert a_j[i].item() < float('inf'), f"Capacity for node {i} is infinite, which is not allowed."
            if P[i]:
                G.add_edge(i, 't', capacity=a_j[i].item())
            else:
                G.add_edge('s', i, capacity=a_j[i].item())
        return G



    optimal_bundle = torch.zeros((n_local_id, self.num_items), dtype=torch.bool, device=self.torch_device)

    for i in range(n_local_id):
        b_j_j = torch.triu(P_i_j_j[i], diagonal=1).clone()
        assert torch.all(b_j_j >= 0)
        b_j = - P_i_j_j[i].diagonal().clone()
        
        # Get posiform
        a_j_j = b_j_j
        val = b_j - b_j_j.sum(1)
        
        P = (val >= 0)
        a_j = torch.where(P, val, -val)

        assert torch.all(a_j >= 0)
        assert torch.all(a_j_j >= 0)
        

        choice_set = torch.where(~constraint_i_j[i])[0].tolist()
        
        assert torch.all(a_j[constraint_i_j[i]] == float('inf'))
        assert torch.all(a_j[choice_set] < float('inf')), f"Capacity for node {i} is infinite, which is not allowed."

        G = build_graph(a_j_j, a_j, P, choice_set)


        cut_value, partition = nx.minimum_cut(G, 's', 't', flow_func=nx.algorithms.flow.preflow_push)
        S, T = partition
        S = list(S - {'s'})

        optimal_bundle[i,S] = True

    # assert torch.all(torch.isinf(error_i_j[optimal_bundle]) == False)
    return optimal_bundle