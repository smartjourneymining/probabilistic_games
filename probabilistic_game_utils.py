import numpy as np
import networkx as nx
import copy

def entropy(p1, p2):
    if p1 == 0 or p2 == 0:
        return 0
    return - p1*np.log2(p1) - p2* np.log2(p2)

def weight(trace):
    return 1 if any("positive" in pos for pos in trace) else -1

def distribution(s,t, log, edge_mapping):
    distr = {1.0: 0 , -1.0 : 0}
    for trace_index in edge_mapping:
        w = weight(log[trace_index])
        distr[w] += 1 #
    return distr[1], distr[-1]

def add_gas_and_user_count(g : nx.digraph, log):
    g = copy.deepcopy(g)
    for e in g.edges:
        g.edges[e]['count'] = 0
        g.edges[e]['gas'] = 0
        g.edges[e]['trace_indices'] = []
    
    for t, index in zip(log, range(len(log))):
        assert t[0] == 'start', 'trace start with ' +  t[0] + ', not "start"'
        current_state = 'q0: start'
        for io_pair in t[1:]:
            e = [e for e in g.out_edges(current_state) if g.edges[e]['action'] == io_pair[0] and io_pair[1] in e[1]]
            assert len(e)==1, "too many edges, should be one: " + str(e)
            next_state = e[0][1]# unpack only edge and use target state
            g.edges[(current_state,next_state)]['count'] = g.edges[(current_state,next_state)]['count']+1
            g.edges[(current_state,next_state)]['trace_indices'].append(index)
            current_state = next_state
        assert "pos" in io_pair[1] or "neg" in io_pair[1] # final states are positive or negative

    # only env and do_nothing actions can have gas 0
    for e in g.edges:
        if g.edges[e]['count'] == 0:
            assert g.edges[e]['action'] == 'env' or g.edges[e]['action'] == 'do_nothing','action "' + g.edges[e]['action'] + '" has count 0'
        else:
            # assign gas
            p1, p2 = distribution(e[0],e[1], log, g.edges[e]['trace_indices'])
            w = 1 if p1 >= p2 else -1
            wp1 = p1/(p1+p2)
            wp2 = p2/(p1+p2)
            scaling = 10
            entro = entropy(wp1, wp2)
            g.edges[e]['gas'] = (((1-entro) * w) -0.21 )*200        

    for e in g.edges:
        print(e, g.edges[e])
        if g.edges[e]['gas'] == 0:
            assert g.edges[e]['action'] == 'env' or g.edges[e]['action'] == 'do_nothing','action "' + g.edges[e]['action'] + '" has gas 0'
    
    return g

def add_neutral_user_transition(g : nx.DiGraph, debug = False):
    g = copy.deepcopy(g)
    for s in g.nodes:
        if "customer" in s:
            print("current state", s)
            # check that ingoing edge is only "env" (not a regular state that contains "user")
            in_activities = [g.edges[t]['action'] for t in g.in_edges(s)]
            if(set(in_activities) == {'env'}):
                for in_e in g.in_edges(s):
                    print("in_e", in_e)
                    before_transition = g[in_e[0]]
                    assert(len(before_transition) <= 2)
                    if len(before_transition) == 2:
                        company_controlled_state = [state for state in before_transition if "company" in state]
                        assert(len(company_controlled_state) == 1)
                        target_state = company_controlled_state[0]
                        if debug:
                            print("added edge from ", s, "to" , target_state, "with" , "do_nothing", "prob_weight", 1)
                        g.add_edge(s, target_state, action="do_nothing", prob_weight=1, controllable = False)
    return g