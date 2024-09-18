import numpy as np
import networkx as nx
import copy
from  matplotlib.colors import LinearSegmentedColormap # for color map
from matplotlib.colors import rgb2hex
from networkx.drawing.nx_agraph import to_agraph

def entropy(p1, p2):
    """Binary shannon entropy for p1 and p2 counts.
    
    Returns:
        float: Entropy
    """
    if p1 == 0 or p2 == 0:
        return 0
    return - p1*np.log2(p1) - p2* np.log2(p2)

def weight(trace):
    """
    Weight of trace is 1 is ending in positive, else -1.
    """
    return 1 if any("positive" in pos for pos in trace) else -1

def distribution(s,t, log, edge_mapping):
    distr = {1.0: 0 , -1.0 : 0}
    for trace_index in edge_mapping:
        w = weight(log[trace_index])
        distr[w] += 1 #
    return distr[1], distr[-1]

def add_gas_and_user_count(g : nx.digraph, log, greps_values = False, debug = False):
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
            if greps_values:
                g.edges[e]['gas'] = (((1-entro) * w) -0.1 )*20
            else:
                g.edges[e]['gas'] = (((1-entro) * w) -0.21 )*20     

    for e in g.edges:
        if debug:
            print(e, g.edges[e])
        if g.edges[e]['gas'] == 0:
            assert g.edges[e]['action'] == 'env' or g.edges[e]['action'] == 'do_nothing','action "' + g.edges[e]['action'] + '" has gas 0'
    
    return g

def add_neutral_user_transition(g : nx.DiGraph, debug = False):
    """Changes g not in-place by adding "no-action" transitions from user state to service provider controlled state.

    Args:
        g (nx.DiGraph): Stochastic user journey game.
        debug (bool, optional): Additional prints if set to true. Defaults to False.

    Returns:
        nx.DiGraph: Stochastic user journey game with additional "no-action" transitions.
    """
    g = copy.deepcopy(g)
    for s in g.nodes:
        if "customer" in s:
            # check that ingoing edge is only "env" (not a regular state that contains "user")
            in_activities = [g.edges[t]['action'] for t in g.in_edges(s)]
            if(set(in_activities) == {'env'}):
                for in_e in g.in_edges(s):
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


def assert_no_det_cycle(g):
    for c in list(nx.simple_cycles(g)):
        found = False
        for i in range(len(c)):
            if g[c[i]][c[(i+1)%len(c)]]['prob_weight'] != 1:
                found = True
        assert found
        
def can_be_merged(g, results_file, accuracy_digits):
    """
    Returns true if two neighbouring nodes have the same value in results_file for the given accuracy.
    """
    for s in g.nodes():
        reachable_values = [round(results_file[t],accuracy_digits) for t in g[s]]
        if round(results_file[s], accuracy_digits) in reachable_values:
            return s 
    return None

def reduce_graph(g, results_file, accuracy_digits):
    """
    NOTE: One positive and one negative node is kept and all remaining from positive/negative cluster are merged into them.
    """
    neg_cluster = []
    pos_cluster = []
    print("size start", len(g.nodes()))
    s = can_be_merged(g, results_file, accuracy_digits)
    while(s != None):
        for t in g[s]:
            if round(results_file[t], accuracy_digits) != round(results_file[s],accuracy_digits):
                continue
            g = nx.contracted_nodes(g, s, t, self_loops = False)
        s = can_be_merged(g, results_file, accuracy_digits)

    for s in g:
        if results_file[s] == 0:
            neg_cluster.append(s)
        if results_file[s] == 1:
            pos_cluster.append(s)
    for s in pos_cluster[1:]:
        g = nx.contracted_nodes(g, pos_cluster[0], s, self_loops=False)
    for s in neg_cluster[1:]:
        g = nx.contracted_nodes(g, neg_cluster[0], s, self_loops=False)

    g.remove_edges_from(nx.selfloop_edges(g))

    print("size reduced", len(g.nodes()))
    return g

def compute_color_map(g, results_file):
    """Returns a matplotlib.colors.rgb2hex colormap for each state in g, determined by results in results_file.

    Args:
        g (nx.DiGraph): Stochastic user journey game.
        results_file (dict): Mapping from states to MC results.

    Returns:
        matplotlib.colors.rgb2hex : Colormap
    """
    #c = ["#0C7BDC", "#FFC20A", "darkgreen"]
    c = ["#D55E00", "#F0E442", "#009E73"]
    v = [0, 0.5, 1]
    l = list(zip(v,c))
    cmap=LinearSegmentedColormap.from_list('rg',l, N=256)
    s = cmap(0.23)
    map = {}
    for s in g.nodes():
        map[s] = rgb2hex(cmap(results_file[s])) # have to convert to hex color
    return map

def draw_dfg(g, name, names={}, layout = "sfdp", color_map = [], add_greps_cluster = False):
    """
    Helper function to draw Networkx graphs.
    """
    scaling = 10
    # build graph with variable thickness
    #scaling = 1/np.mean(list(nx.get_edge_attributes(g,'edge_weight').values()))

    A = to_agraph(g)

    edge_weights = nx.get_edge_attributes(g,'edge_weight')
    for e in edge_weights:
        e = A.get_edge(e[0], e[1])
        e.attr["penwidth"] = edge_weights[e]*scaling
        e.attr["fontsize"] = "20"
    for e in g.edges:
        edge = A.get_edge(e[0], e[1])
        if 'controllable' in g[e[0]][e[1]]:
            if not g[e[0]][e[1]]['controllable']:
                edge.attr["style"] = "dotted"
                #edge.attr["label"] =  str(g[e[0]][e[1]]["prob_weight"])
        #A.add_edge(e[0], e[1], penwidth = edge_weights[e]*scaling)

    for n in A.nodes():
        if n in names:
            new = names[n]
            if isinstance(names[n], float): 
                new = round(names[n], 2)
            n.attr['label'] = new
            #if new == 1:
            #    n.attr['label'] = "pos"
            #elif new == 0:
            #    n.attr['label'] = "neg"
            #else:
            #    n.attr["label"] = "" # uncomment to print state names
        if n in color_map:
            n.attr['color'] = color_map[n]
    
        n.attr['fontsize'] = 120
        n.attr['penwidth'] = 30
        n.attr['height'] = 3
        n.attr['width'] = 3

    for e in A.edges():
        e.attr['penwidth'] = 20
        e.attr["fontsize"] = 120
        e.attr["label"] = str(round(g[e[0]][e[1]]["prob_weight"],2))
        e.attr["color"] = "black"

        if g[e[0]][e[1]]['gas'] > 0:
            e.attr["color"] ="#009E73"
        if g[e[0]][e[1]]['gas'] < 0:
            e.attr["color"] ="#D55E00"
    
    if add_greps_cluster:
        # Adding clusters for the GrepS phases
        onboarding = ["T"+str(i) for i in range(0,6)]
        onboarding = [n for n in A.nodes() if names[n] in onboarding]
        A.add_subgraph(onboarding, name='cluster_onboarding', label= "Sign-up", color = "orange", fontsize = 90, fontcolor = "orange", penwidth= 10)
        task = ["T"+str(i) for i in range(6,21)]
        task = [n for n in A.nodes() if names[n] in task]
        A.add_subgraph(task, name='cluster_task', label= "Solve tasks", color = "blue", fontsize = 90, fontcolor = "blue", penwidth= 10)
        evaluation = ["T"+str(i) for i in range(21,27)]
        evaluation = [n for n in A.nodes() if names[n] in evaluation]
        A.add_subgraph(evaluation, name='cluster_evaluation', label= "Review and share", color = "purple", fontsize = 90, fontcolor = "purple", penwidth= 10)
                
    A.write(name.split(".")[0]+".dot")
    A.layout(layout)
    A.draw(name)
    print("Plotted", name)
    

def get_probs_file(results_file, g, printer):
    """
    Returns a mapping from states in g to results in results_file.
    Results_file initially maps from the adapted names in printer to MC results.
    """
    isomorphism = nx.vf2pp_isomorphism(printer.g, g, node_label=None)
    parsed_results_file = {isomorphism[r] : results_file[r] for r in results_file}
    return parsed_results_file


def plot_reduction(g, name, results_file, accuracy_digits, layout = "sdf"):
    """
    Calls the reduction function merging states with equal results and plots the reduced graph.
    """
    g = copy.deepcopy(g)
    g = reduce_graph(g, results_file, accuracy_digits)
    color_map = compute_color_map(g, results_file)
    draw_dfg(g, name, names = results_file, layout = layout, color_map=color_map, add_greps_cluster=False)