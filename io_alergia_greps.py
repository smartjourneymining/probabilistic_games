PRISM_PATH = "/home/prism-games-3.2.1-src/prism/bin/prism"  # path to PRISM-games install
STRATEGY_PATH = "adv.tra" # path to where strategies shall be stored
STORE_PATH = "/home/generated/" # path to where generated models can be stored
QUERY_PATH = "/home/queries/" # path to queries
OUTPUT_PATH = "/home/out/" # path to PRISM-games generated output files


from journepy.src.preprocessing.greps import preprocessed_log
from journepy.src.alergia_utils import convert_utils
from journepy.src.mc_utils.prism_utils import PrismPrinter
from journepy.src.mc_utils.prism_utils import PrismQuery

import probabilistic_game_utils as pgu 
from aalpy.learning_algs import run_Alergia
from aalpy.utils import save_automaton_to_file
import pandas as pd
import matplotlib.pyplot as plt
from networkx.drawing.nx_agraph import to_agraph
import json
import networkx as nx
import subprocess
import copy

from  matplotlib.colors import LinearSegmentedColormap # for color map
from matplotlib.colors import rgb2hex

import plotly.graph_objects as go



def assert_no_det_cycle(g):
    for c in list(nx.simple_cycles(g)):
        found = False
        for i in range(len(c)):
            if g[c[i]][c[(i+1)%len(c)]]['prob_weight'] != 1:
                found = True
        assert found
        
def plot_fig_4a(file_name):
    df_visual = pd.read_csv(file_name)
    plt.plot(df_visual['envprob'], df_visual['Result'],linewidth=3)
    plt.vlines(x=0, ymin=0, ymax = 1, linewidth=2, color = 'grey', linestyles='--')
    plt.text(-1, 0.05, 'Service Provider', fontsize = 18)
    plt.text(0.8, 0.05, 'User', fontsize = 18)
    plt.xlabel("Scaled activity (q)", fontsize=22)
    plt.ylabel("Success probability", fontsize=22)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=20)
    plt.tight_layout()
    plt.savefig("out/greps/fig4a.png", dpi=300)
    plt.close()
    
def plot_fig_4b():
    # produces Fig. 1b
    file_name = OUTPUT_PATH+"steps_gas_pos_bound.txt"
    df_visual = pd.read_csv(file_name)
    plt.plot(df_visual['m1']/4, df_visual['Result'], label="Max pos", linewidth = 3)
    file_name = OUTPUT_PATH+"steps_gas_neg_bound.txt"
    df_visual = pd.read_csv(file_name)
    plt.plot(df_visual['m1']/4, df_visual['Result'], label="Min neg", linewidth = 3)
    # divide by 4 to account for (1) env action and (2) dummy actions to calculate rewards

    plt.legend(fontsize=18)
    plt.xlabel("Steps S", fontsize=22)
    plt.ylabel("Accumulated weight", fontsize=22)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.tight_layout()
    plt.savefig("out/greps/fig4b.png", dpi=300)
    plt.close()
    
def plot_fig_4c():
    # plot Fig. 1c
    file_name = OUTPUT_PATH+"bounded_steps_gas_min_gas_greps.txt"
    df_visual = pd.read_csv(file_name)

    df_visual_grouped = df_visual.groupby(['m0','m2'])

    result_dict = {}
    for group in df_visual_grouped.groups.keys():
        r = tuple([round(h,2) for h in df_visual_grouped.get_group(group)['Result'].values])
        if r in result_dict:
            result_dict[r].append(group)
        else:
            result_dict[r] = [group]

    for k in result_dict:
        max_m0 = max(h[0] for h in result_dict[k])
        max_m2 = max(h[1] for h in result_dict[k])
        assert((max_m0, max_m2) in result_dict[k])
        if len(set(k)) != 1:
            plt.plot(df_visual_grouped.get_group((max_m0, max_m2))['m1'], k, label = str((max_m0, max_m2)))

    plt.legend(fontsize=16)
    plt.xlabel("Steps S", fontsize=22)
    plt.ylabel("Success probability", fontsize=22)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.tight_layout()
    plt.savefig("out/greps/fig4c.png", dpi=300)
    plt.close()

def plot_reduction(g, name, results_file, layout = "sdf"):
    g = copy.deepcopy(g)
    g = reduce_graph(g, results_file)
    color_map = compute_color_map(g, results_file)
    draw_dfg(g, name, names = results_file, layout = layout, color_map=color_map)
    
def can_be_merged(g, results_file):
    for s in g.nodes():
        reachable_values = [round(results_file[t],2) for t in g[s]]
        if round(results_file[s],2) in reachable_values:
            return s 
    return None

"""
NOTE: One positive and one negative node is kept and all remaining from positive/negative cluster are merged into them.
"""
def reduce_graph(g, results_file):
    neg_cluster = []
    pos_cluster = []
    print("size start", len(g.nodes()))
    s = can_be_merged(g, results_file)
    while(s != None):
        for t in g[s]:
            if round(results_file[t],2) != round(results_file[s],2):
                continue
            g = nx.contracted_nodes(g, s, t, self_loops = False)
        s = can_be_merged(g, results_file)

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
    c = ["darkred","gold","darkgreen"]
    v = [0,0.5,1]
    l = list(zip(v,c))
    cmap=LinearSegmentedColormap.from_list('rg',l, N=256)
    s = cmap(0.23)
    map = {}
    for s in g.nodes():
        map[s] = rgb2hex(cmap(results_file[s])) # have to convert to hex color
    return map

def draw_dfg(g, name, names={}, layout = "sfdp", color_map = []):
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
            e.attr["color"] ="darkgreen"
        if g[e[0]][e[1]]['gas'] < 0:
            e.attr["color"] ="red"
            
    print(A.nodes())
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
    isomorphism = nx.vf2pp_isomorphism(printer.g, g, node_label=None)
    parsed_results_file = {isomorphism[r] : results_file[r] for r in results_file}
    return parsed_results_file

def transform_strategy(strategy, g, printer):
    """ 
    Adjusts the strategy naming from PRISM node naming to original node naming in g
    """
    isomorphism = nx.vf2pp_isomorphism(printer.g, g, node_label=None)
    strategy_isomorphism = {k[len(isomorphism[k].split(": ")[0]):] : ": ".join(isomorphism[k].split(": ")[1:]) for k in isomorphism}
    parsed_strategy = {isomorphism[r] : strategy_isomorphism[strategy[r]] if strategy[r] not in ["env", "user", "company", "do_nothing"] else strategy[r] for r in strategy}
    return parsed_strategy

def lost_users(g, results_file, strategy):
    for s in strategy:
        assert s in g
        next_states = [t for t in g[s] if g[s][t]['action'] == strategy[s]]

        assert(next_states)
        total_lost_users = 0
        for t in next_states:
            action_outcome_cost = len(g[s][t]['trace_indices']) * abs(round(results_file[s],5)-round(results_file[t],5)) #* g[s][t]['prob_weight']
            
            total_lost_users += action_outcome_cost

            if action_outcome_cost!= 0:
                print(len(g[s][t]['trace_indices']), "*", (round(results_file[s],5), round(results_file[t],5)))
                print(action_outcome_cost)
        if total_lost_users != 0:
            print("at", s, "is", strategy[s], "selected")
            print("total", total_lost_users)
            print()
      
def reduced_sankey_diagram(g, results_file):
    
    naming = {
    "q20: companyTask event: 1": "T11",
    "q27: companyTask event: 2": "T13",
    "q31: companyTask event: 3": "T25",
    "q52: customerwaitingForActivityReport" : "unsucc",
    "q51: companywaitingForActivityReport" : "succ",
    "Results shared" : "T26",
    "Logged in: Web page - Approval" : "T26",
    "negative" : "neg",
    "positive" : "pos",
    "q0: start": "T9",
    "finished":"finished"}
    
    g = copy.deepcopy(g)
    g = reduce_graph(g, results_file)
    color_map = compute_color_map(g, results_file)


    node_list = list(g.nodes())
    node_dict = {node_list[i] : i for i in range(len(node_list))}
    edge_list = g.edges()
    print(edge_list)

    print([len(g[e[0]][e[1]]['trace_indices'])  * abs(round(results_file[e[0]],5)-round(results_file[e[1]],5)) for e in edge_list])
    #print([len(g.edges[e]['trace_indices']) for e in edge_list])
    fig = go.Figure(data=[go.Sankey(
    node = dict(
        pad = 15,
        thickness = 20,
        line = dict(color = "black", width = 0.5),
        label = [naming[str(s)] for s in node_list],
        color = [color_map[s] for s in node_list],
        align = "right"
    ),
    link = dict(
        source = [node_dict[e[0]] for e in edge_list],
        target = [node_dict[e[1]] for e in edge_list],
        value = [len(g[e[0]][e[1]]['trace_indices']) * abs(round(results_file[e[0]],5)-round(results_file[e[1]],5)) for e in edge_list]
        #[len(g.edges[e]['trace_indices']) for e in edge_list]
    ))])
    fig.update_layout(
    font=dict(size = 40)
    )
    #fig.show()
    fig.to_image(format = "png", engine = "kaleido")
    fig.write_image("out/greps/fig5.png")
    fig.write_html("out/greps/fig5.html")
            
def main():   
    # load files
    filtered_log = preprocessed_log("data/data.csv", include_loggin=False) # also discards task-event log-in
    
    # load actor mapping: maps events to an actor (service provider or user)
    with open('data/activities_greps.xml') as f:
        data = f.read()
    actors = json.loads(data)
    
    actions_to_activities = {}
    for a in actors:
        if actors[a] == "company":
            if a in ['vpcAssignInstance', 'Give feedback 0', 'Results automatically shared', 'waitingForActivityReport']: # todo: might be quite realistic?
                actions_to_activities[a] = "company"
            else:  
                actions_to_activities[a] = a
        else:
            if a == "negative":
                actions_to_activities[a] = "user"
            elif "Give feedback" in a or "Task event" in a:
                actions_to_activities[a] = a
            else:
                actions_to_activities[a] = "user"
                
    print()
    print("Action mapping")
    print(actions_to_activities)
    
    filtered_log_activities = [[e['concept:name'] for e in t] for t in filtered_log]
    
    data = [[(actions_to_activities[t[i]], t[i]) for i in range(1, len(t))] for t in filtered_log_activities]
    for d in data:
        d.insert(0, 'start')
        
    model = run_Alergia(data, automaton_type='mdp', eps=0.9, print_info=True)
    
    # quantify environment - distribution of players after for events is learned
    data_environment = []
    for trace in data:
        current = [trace[0]]
        for i in range(1, len(trace)):
            e = trace[i]
            previous_state = "start" if i == 1 else trace[i-1][1]
            
            # encode decision in one step
            current.append(('env', actors[e[1]] + previous_state))
            current.append(e)
        data_environment.append(current)
        
    print()
    print("Action mapping for environment")
    print(actions_to_activities)
    
    model_environment = run_Alergia(data_environment, automaton_type='mdp', eps=0.1, print_info=True)
    
    g = convert_utils.mdp_to_nx(model_environment, actors)
    # users can decide to "do nothing"
    g = pgu.add_neutral_user_transition(g)
    g = pgu.add_gas_and_user_count(g, data_environment, greps_values=True)
    assert_no_det_cycle(g)
    
    
    # MODEL CHECKING # 
    printer = PrismPrinter(g, STORE_PATH, "alergia_reduction_model.prism")
    printer.write_to_prism()
    
    query = PrismQuery(g, STORE_PATH, "alergia_reduction_model.prism", PRISM_PATH)
    # Query Q1 from Table 1
    results_file = query.query(QUERY_PATH+"pos_alergia.props", write_parameterized=True)
    print(results_file['q0start'])
    
    # Query Q2, Q3, and Q4 from Table 1
    subprocess.run([PRISM_PATH, STORE_PATH+"alergia_reduction_model.prism", QUERY_PATH+"mc_runs.props", "-const", "envprob=0"]) 
    
    # run Activity experiment
    # remove "stdout=subprocess.DEVNULL" to print output again
    file_name = OUTPUT_PATH+"succ_prop_cond.txt"
    subprocess.run([PRISM_PATH, STORE_PATH+"alergia_reduction_model.prism", 
                    QUERY_PATH+"pos_alergia.props", 
                    "-const", "envprob=-0.95:0.05:0.95", "-exportresults", file_name+":dataframe"], stdout=subprocess.DEVNULL) 
    
    # produces Fig. 4a
    plot_fig_4a(file_name)
    
    # run gas upper and lower bound under limited steps
    PrismPrinter(g, STORE_PATH, "alergia_reduction_model.prism").write_to_prism(write_extended_parameterized=True)
    file_name = OUTPUT_PATH+"steps_gas_pos_bound.txt"
    subprocess.run([PRISM_PATH, STORE_PATH+"alergia_reduction_model.prism", 
                    QUERY_PATH+"reward_props.props", "-prop", "3",
                    "-const", "m0=0,m1=0:1:140,m2=0,", "-exportresults", file_name+":dataframe"], stdout=subprocess.DEVNULL) 
    file_name = OUTPUT_PATH+"steps_gas_neg_bound.txt"
    subprocess.run([PRISM_PATH, STORE_PATH+"alergia_reduction_model.prism", 
                    QUERY_PATH+"reward_props.props", "-prop", "4",
                    "-const", "m0=0,m1=0:1:140,m2=0,", "-exportresults", file_name+":dataframe"], stdout=subprocess.DEVNULL) 
    plot_fig_4b()
    
    
    # induces and reduced model
    query = PrismQuery(g, STORE_PATH, "alergia_reduction_model.prism", PRISM_PATH)
    strategy = query.get_strategy(QUERY_PATH+"pos_alergia.props")
    
    # Naming for Fig. 3
    naming = {
        "registered" : "T0",
        "activated": "T1",
        #"Logged in: Web page - Sign up" : "T2",
        "vpcCreateUserOnInstance" : "T3",
        "vpcAssignInstance" : "T2",
        "readyToStart" : "T4",
        "Task event: loggedIn" : "T5",
        "started" : "T6",
        "Task event: 0": "T7",
        "Give feedback 0" : "T8",
        "Task event: 1": "T9",
        "Give feedback 1" : "T10",
        "Task event: 2": "T11",
        "Give feedback 2" : "T12",
        "Task event: 3": "T13",
        "Give feedback 3" : "T14",
        "Task event: 4": "T15",
        "Give feedback 4" : "T16",
        "Task event: 5": "T17",
        "waitingForManualScores" : "T18",
        "Logged in: Web page - Task" : "T19",
        "waitingForScores" : "T20",
        "waitingForResultApproval" : "T21",
        "waitingForSubjectAcceptance" : "T22",
        "subjectAcceptanceReceived" : "T23",
        "Results automatically shared" : "T24",
        "waitingForActivityReport" : "T25",
        "Results shared" : "T26",
        "Logged in: Web page - Approval" : "T26",
        "negative" : "unsucc",
        "positive" : "succ",
        "start": "s₀",
        "finished":"finished"
    }

    extended_naming = {}
    for k in naming:
        extended_naming['customer'+k] = "U-"+naming[k]
        extended_naming['company'+k] = "C-"+naming[k]
        extended_naming[k] = naming[k]

    extended_id_naming = {}
    for k in g.nodes():
        name = ": ".join(k.split(": ")[1:])
        if name in extended_naming:
            assert name in extended_naming, name
            extended_id_naming[k] = extended_naming[name]
    print(extended_id_naming)
    
    color_map = compute_color_map(g, get_probs_file(results_file, g, printer))

    reduced_graph = copy.deepcopy(g)
    for s in g:
        for t in g:
            if ("C-"+extended_id_naming[s] == extended_id_naming[t] or "U-"+extended_id_naming[s] == extended_id_naming[t]):
                reduced_graph = nx.contracted_nodes(reduced_graph, s, t, self_loops=False)

    draw_dfg(reduced_graph, "out/greps/fig3.png", names=extended_id_naming, layout = "dot", color_map=color_map)
    
    # Produces Figure 3
    plot_reduction(g, "out/greps/alergia_reduced.png", get_probs_file(results_file, g, printer), layout = "dot")
    
    
    # Constrainted steps and parameterized transitions
    subprocess.run([PRISM_PATH, STORE_PATH+"alergia_reduction_model.prism", QUERY_PATH+"exp_values.props"]) 
    
    steps_max = 20
    max_gas = 45
    min_gas= 16
    stepsize = 2
    
    query = PrismQuery(g, STORE_PATH, "alergia_reduction_model_param.prism", PRISM_PATH)
    results_file = query.query(QUERY_PATH+"pos_alergia.props", 
                            write_attributes=True, write_parameterized=True, envprob=0, 
                            steps_max=10*steps_max, min_gas=-10*min_gas, max_gas=10*max_gas)
    print("Probability under 90% confidence", results_file['q0start'])

    # experiment over gas (m0), steps (m1), and min_gas (m2)
    # Takes some time to execute
    PrismPrinter(g, STORE_PATH, "alergia_reduction_model_param.prism").write_to_prism(write_extended_parameterized=True, write_attributes=True, steps_max=10*steps_max, min_gas=-10*min_gas, max_gas=10*max_gas)
    file_name = OUTPUT_PATH+"bounded_steps_gas_min_gas_greps.txt"
    subprocess.run([PRISM_PATH, STORE_PATH+"alergia_reduction_model_param.prism", 
                    QUERY_PATH+"bounded_props.props",
                    "-const", f'm0=-10:{stepsize}:30,m1=12:{2*stepsize}:36,m2=-70:{stepsize}:-30,', "-exportresults", file_name+":dataframe"])
    plot_fig_4c()
    
    
    # Improvement recommendation ranking
    query = PrismQuery(g, STORE_PATH, "alergia_reduction_model.prism", PRISM_PATH)
    strategy = query.get_strategy(QUERY_PATH+"pos_alergia.props")
    lost_users(g, get_probs_file(results_file, g, printer), transform_strategy(strategy, g, printer))
    
    
    reduced_sankey_diagram(g, get_probs_file(results_file, g, printer))