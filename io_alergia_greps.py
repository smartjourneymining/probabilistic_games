from journepy.src.preprocessing.greps import preprocessed_log
from journepy.src.alergia_utils import convert_utils
from journepy.src.mc_utils.prism_utils import PrismPrinter
from journepy.src.mc_utils.prism_utils import PrismQuery
import probabilistic_game_utils as pgu 

from aalpy.learning_algs import run_Alergia
import pandas as pd
import matplotlib.pyplot as plt
import json
import networkx as nx
import subprocess
import copy
import plotly.graph_objects as go
import os 
        
global PRISM_PATH
global STORE_PATH
global QUERY_PATH
global OUTPUT_PATH
PRISM_PATH = ""  # path to PRISM-games install
STORE_PATH = "" # path to where generated models can be stored
QUERY_PATH = "" # path to queries
OUTPUT_PATH = "" # path to PRISM-games generated output files

def plot_fig_4a(g):
    # remove "stdout=subprocess.DEVNULL" to print output again
    PrismPrinter(g, STORE_PATH, "alergia_reduction_model.prism").write_to_prism(write_parameterized=True)
    file_name = OUTPUT_PATH+"succ_prop_cond.txt"
    subprocess.run([PRISM_PATH, STORE_PATH+"alergia_reduction_model.prism", 
                    QUERY_PATH+"pos_alergia.props", 
                    "-const", "envprob=-0.95:0.05:0.95", "-exportresults", file_name+":dataframe"], stdout=subprocess.DEVNULL) 
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
    
def plot_fig_4b(g):
    # produces Fig. 4b
    PrismPrinter(g, STORE_PATH, "alergia_reduction_model.prism").write_to_prism(write_extended_parameterized=True)
    file_name = OUTPUT_PATH+"steps_gas_pos_bound.txt"
    subprocess.run([PRISM_PATH, STORE_PATH+"alergia_reduction_model.prism", 
                    QUERY_PATH+"reward_props.props", "-prop", "3",
                    "-const", "m0=0,m1=0:1:140,m2=0,", "-exportresults", file_name+":dataframe"], stdout=subprocess.DEVNULL) 
    file_name = OUTPUT_PATH+"steps_gas_neg_bound.txt"
    subprocess.run([PRISM_PATH, STORE_PATH+"alergia_reduction_model.prism", 
                    QUERY_PATH+"reward_props.props", "-prop", "4",
                    "-const", "m0=0,m1=0:1:140,m2=0,", "-exportresults", file_name+":dataframe"], stdout=subprocess.DEVNULL) 

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
    
def plot_fig_4c(short_execution, g):
    # plot Fig. 4c
    print("### Greps Expected Values ###")
    query = PrismQuery(g, STORE_PATH, "alergia_reduction_model.prism", PRISM_PATH)
    results_file = query.query(QUERY_PATH+"exp_values:max_steps.props", write_parameterized=True)
    print("E(max(steps))", results_file['q0start'])
    results_file = query.query(QUERY_PATH+"exp_values:max_gas_neg.props", write_parameterized=True)
    print("E(max(gas_neg))", results_file['q0start']) 
    results_file = query.query(QUERY_PATH+"exp_values:max_gas_pos.props", write_parameterized=True)
    print("E(max(gas_pos))", results_file['q0start']) 
    print()
    
    steps_max = 20
    max_gas = 45
    min_gas= 16
    stepsize = 10 if short_execution else 2
    
    query = PrismQuery(g, STORE_PATH, "alergia_reduction_model_param.prism", PRISM_PATH)
    results_file = query.query(QUERY_PATH+"pos_alergia.props", 
                            write_attributes=True, write_parameterized=True, envprob=0, 
                            steps_max=10*steps_max, min_gas=-10*min_gas, max_gas=10*max_gas)
    print("Probability under 90% confidence", results_file['q0start'])

    # experiment over gas (m0), steps (m1), and min_gas (m2)
    # Takes some time to execute
    multiplier = 4 if short_execution else 10
    PrismPrinter(g, STORE_PATH, "alergia_reduction_model_param.prism").write_to_prism(write_extended_parameterized=True, write_attributes=True, steps_max=multiplier*steps_max, min_gas=-multiplier*min_gas, max_gas=multiplier*max_gas)
    file_name = OUTPUT_PATH+"bounded_steps_gas_min_gas_greps.txt"
    subprocess.run([PRISM_PATH, STORE_PATH+"alergia_reduction_model_param.prism", 
                    QUERY_PATH+"bounded_props.props",
                    "-const", f'm0=-10:{stepsize}:30,m1=12:{2*stepsize}:36,m2=-70:{3*stepsize}:-30,', "-exportresults", file_name+":dataframe"], stdout=subprocess.DEVNULL)
 
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
    g = pgu.reduce_graph(g, results_file, 2)
    color_map = pgu.compute_color_map(g, results_file)


    node_list = list(g.nodes())
    node_dict = {node_list[i] : i for i in range(len(node_list))}
    edge_list = g.edges()
    print(edge_list)

    print([len(g[e[0]][e[1]]['trace_indices'])  * abs(round(results_file[e[0]],5)-round(results_file[e[1]],5)) for e in edge_list])
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
    ))])
    fig.update_layout(
    font=dict(size = 40)
    )
    fig.to_image(format = "png", engine = "kaleido")
    fig.write_image("out/greps/fig5.png")
    fig.write_html("out/greps/fig5.html")

def get_data(actors, filtered_log):
    actions_to_activities = {}
    for a in actors:
        if actors[a] == "company":
            if a in ['vpcAssignInstance', 'Give feedback 0', 'Results automatically shared', 'waitingForActivityReport']:
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
    
    filtered_log_activities = [[e['concept:name'] for e in t] for t in filtered_log]
    data = [[(actions_to_activities[t[i]], t[i]) for i in range(1, len(t))] for t in filtered_log_activities]
    for d in data:
        d.insert(0, 'start')
    
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

    return data_environment
        
def compute_extended_id_naming(g):
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
            
    return extended_id_naming
 
def main(pPRISM_PATH, pSTORE_PATH, pQUERY_PATH, pOUTPUT_PATH, short_execution = True):
    global PRISM_PATH
    global STORE_PATH
    global QUERY_PATH
    global OUTPUT_PATH
    PRISM_PATH = pPRISM_PATH
    STORE_PATH = pSTORE_PATH
    QUERY_PATH = pQUERY_PATH
    OUTPUT_PATH = pOUTPUT_PATH
    os.makedirs("out/greps/", mode=0o777)
    
    print("current path", PRISM_PATH)
    
    # load files
    filtered_log = preprocessed_log("data/data.csv", include_loggin=False) # also discards task-event log-in
    
    # load actor mapping: maps events to an actor (service provider or user)
    with open('data/activities_greps.xml') as f:
        data = f.read()
    actors = json.loads(data)
    
    data_environment = get_data(actors, filtered_log)
    
    model_environment = run_Alergia(data_environment, automaton_type='mdp', eps=0.1, print_info=True)
    
    # Extend to Stochastic User Journey Game
    g = convert_utils.mdp_to_nx(model_environment, actors)
    # users can decide to "do nothing"
    g = pgu.add_neutral_user_transition(g)
    g = pgu.add_gas_and_user_count(g, data_environment, greps_values=True)
    pgu.assert_no_det_cycle(g)
    
    
    # MODEL CHECKING # 
    printer = PrismPrinter(g, STORE_PATH, "alergia_reduction_model.prism")
    printer.write_to_prism()
    
    # Query Q1 from Table 1
    print("### Greps Table 1 ###")
    query = PrismQuery(g, STORE_PATH, "alergia_reduction_model.prism", PRISM_PATH)
    results_file = query.query(QUERY_PATH+"pos_alergia.props", write_parameterized=True)
    print("Q1", results_file['q0start'])
    # Q2
    results_file = query.query(QUERY_PATH+"mc_runs:min_gas_neg_user_provider.props", write_parameterized=True)
    print("Q2", results_file['q0start'])
    # Q3
    results_file = query.query(QUERY_PATH+"mc_runs:min_gas_neg_provider.props", write_parameterized=True)
    print("Q3", results_file['q0start'])
    # Q4
    results_file = query.query(QUERY_PATH+"mc_runs:max_gas_pos_provider.props", write_parameterized=True)
    print("Q4", results_file['q0start']) 
    print()
    
    # run Activity experiment
    # produces Fig. 4a
    plot_fig_4a(g)
    
    # run gas upper and lower bound under limited steps
    plot_fig_4b(g)
    
    query = PrismQuery(g, STORE_PATH, "alergia_reduction_model.prism", PRISM_PATH)
    results_file = query.query(QUERY_PATH+"pos_alergia.props", write_parameterized=True)
    extended_id_naming = compute_extended_id_naming(g)
    
    # Produces Figure 3 and reduced version
    reduced_graph = copy.deepcopy(g)
    for s in g:
        for t in g:
            if ("C-"+extended_id_naming[s] == extended_id_naming[t] or "U-"+extended_id_naming[s] == extended_id_naming[t]):
                reduced_graph = nx.contracted_nodes(reduced_graph, s, t, self_loops=False)
    color_map = pgu.compute_color_map(g, pgu.get_probs_file(results_file, g, printer))
    pgu.draw_dfg(reduced_graph, "out/greps/fig3.png", names=extended_id_naming, layout = "dot", color_map=color_map)
    pgu.plot_reduction(g, "out/greps/alergia_reduced.png", pgu.get_probs_file(results_file, g, printer), 2, layout = "dot")
    
    # Constrained steps and parameterized transitions
    plot_fig_4c(short_execution, g)
    
    # Improvement recommendation ranking
    query = PrismQuery(g, STORE_PATH, "alergia_reduction_model.prism", PRISM_PATH)
    strategy = query.get_strategy(QUERY_PATH+"pos_alergia.props")
    lost_users(g, pgu.get_probs_file(results_file, g, printer), transform_strategy(strategy, g, printer))
    reduced_sankey_diagram(g, pgu.get_probs_file(results_file, g, printer))
    