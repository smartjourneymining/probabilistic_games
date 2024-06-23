global PRISM_PATH
global STORE_PATH
global QUERY_PATH
global OUTPUT_PATH
PRISM_PATH = ""  # path to PRISM-games install
STORE_PATH = "" # path to where generated models can be stored
QUERY_PATH = "" # path to queries
OUTPUT_PATH = "" # path to PRISM-games generated output files

from journepy.src.preprocessing.bpic17 import preprocessed_log
from journepy.src.alergia_utils import convert_utils
from journepy.src.mc_utils.prism_utils import PrismPrinter
from journepy.src.mc_utils.prism_utils import PrismQuery
import probabilistic_game_utils as pgu 
from aalpy.learning_algs import run_Alergia
import pandas as pd
import json
import networkx as nx
import subprocess
import matplotlib.pyplot as plt
import copy
import plotly.graph_objects as go
import os 

def parse(s:str):
    """Helper-function to parse event names to prism accepted format, removes brackets, and whitespaces.

    Args:
        s (str): Event name

    Returns:
        str : Parsed event name. 
    """
    return s.replace('(', ')').replace(')', '').replace(' ', '')

def plot_fig_6(g_before, g_after):
    PrismPrinter(g_before, STORE_PATH, "bpic_17_1_alergia.prism").write_to_prism(write_parameterized=True)
    PrismPrinter(g_after, STORE_PATH, "bpic_17_2_alergia.prism").write_to_prism(write_parameterized=True)
    
    file_name = OUTPUT_PATH+"succ_prop_cond-bpic-17-1.txt"
    subprocess.run([PRISM_PATH, STORE_PATH+"bpic_17_1_alergia.prism",
                    QUERY_PATH+"pos_alergia.props",
                    "-const", "envprob=-0.95:0.05:0.95", "-exportresults", file_name+":dataframe"], stdout=subprocess.DEVNULL)
    df_visual = pd.read_csv(file_name)
    plt.plot(df_visual['envprob'], df_visual['Result'], label = "BPIC'17-1", linewidth=3)

    file_name = OUTPUT_PATH+"succ_prop_cond-bpic-17-2.txt"
    subprocess.run([PRISM_PATH, STORE_PATH+"bpic_17_2_alergia.prism",
                    QUERY_PATH+"pos_alergia.props",
                    "-const", "envprob=-0.95:0.05:0.95", "-exportresults", file_name+":dataframe"], stdout=subprocess.DEVNULL) 

    df_visual = pd.read_csv(file_name)
    plt.plot(df_visual['envprob'], df_visual['Result'], label = "BPIC'17-2", linewidth=3)

    plt.vlines(x=0, ymin=0, ymax = 1, linewidth=2, color = 'grey', linestyles='--')
    plt.text(-1, 0.05, 'Service Provider', fontsize = 18)
    plt.text(0.8, 0.05, 'User', fontsize = 18)
    plt.legend(fontsize=16)
    plt.xlabel("Scaled activity", fontsize=22)
    plt.ylabel("Success probability", fontsize=22)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=20)
    plt.tight_layout()
    plt.savefig("out/bpic17/fig6.png", dpi=300)
    plt.close()
    
def plot_weight_steps_experiment(g_before, model_name_before, g_after, model_name_after):
    # run gas upper and lower bound under limited steps
    PrismPrinter(g_before, STORE_PATH, "bpic_17_1_alergia.prism").write_to_prism(write_extended_parameterized=True)
    file_name_pos = OUTPUT_PATH+"steps_gas_pos_bound-bpic-17-1.txt"
    subprocess.run([PRISM_PATH, model_name_before, 
                    QUERY_PATH+"reward_props.props", "-prop", "3",
                    "-const", "m0=0,m1=0:1:80,m2=0,", "-exportresults", file_name_pos+":dataframe"], stdout=subprocess.DEVNULL) 
    file_name_neg = OUTPUT_PATH+"steps_gas_neg_bound-bpic-17-1.txt"
    subprocess.run([PRISM_PATH, model_name_before, 
                    QUERY_PATH+"reward_props.props", "-prop", "4",
                    "-const", "m0=0,m1=0:1:80,m2=0,", "-exportresults", file_name_neg+":dataframe"], stdout=subprocess.DEVNULL) 

    df_visual = pd.read_csv(file_name_pos)
    plt.plot(df_visual['m1']/4, df_visual['Result'], label="BPIC'17-1 max pos", c = "blue")
    df_visual = pd.read_csv(file_name_neg)
    plt.plot(df_visual['m1']/4, df_visual['Result'], label="BPIC'17-1 min neg", c = "orange")

    PrismPrinter(g_after, STORE_PATH, "bpic_17_2_alergia.prism").write_to_prism(write_extended_parameterized=True)
    file_name_pos = OUTPUT_PATH+"steps_gas_pos_bound-bpic-17-2.txt"
    subprocess.run([PRISM_PATH, model_name_after, 
                    QUERY_PATH+"reward_props.props", "-prop", "3",
                    "-const", "m0=0,m1=0:1:80,m2=0,", "-exportresults", file_name_pos+":dataframe"], stdout=subprocess.DEVNULL) 
    file_name_neg = OUTPUT_PATH+"steps_gas_neg_bound-bpic-17-2.txt"
    subprocess.run([PRISM_PATH, model_name_after, 
                    QUERY_PATH+"reward_props.props", "-prop", "4",
                    "-const", "m0=0,m1=0:1:80,m2=0,", "-exportresults", file_name_neg+":dataframe"], stdout=subprocess.DEVNULL) 

    df_visual = pd.read_csv(file_name_pos)
    plt.plot(df_visual['m1']/4, df_visual['Result'], label="BPIC'17-2 max pos", c = "blue", linestyle="dashed")
    df_visual = pd.read_csv(file_name_neg)
    plt.plot(df_visual['m1']/4, df_visual['Result'], label="BPIC'17-2 min neg", c = "orange", linestyle="dashed")

    plt.legend(fontsize=14)
    plt.xlabel("Steps S", fontsize=18)
    plt.ylabel("Accumulated weight", fontsize=18)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=18)
    plt.tight_layout()
    plt.savefig("out/bpic17/bpic_17_steps.png", dpi=300)
    plt.close()

# bounded reachability plot for BPIC'17
def plot_df(df, dashed = False):
    df_visual_grouped = df.groupby(['m0','m2'])
    result_dict = {}
    for g in df_visual_grouped.groups.keys():
        r = tuple([round(h,2) for h in df_visual_grouped.get_group(g)['Result'].values])
        if r in result_dict:
            result_dict[r].append(g)
        else:
            result_dict[r] = [g]

    for k in result_dict:
        g = result_dict[k]
        max_m0 = max(h[0] for h in result_dict[k])
        max_m2 = max(h[1] for h in result_dict[k])
        print(k, result_dict[k])
        print(max_m0, max_m2)
        assert((max_m0, max_m2) in result_dict[k])
        if len(set(k)) != 1:
            if dashed:
                plt.plot(df_visual_grouped.get_group((max_m0, max_m2))['m1'], k, label = str((max_m0, max_m2)), linestyle="dashed")
            else:
                plt.plot(df_visual_grouped.get_group((max_m0, max_m2))['m1'], k, label = str((max_m0, max_m2)))

def transform_strategy(strategy, g, printer):
    """ 
    Adjusts the strategy naming from PRISM node naming to original node naming in g
    """
    isomorphism = nx.vf2pp_isomorphism(printer.g, g, node_label=None)
    strategy_isomorphism = {k[len(isomorphism[k].split(": ")[0]):] : ": ".join(isomorphism[k].split(": ")[1:]) for k in isomorphism}
    parsed_strategy = {isomorphism[r] : strategy_isomorphism[strategy[r]] if strategy[r] not in ["env", "user", "company", "sent", "call"] else strategy[r] for r in strategy}
    assert(len(parsed_strategy)==len(strategy))
    return parsed_strategy

def lost_users(g, results_file, strategy):
    lost_users_dict = {}
    total_lost_users_dict = {}
    for s in strategy:
        assert s in g
        next_states = [t for t in g[s] if g[s][t]['action'] == strategy[s]]

        assert(next_states)
        total_lost_users = 0
        for t in next_states:
            action_outcome_cost = len(g[s][t]['trace_indices']) * abs(round(results_file[s],4)-round(results_file[t],4))
            lost_users_dict[(s,t)] = action_outcome_cost

            total_lost_users += action_outcome_cost
                
        if total_lost_users != 0:
            total_lost_users_dict[s] = total_lost_users
    total = sum(total_lost_users_dict.values())
    print("total", total)
    for k in (sorted( ((v,k) for k,v in total_lost_users_dict.items()), reverse=True)):
        print(k, k[0]/total)
    print(sorted(lost_users_dict.values(), reverse=True))
    return lost_users_dict

def flatten(d, l):
    """
    Helper function to get contracted nodes dict.
    """
    for e in d:
        l.append(e)
        if 'contraction' in d[e]:
                flatten(d[e]['contraction'], l)
    return l

def reduced_sankey_diagram(g, results_file, strategy, lost_users_dict, name):
    g = copy.deepcopy(g)
    print(strategy.keys())
    g = g.subgraph(strategy.keys()) # consider only nodes seen in strategy
    g = pgu.reduce_graph(g, results_file, 4)
    color_map = pgu.compute_color_map(g, results_file)

    d = nx.get_node_attributes(g, "contraction")
    reduction_mapping = {}
    for k in d:
        reduction_mapping[k] = flatten(d[k], [])
    print("reduction mapping", reduction_mapping)
                            
    node_list = list(g.nodes())
    print("Node_list", node_list)
    print(len(node_list))
    node_dict = {node_list[i] : i for i in range(len(node_list))}

    print("lost users dict", lost_users_dict)

    edge_list = []
    value_list = []
    for e in [e for e in lost_users_dict if lost_users_dict[e] != 0]:
        if e[0] in node_list:
                s = e[0]
        else:
                s = None
                for h in reduction_mapping:
                    if e[0] in reduction_mapping[h]:
                            s = h
                assert s
        if e[1] in node_list:
                t = e[1]
        else:
                t = None
                for h in reduction_mapping:
                    if e[1] in reduction_mapping[h]:
                            t = h
                assert t
        edge_list.append((s,t))
        value_list.append(lost_users_dict[e])

    labels = []
    
    # build label list with right states
    if name == "fig7-1":
        for s in node_list:
            if s not in reduction_mapping or s in lost_users_dict:
                    label = str(s).split(": ")[1].replace("LONG", "").replace("SHORT", "").replace("W_", "").replace("Callincompletefiles", 'Call incomplete files')
                    labels.append(label)
            elif any(["start" in h for h in reduction_mapping[s]]):
                    labels.append("Start")
            elif any(["q30" in h for h in reduction_mapping[s]]) or "q30" in s:
                    labels.append("Sent (mail and online)")
            elif any(["q23" in h for h in reduction_mapping[s]]) or "q23" in s:
                    labels.append("Create Offer 0")
            elif any(["q13" in h for h in reduction_mapping[s]]) or "q13" in s:
                    labels.append("Create Offer 0")
            elif results_file[s] == 1:
                    labels.append("succ")
            elif results_file[s] == 0:
                    labels.append("unsucc")
            else:
                    # s is in reduction_mapping and not in lost_users_dict
                    labels.append("")
                    #labels.append("C"+str(s).split(": ")[0])
    if name == "fig7-2":
        for s in node_list:
            if s not in reduction_mapping or s in lost_users_dict:
                  label = str(s).split(": ")[1].replace("LONG", "").replace("SHORT", "").replace("W_", "").replace("Callincompletefiles", 'Call incomplete files')
                  labels.append(label)
            elif any(["start" in h for h in reduction_mapping[s]]):
                  labels.append("Start")
            elif any(["q64" in h for h in reduction_mapping[s]]) or "q64" in s:
                  labels.append("Sent (mail and online)")
            elif any(["q23" in h for h in reduction_mapping[s]]) or "q23" in s:
                  labels.append("Create Offer 0")
            elif any(["q13" in h for h in reduction_mapping[s]]) or "q13" in s:
                  labels.append("Create Offer 0")
            elif any(["q42" in h for h in reduction_mapping[s]]) or "q42" in s:
                  labels.append("Create Offer 1")
            elif any(["q32" in h for h in reduction_mapping[s]]) or "q42" in s:
                  labels.append("Create Offer 2")
            elif results_file[s] == 1:
                  labels.append("succ")
            elif results_file[s] == 0:
                  labels.append("unsucc")
            else:
                  # s is in reduction_mapping and not in lost_users_dict
                  labels.append("")
                  #labels.append("C"+str(s).split(": ")[0])


    fig = go.Figure(data=[go.Sankey(
    node = dict(
        pad = 5 if name == "before" else 10,
        thickness = 30,
        line = dict(color = "black", width = 0.5),
        label = labels,
        color = [color_map[s] for s in node_list],
        align = "right"
    ),
    link = dict(
        source = [node_dict[e[0]] for e in edge_list],
        target = [node_dict[e[1]] for e in edge_list],
        value = value_list,
        arrowlen=15,
    ))])
    fig.update_layout(
    font=dict(size = 30)
)
    
    fig.to_image(format = "png", engine = "kaleido")
    fig.write_image(f"out/bpic17/{name}.png")
    fig.write_html(f"out/bpic17/{name}.html")

def get_data(actors, filtered_log_before_activities, filtered_log_after_activities):
    # build action mapping: assigns each event to an actor
    actions_to_activities = {}
    actions_observed = set()
    for trace in filtered_log_before_activities:
        actions_observed.update(trace)

    actors_observed_actions = {}
    for action in actions_observed:
        contained = [a for a in actors if a in action]
        assert len(contained) == 1# each once - skip enumerating
        if actors[contained[0]] == "company": # events where company is NOT deterministic
            if 'O_Sentonlineonly' in action or 'O_Sentmailandonline' in action: 
                # sent online vs sent mail seems to be critical interaction - if aggregated together and diff. from others- 90%
                actions_to_activities[action] = "sent"
            elif 'O_CreateOffer0' in action:
                actions_to_activities[action] = action
            elif 'O_CreateOffer1' in action:
                actions_to_activities[action] = action
            elif 'O_CreateOffer2' in action:
                actions_to_activities[action] = action
            elif 'O_CreateOffer3' in action:
                actions_to_activities[action] = action
            elif 'O_CreateOffer4' in action:
                actions_to_activities[action] = action
            elif 'O_CreateOffer5' in action:
                actions_to_activities[action] = action
            elif 'A_Denied' in action:
                actions_to_activities[action] = action
            elif 'A_Incomplete' in action:
                actions_to_activities[action] = action
            elif 'A_Validating' in action:
                actions_to_activities[action] = action
            else:
                actions_to_activities[action] = action
        else:
            if "A_Submitted" in action: # event where user is deterministic
                actions_to_activities[action] = action
            elif "Callafteroffers" in action: # event where user is deterministic
                actions_to_activities[action] = "call"
            elif "Callincomplete" in action: # event where user is deterministic
                actions_to_activities[action] = "call"
            else: # includes "negative" action
                actions_to_activities[action] = "user"

        actors_observed_actions[action] = actors[contained[0]]

    actors_observed_actions['offer'] = 'company'
    actors_observed_actions['company'] = 'company'
    actors_observed_actions['user'] = 'customer'
    actors_observed_actions['sent'] = 'company'
    actors_observed_actions['offer_response'] = 'company'
    actors_observed_actions['callafter'] = 'customer'
    actors_observed_actions['callincomplete'] = 'customer'
    
    # add activities
    data_before = [[(actions_to_activities[t[i]], t[i]) for i in range(1, len(t))] for t in filtered_log_before_activities]
    for d in data_before:
        d.insert(0, 'start')

    data_after = [[(actions_to_activities[t[i]], t[i]) for i in range(1, len(t))] for t in filtered_log_after_activities]
    for d in data_after:
        d.insert(0, 'start')
        
        # quantify environment -> becomes MDP
    data_before_environment = []
    for trace in data_before:
        current = [trace[0]]
        for i in range(1, len(trace)):
            e = trace[i]
            previous_state = "start" if i == 1 else trace[i-1][1]
            
            # encode decision in one step
            current.append(('env', actors_observed_actions[e[1]] + previous_state))
            current.append(e)
        data_before_environment.append(current)

    data_after_environment = []
    for trace in data_after:
        current = [trace[0]]
        for i in range(1, len(trace)):
            e = trace[i]
            previous_state = "start" if i == 1 else trace[i-1][1]
            
            # encode decision in one step
            current.append(('env', actors_observed_actions[e[1]] + previous_state))
            current.append(e)
        data_after_environment.append(current)
        
    return data_before_environment, data_after_environment, actors_observed_actions

def constrained_experiment(short_execution, g_before, g_after):    
    print("### BPIC'17-1 Expected Values ###")
    query = PrismQuery(g_before, STORE_PATH, "bpic_17_1_alergia.prism", PRISM_PATH)
    results_file = query.query(QUERY_PATH+"exp_values:max_steps.props", write_parameterized=True)
    print("E(max(steps))", results_file['q0start'])
    results_file = query.query(QUERY_PATH+"exp_values:max_gas_neg.props", write_parameterized=True)
    print("E(max(gas_neg))", results_file['q0start']) 
    results_file = query.query(QUERY_PATH+"exp_values:max_gas_pos.props", write_parameterized=True)
    print("E(max(gas_pos))", results_file['q0start']) 
    print()
    
    print("### BPIC'17-2 Expected Values ###")
    query = PrismQuery(g_after, STORE_PATH, "bpic_17_2_alergia.prism", PRISM_PATH)
    results_file = query.query(QUERY_PATH+"exp_values:max_steps.props", write_parameterized=True)
    print("E(max(steps))", results_file['q0start'])
    results_file = query.query(QUERY_PATH+"exp_values:max_gas_neg.props", write_parameterized=True)
    print("E(max(gas_neg))", results_file['q0start']) 
    results_file = query.query(QUERY_PATH+"exp_values:max_gas_pos.props", write_parameterized=True)
    print("E(max(gas_pos))", results_file['q0start']) 
    print()
    
    steps_max_before = 40#int(62/2)
    max_gas_before = int(68/2)
    min_gas_before = 50 #int(1080/10)

    steps_max_after = 40 #int(76/2)
    max_gas_after = int(68/2)
    min_gas_after = 50 #int(1323/17)
    stepsize = 2 if short_execution else 1
    
    # Query Q1 on bounded model BPIC'17-1
    query = PrismQuery(g_before, STORE_PATH, "bpic_17_1_alergia_param.prism", PRISM_PATH)
    results_file = query.query(QUERY_PATH+"pos_alergia.props", 
                            write_attributes=True, write_parameterized=True, envprob=0, 
                            steps_max=steps_max_before, min_gas=-min_gas_before, max_gas=max_gas_before)
    print("Q1 under adjusted bounds for BPIC'17-1", results_file['q0start'])
    
    # Query Q1 on bounded model BPIC'17-2
    query = PrismQuery(g_after, STORE_PATH, "bpic_17_2_alergia_param.prism", PRISM_PATH)
    results_file = query.query(QUERY_PATH+"pos_alergia.props", 
                            write_attributes=True, write_parameterized=True, envprob=0, 
                            steps_max=steps_max_after, min_gas=-min_gas_after, max_gas=max_gas_after)
    print("Q1 under adjusted bounds for BPIC'17-2", results_file['q0start'])
    
    # experiment over gas (m0), steps (m1), and min_gas (m2)
    # Takes some time to execute
    PrismPrinter(g_before, STORE_PATH, "bpic_17_1_alergia_param.prism").write_to_prism(write_extended_parameterized=True, 
                                                                                    write_attributes=True, steps_max=steps_max_before, min_gas=-min_gas_before, max_gas=max_gas_before)
    file_name = OUTPUT_PATH+"bounded_steps_gas_min_gas_bpic_17-1.txt"
    subprocess.run([PRISM_PATH, STORE_PATH+"bpic_17_1_alergia_param.prism", 
                    QUERY_PATH+"bounded_props.props",
                    "-const", f"m0=32:{1*stepsize}:36,m1=12:{10*stepsize}:32,m2=-35:{5*stepsize}:-15,", "-exportresults", file_name+":dataframe", '-javamaxmem', '20g'], stdout=subprocess.DEVNULL)
    
    PrismPrinter(g_after, STORE_PATH, "bpic_17_2_alergia_param.prism").write_to_prism(write_extended_parameterized=True,
                                                                                  write_attributes=True, steps_max=steps_max_after, min_gas=-min_gas_after, max_gas=max_gas_after)
    file_name = OUTPUT_PATH+"bounded_steps_gas_min_gas_bpic_17-2.txt"
    subprocess.run([PRISM_PATH, STORE_PATH+"bpic_17_2_alergia_param.prism", 
                    QUERY_PATH+"bounded_props.props",
                    "-const", f"m0=32:{1*stepsize}:36,m1=12:{10*stepsize}:32,m2=-35:{5*stepsize}:-15,", "-exportresults", file_name+":dataframe", '-javamaxmem', '20g'], stdout=subprocess.DEVNULL)
    
    file_name = OUTPUT_PATH+"bounded_steps_gas_min_gas_bpic_17-1.txt"
    df_visual = pd.read_csv(file_name)
    plot_df(df_visual)

    file_name = OUTPUT_PATH+"bounded_steps_gas_min_gas_bpic_17-2.txt"
    df_visual = pd.read_csv(file_name)
    plot_df(df_visual, dashed=True)

    plt.legend(fontsize=10)
    plt.xlabel("Steps", fontsize=18)
    plt.ylabel("Success probability", fontsize=18)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.tight_layout()
    plt.savefig("out/bpic17/bpic_bounded.png", dpi=300)
    plt.close()

def write_table2(g_before, g_after):
    printer_before = PrismPrinter(g_before, STORE_PATH, "bpic_17_1_alergia.prism")
    printer_before.write_to_prism()
    printer_after = PrismPrinter(g_after, STORE_PATH, "bpic_17_2_alergia.prism")
    printer_after.write_to_prism()

    # read content so far
    if os.path.isfile('out/table2.md'):
        with open("out/table2.md", "r") as file:
            lines = [line.rstrip() for line in file]
    else:
        lines = []
    # append new table
    with open("out/table2.md", "w+") as f:
        if lines == []:
            f.write("|Name|BPIC'17-1|\n")
            f.write("|---|---|\n")
        else:
            assert len(lines)==5
            f.write(lines[0]+"BPIC'17-1|\n")
            f.write(lines[1]+"---|\n")
        # Query Q1 from Table 2
        print("### BPIC'17-1 Table 2 ###")
        query_before = PrismQuery(g_before, STORE_PATH, "bpic_17_1_alergia.prism", PRISM_PATH)
        results_file = query_before.query(QUERY_PATH+"pos_alergia.props", write_parameterized=True)
        print("Q1", results_file['q0start'])
        # Q2
        results_file = query_before.query(QUERY_PATH+"mc_runs:min_gas_neg_user_provider.props", write_parameterized=True)
        print("Q2", results_file['q0start'])
        if lines == []:
            f.write("|Q2|"+str(round(results_file['q0start'],2))+"|\n")
        else:
            f.write(lines[2]+str(round(results_file['q0start'],2))+"|\n")
        # Q3
        results_file = query_before.query(QUERY_PATH+"mc_runs:min_gas_neg_provider.props", write_parameterized=True)
        print("Q3", results_file['q0start'])
        if lines == []:
            f.write("|Q3|"+str(round(results_file['q0start'],2))+"|\n")
        else:
            f.write(lines[3]+str(round(results_file['q0start'],2))+"|\n")
        # Q4
        results_file = query_before.query(QUERY_PATH+"mc_runs:max_gas_pos_provider.props", write_parameterized=True)
        print("Q4", results_file['q0start'])
        if lines == []:
            f.write("|Q4|"+str(round(results_file['q0start'],2))+"|\n")
        else:
            f.write(lines[4]+str(round(results_file['q0start'],2))+"|\n")
        print()
    
    # read content from bpic'17-1
    with open("out/table2.md", "r") as file:
        lines = [line.rstrip() for line in file]
    assert len(lines) == 5, print(len(lines))
    with open("out/table2.md", "w+") as f:
        f.write(lines[0]+"BPIC'17-2|\n")
        f.write(lines[1]+"---|\n")
        # Query Q1 from Table 2
        print("### BPIC'17-2 Table 2 ###")
        query_after = PrismQuery(g_after, STORE_PATH, "bpic_17_2_alergia.prism", PRISM_PATH)
        results_file = query_after.query(QUERY_PATH+"pos_alergia.props", write_parameterized=True)
        print("Q1", results_file['q0start'])
        # Q2
        results_file = query_after.query(QUERY_PATH+"mc_runs:min_gas_neg_user_provider.props", write_parameterized=True)
        print("Q2", results_file['q0start'])
        f.write(lines[2]+str(round(results_file['q0start'],2))+"|\n")
        # Q3
        results_file = query_after.query(QUERY_PATH+"mc_runs:min_gas_neg_provider.props", write_parameterized=True)
        print("Q3", results_file['q0start'])
        f.write(lines[3]+str(round(results_file['q0start'],2))+"|\n")
        # Q4
        results_file = query_after.query(QUERY_PATH+"mc_runs:max_gas_pos_provider.props", write_parameterized=True)
        print("Q4", results_file['q0start']) 
        f.write(lines[4]+str(round(results_file['q0start'],2))+"|\n")
        print()     
        
        
def main(pPRISM_PATH, pSTORE_PATH, pQUERY_PATH, pOUTPUT_PATH, DATA_PATH, short_execution = True):
    global PRISM_PATH
    global STORE_PATH
    global QUERY_PATH
    global OUTPUT_PATH
    PRISM_PATH = pPRISM_PATH
    STORE_PATH = pSTORE_PATH
    QUERY_PATH = pQUERY_PATH
    OUTPUT_PATH = pOUTPUT_PATH
    os.makedirs("out/bpic17/", mode=0o777, exist_ok=True)
    
    filtered_log_before, filtered_log_after = preprocessed_log(DATA_PATH+'BPI Challenge 2017.xes') # uses common preprocessing
    print(len(filtered_log_before))
    print(len(filtered_log_after))
    
    # change from xes format
    filtered_log_before_activities = [[parse(e['concept:name']) for e in t] for t in filtered_log_before]
    filtered_log_after_activities = [[parse(e['concept:name']) for e in t] for t in filtered_log_after]
    
    # load actor mapping: maps events to an actor (service provider or user)
    with open(DATA_PATH+'activities.xml') as f:
        data = f.read()
    actors = json.loads(data)
    actors = {parse(k) : parse(actors[k]) for k in actors}
        
    data_before_environment, data_after_environment, actors_observed_actions = get_data(actors, filtered_log_before_activities, filtered_log_after_activities)
    
    model_before_environment = run_Alergia(data_before_environment, automaton_type='mdp', eps=0.8, print_info=True) # 0.1 plot interesting, 0.8 is confirms knowledge, 1.2 : plot align, 2 grows stronger, 1.8 : 2 is worse, grows stronger, gas is bit more interesting
    model_after_environment = run_Alergia(data_after_environment, automaton_type='mdp', eps=0.8, print_info=True)
    
    # convert mdp to game
    g_before = convert_utils.mdp_to_nx(model_before_environment, actors_observed_actions)
    g_after = convert_utils.mdp_to_nx(model_after_environment, actors_observed_actions)
    
    g_before = pgu.add_gas_and_user_count(g_before, data_before_environment)
    g_after = pgu.add_gas_and_user_count(g_after, data_after_environment)
    
    pgu.assert_no_det_cycle(g_before)
    pgu.assert_no_det_cycle(g_after)
    
    # model checking stochastic user journey games
    
    write_table2(g_before, g_after)    
    
    # run Activity experiment and produce Fig. 7
    # remove "stdout=subprocess.DEVNULL" to print output again
    plot_fig_6(g_before, g_after)
    
    model_name_before = STORE_PATH+"bpic_17_1_alergia.prism"
    model_name_after = STORE_PATH+"bpic_17_2_alergia.prism"
    plot_weight_steps_experiment(g_before, model_name_before, g_after, model_name_after)
    
    # Induced and reduced model
    printer_before = PrismPrinter(g_before, STORE_PATH, "bpic_17_1_alergia.prism")
    printer_before.write_to_prism()
    printer_after = PrismPrinter(g_after, STORE_PATH, "bpic_17_2_alergia.prism")
    printer_after.write_to_prism()
    query_before = PrismQuery(g_before, STORE_PATH, "bpic_17_1_alergia.prism", PRISM_PATH)
    results_file_before = query_before.query(QUERY_PATH+"pos_alergia.props", write_parameterized=True)
    pgu.plot_reduction(g_before, "out/bpic17/before_reduced.png", pgu.get_probs_file(results_file_before, g_before, printer_before), 4, layout = "dot")
    query_after = PrismQuery(g_after, STORE_PATH, "bpic_17_2_alergia.prism", PRISM_PATH)
    results_file_after = query_after.query(QUERY_PATH+"pos_alergia.props", write_parameterized=True)
    pgu.plot_reduction(g_after, "out/bpic17/after_reduced.png", pgu.get_probs_file(results_file_after, g_after, printer_after), 4, layout = "dot")
    
    
    # constrained steps and parameterized transitions
    constrained_experiment(short_execution, g_before, g_after)
    
    print("Strategy before skips:")
    strategy_before = query_before.get_strategy(QUERY_PATH+"pos_alergia.props")
    print()
    print("Strategy after skips:")
    strategy_after = query_after.get_strategy(QUERY_PATH+"pos_alergia.props")
    print()
    
    lost_users_dict_before = lost_users(g_before, pgu.get_probs_file(results_file_before, g_before, printer_before), transform_strategy(strategy_before, g_before, printer_before))
    reduced_sankey_diagram(g_before, pgu.get_probs_file(results_file_before, g_before, printer_before), transform_strategy(strategy_before, g_before, printer_before), lost_users_dict_before, "fig7-1")
    
    lost_users_dict_after = lost_users(g_after, pgu.get_probs_file(results_file_after, g_after, printer_after), transform_strategy(strategy_after, g_after, printer_after))
    reduced_sankey_diagram(g_after, pgu.get_probs_file(results_file_after, g_after, printer_after),  transform_strategy(strategy_after, g_after, printer_after), lost_users_dict_after, "fig7-2")