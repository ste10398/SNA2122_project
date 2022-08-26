from collections import Counter, defaultdict
from email import header
import pickle
import random
from re import TEMPLATE, template
from tabnanny import verbose
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import csv
from tqdm import tqdm
import itertools
import plotly.offline as py
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
import plotly.io as pio
import re
from plotly.subplots import make_subplots
from PIL import Image

#HERE WE PRE-PROCESS THE DATES
hero_net_df = pd.read_csv('./data/hero-network.csv')
edge_df = pd.read_csv('./data/edges.csv')
#removing redundant values
for c in ['hero1', 'hero2']:
    hero_net_df[c] = hero_net_df[c].apply(lambda x : x[:20].split("/")[0])
edge_df['hero'] = edge_df['hero'].apply(lambda x : x[:20].split("/")[0])

HERO_COLOR = {
'CAPTAIN AMERICA':'darkblue',
'HUMAN TORCH':'orange',
'SPIDER-MAN':'darkred',
'WOLVERINE':'yellow',
'DR. STRANGE':'purple',
'THOR' : 'lightblue'
}

#function to create the network based on the number of heroes you want to enter
#return marvel_net, appto_df
def def_net(n):
    topn_hero = edge_df.groupby(['hero'])[['comic']].count().sort_values(by=['comic'], ascending=False).head(n).index

    h1_ = []; h2_ = []; cnt_ = []
    for comb in list(itertools.combinations(topn_hero, 2)):    
        temp1 = set(edge_df[edge_df['hero']==comb[0]]['comic'])
        temp2 = set(edge_df[edge_df['hero']==comb[1]]['comic'])
        cnt = len(temp1.intersection(temp2)) # Appear Together    
        h1_.append(comb[0]); h2_.append(comb[1]); cnt_.append(cnt)
    appto_df = pd.DataFrame({'H1':h1_, 'H2':h2_, 'COUNT':cnt_})

    #print(appto_df.tail(5))
    #print(appto_df.shape[0])

    marvel_net = nx.Graph() 
    for i, row in appto_df.iterrows():
        marvel_net.add_edge(row['H1'], row['H2'], weight=row['COUNT'])  # specify edge data
    nx.draw(marvel_net)

    return marvel_net, appto_df

## Custom function to create an edge between node x and node y, with a given text and width
def make_edge(x, y, text, width):
    return  go.Scatter(x=x, y=y, line=dict(width=width, color='lightgray'), hoverinfo='text', text=([text]), mode='lines')

def show_n_hero_network(network, a, n, all_net):

    marvel_net = network
    appto_df = a
    pos_ = nx.spring_layout(marvel_net, seed=11)
    cent_ = nx.pagerank(marvel_net, weight='weight') # page rank of heroes based on their link with other heroes count value previously finded
    cent_top = sorted(cent_.items(), key=lambda item: item[1], reverse=True)[:1] # page rank top 1
    #print("Top centered hero: ", cent_top)

    # For each edge, make an edge_trace, append to list
    edge_trace = []
    for edge in marvel_net.edges():    
        if marvel_net.edges()[edge]['weight'] > 0:
            char_1 = edge[0]
            char_2 = edge[1]
            x0, y0 = pos_[char_1]
            x1, y1 = pos_[char_2]
            trace  = make_edge([x0, x1, None], [y0, y1, None], None, width=5*(marvel_net.edges()[edge]['weight']/appto_df['COUNT'].max()))
            edge_trace.append(trace)
                    
    ## Make a node trace
    node_trace = go.Scatter(x=[], y=[], text=[], textposition="top center", textfont_size=10, mode='markers+text', hoverinfo='none',
                            marker=dict(color=[], size=[], line_width=[], line_color=[]))

    # For each node in network, get the position and size and add to the node_trace
    for node in marvel_net.nodes():
        x, y = pos_[node]
        node_trace['x'] += tuple([x])
        node_trace['y'] += tuple([y])
        color = 'gray'
        line_width = 2
        line_color = 'darkgray'
        name_text = ''
        #if node == 'SCARLET WITCH': name_text = node 
        name_text = node
        
        if node in HERO_COLOR:
            color = HERO_COLOR[node] 
            line_color='black'
            name_text = node
            
        if node in [v[0] for v in cent_top]:
             name_text = '<b>' + node + '</b>'
            
        node_trace['marker']['color'] += tuple([color])
        node_trace['marker']['size'] += tuple([int(400*cent_[node])]) # node size is proportional to page rank
        node_trace['marker']['line_width'] += tuple([line_width])
        node_trace['marker']['line_color'] += tuple([line_color])
        node_trace['text'] += tuple([name_text])
        
        
    ## Customize layout
    layout = go.Layout(
        paper_bgcolor='rgba(0,0,0,0)', # transparent background
        plot_bgcolor='rgba(0,0,0,0)', # transparent 2nd background
        xaxis =  {'showgrid': False, 'zeroline': False}, # no gridlines
        yaxis = {'showgrid': False, 'zeroline': False}, # no gridlines
    )

    ## Create figure
    fig = go.Figure(layout = layout)
    ## Add all edge traces
    for trace in edge_trace:
        fig.add_trace(trace)
    fig.add_trace(node_trace)
    fig.update_layout(showlegend = False)
    fig.update_xaxes(showticklabels = False)
    fig.update_yaxes(showticklabels = False)
    if(all_net == True):
        fig.update_layout(title=f"<b>Top {n} Heroes Network</b>")
        fig.write_image(f'./images/top{n}HeroNetwork.png')
    else:
        fig.update_layout(title=f"<b>Top {n} Heroes Reduced Network</b>")
        fig.write_image(f'./images/top{n}Hero_REDUCED_Network.png')

    fig.show()

def missing_intersection():
    print("THOR & IRON MAN in hero-network.csv")
    print(f"hero1=HULK, hero2=IRON MAN : {len(hero_net_df[(hero_net_df['hero1']=='HULK')&(hero_net_df['hero2']=='IRON MAN')])}")
    print(f"hero1=IRON MAN, hero2=HULK : {len(hero_net_df[(hero_net_df['hero2']=='HULK')&(hero_net_df['hero1']=='IRON MAN')])}")

    temp1 = set(edge_df[edge_df['hero']=='HULK']['comic'])
    temp2 = set(edge_df[edge_df['hero']=='IRON MAN']['comic'])
    print(f"Intersection in edges.csv : {len(temp1.intersection(temp2))}")

#ricordarsi di aggiornare la posizione dei ligthgray se si vogliono aggiungere o modificare i colori delle colonne
def print_bar_figure(cent_df, n, all_net):

    if(n == 30 and all_net == True):
        colors = [HERO_COLOR['CAPTAIN AMERICA']]+['lightgray']+\
                    [HERO_COLOR['HUMAN TORCH']]+['lightgray']*7+\
                    [HERO_COLOR['THOR']]+['lightgray']*3+\
                    [HERO_COLOR['SPIDER-MAN']]+\
                    [HERO_COLOR['WOLVERINE']]+['lightgray']*11+\
                    [HERO_COLOR['DR. STRANGE']]+['lightgray']*2
    
    else : colors = ['lightgray']*n
    fig = go.Figure(data=[go.Bar(
        x=cent_df.index,
        y=cent_df['mean_cent'],
        marker_color= colors
    )])
    fig.update_layout(title_text=f'<b>Mean Centrality of {n} Heros</b>', template='simple_white')
    fig.show()
    if all_net == True:
        fig.write_image(f'./images/MeanCentralityOf{n}HeroNetwork.png')
    else: 
        fig.write_image(f'./images/MeanCentralityOf{n}Hero_Reduced_Network.png')

#return cent_
def get_cent(network, a):
    marvel_net, appto_df = network.copy(), a.copy()
    cent_df = pd.DataFrame(index=list(marvel_net.nodes()))
    print(f"The number of hero pairs that never came out together : {len(appto_df[appto_df['COUNT']==0])}")
    # PAGERANK
    cent_ = nx.pagerank(marvel_net, weight='weight')
    cent_df['w_pagerank_cent'] = pd.Series(index=[k for k, v in cent_.items()], data=[float(v) for k, v in cent_.items()])

    # EIGENVALUE CENTRALITY
    cent_ = nx.eigenvector_centrality(marvel_net, weight='weight')
    cent_df['w_eigenvector_cent'] = pd.Series(index=[k for k, v in cent_.items()], data=[float(v) for k, v in cent_.items()])

    # DEGREE CENTRALITY
    cent_ = {h:0.0 for h in marvel_net.nodes()}
    for u, v, d in marvel_net.edges(data=True):
        cent_[u]+=d['weight']; cent_[v]+=d['weight'];
    cent_df['w_degree_cent'] = pd.Series(index=[k for k, v in cent_.items()], data=[float(v) for k, v in cent_.items()])

    # CLOSENESS CENTRALITY
    temp_net = marvel_net.copy()
    for u,v,d in temp_net.edges(data=True):
        if 'distance' not in d:
            if d['weight'] != 0:
                d['distance'] = 1.0/d['weight']
            else:
                d['distance'] = 2
    cent_ = nx.closeness_centrality(temp_net, distance='distance')
    cent_df['w_closeness_cent'] = pd.Series(index=[k for k, v in cent_.items()], data=[float(v) for k, v in cent_.items()])

    print(cent_df.head(5))
     # BETWEENES CENTRALITY
    #cent_ = nx.betweenness_centrality(marvel_net, weight='weight')
    #cent_df['w_betweenness_cent'] = pd.Series(index=[k for k, v in cent_.items()], data=[float(v) for k, v in cent_.items()])
    print("last")
    print(cent_df.tail(5))
    
    # SCALING
    for c in cent_df.columns:
        s = MinMaxScaler()
        cent_df[[c]] = s.fit_transform(cent_df[[c]])  
    cent_df['mean_cent'] = cent_df.mean(axis=1)
    cent_df = cent_df.sort_values(by=['mean_cent'], ascending=False)

    print("mean centrality of first 5 heroes :\n", cent_df.head(5))

    #JACCARD COEFFICIENT FOR 2 MAIN SUPERHEROES OF THE NET
    first, second = cent_df.index[[0, 2]]
    jc = nx.jaccard_coefficient(network, [(first, second)])
    for u,v,p in jc:
        print(f"\nJaccard coefficient for nodes ({u,v}): {p}" )

    return cent_df

# prepare Data
# to visualize the means on graphs
def iterate_mean_cent_visualization():

    mean_cent_df = pd.DataFrame(index=HERO_COLOR.keys())
    #for topn in [25, 50, 100, 200, 500]:
    for topn in [25, 50]:
        mean_cent_df[f"mean_cent_{topn}"] = get_cent(def_net(topn)).loc[HERO_COLOR.keys(), :]['mean_cent']
        
    # visualization
    fig = go.Figure()
    for c in mean_cent_df.T.columns:
        temp_ = mean_cent_df.T[c]
        fig.add_trace(go.Scatter(x=[f"In Top {n} Heroes" for n in [25, 50]], y=temp_, mode='lines+markers', name=c,
                                line=dict(color=HERO_COLOR[c])
                                ))

    fig.update_layout(title_text='<b>Mean Centrality</b>', template='simple_white')
    fig.show()

def small_world_experiment():
    ncr_ = []; spl_ = []; dia_ = []; rad_ = []
    for n in [25, 50, 100]:
        test_net = def_net(n)[0].copy()
        test_appto = def_net(n)[1].copy()

        ncr_.append(len(test_appto[test_appto['COUNT']==0])/len(test_appto))
        spl_.append(nx.average_shortest_path_length(test_net))
        dia_.append(nx.diameter(test_net))
        rad_.append(nx.radius(test_net))
        print('n: ', n, 'Non-Connection Ratio: ', ncr_, 'Avg Shortest Path Length: ', spl_, 'Diameter: ', dia_, 'Radius: ', rad_, '\n')
    ch_df = pd.DataFrame({'Heroes':[25, 50, 100], 'Non-Connection Ratio':ncr_, 'Avg Shortest Path Length':spl_, 'Diameter':dia_, 'Radius':rad_})

    # visualization
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    for c in ['Non-Connection Ratio', 'Avg Shortest Path Length', 'Diameter', 'Radius']:
        temp_ = ch_df[c]
        if c == 'Non-Connection Ratio':
            fig.add_trace(go.Scatter(x=[f"In Top {n} Heroes" for n in [25, 50, 100]], y=temp_, name=c, line=dict(color='darkgray')),
                        secondary_y=True)
        else:
            fig.add_trace(go.Bar(x=[f"In Top {n} Heroes" for n in [25, 50, 100]], y=temp_, name=c
                                    ))

    fig.update_layout(barmode='group')
    fig.update_yaxes(title_text="length", secondary_y=False)
    fig.update_yaxes(title_text="ratio", secondary_y=True)
    fig.update_layout(title_text='<b>Changes in Network Characteristics according to the number of heroes</b>', template='simple_white')
    fig.show()

def finger_snap(n):
    topn = n
    top_hero = edge_df.groupby(['hero'])[['comic']].count().sort_values(by=['comic'], ascending=False).head(topn).index
    top_hero = [h for i, h in enumerate(top_hero)] # after finger snap
    #print(top_hero, '\n', len(top_hero))
    no_elements_to_delete = len(top_hero) // 2
    no_elements_to_keep = len(top_hero) - no_elements_to_delete
    top_hero_reduct = set(random.sample(top_hero, no_elements_to_keep))  
    top_hero_reduct = [i for i in top_hero if i in top_hero_reduct]  

    h1_ = []; h2_ = []; cnt_ = []
    for comb in list(itertools.combinations(top_hero_reduct, 2)):    
        temp1 = set(edge_df[edge_df['hero']==comb[0]]['comic'])
        temp2 = set(edge_df[edge_df['hero']==comb[1]]['comic'])
        cnt = len(temp1.intersection(temp2)) # Appear Together    
        h1_.append(comb[0]); h2_.append(comb[1]); cnt_.append(cnt)
    appto_df50 = pd.DataFrame({'H1':h1_, 'H2':h2_, 'COUNT':cnt_})


    marvel_net50 = nx.Graph() 
    for i, row in appto_df50.iterrows():
        if row['COUNT'] > 0:
            marvel_net50.add_edge(row['H1'], row['H2'], weight=row['COUNT'])

    return marvel_net50, appto_df50 

def k_core_k_component(net, n):

    #K-CORE 
    n_nodes = range(1, n)
    k_cores = [nx.k_core(net, k).order() for k in n_nodes]
    #print("K-cores analysis: ", k_cores)
    #print("k-core count: ", len(k_cores))
    
    k = 0
    for k in n_nodes:
        k_cores_net = nx.k_core(net, k)
        print(f" {k}-core  nodes: ", k_cores_net)
        #print(k_cores_net.nodes())
    
    #fig = plt.figure(figsize = (10, 5))
    
    # creating the bar plot
    #plt.bar(n_nodes, k_cores, color ='blue', width = 0.4)   
    #plt.xlabel("values of k-core")
    #plt.ylabel("sets of node with correspondant k value")
    #plt.title("Distributions of k-cores")
    #plt.show()
    
    #K-COMPONENT
    #k_component = nx.k_components(net)
    #print(k_component)
    print(sorted(map(sorted, nx.k_edge_components(net, k=len(k_cores)))))

def grouping_measures(net):

    #AVERAGE SHORTES PATH LENGTH AND  CLUSTERING
    aspl = nx.average_shortest_path_length(net) # no weight 
    aspl2 = nx.average_clustering(net) #                    
    adgr = sum(dict(net.degree()).values())/float(len(net)) # no weight
    print("shortes path length: ", aspl)
    print("averege clustering: ", aspl2)
    print(adgr)

    #CLUSTERING COEFFICIENT
    print("Number of connected components: ", nx.number_connected_components(net))
    clus_coef = nx.clustering(net)
    print("Clustering coeffincient: ")
    c = Counter(clus_coef)
    print(len(c))
    n = 5
    print("Least common: ", c.most_common()[:-n-1:-1])
    print("Most common: ", c.most_common(5))

    #NUMBER OF TRIANGLES
    triangles = nx.triangles(net)
    print("Number of triangles and nodes: ", triangles)
    c = Counter(triangles)
    print(len(c))
    n = 5
    print("Least 5 traingles value: ", c.most_common()[:-n-1:-1])

    #TRANSATIVITY
    print("\nTransativity: ", nx.transitivity(net))

    #DIAMETER
    diameter = nx.diameter(net)
    print("Diameter: ", diameter)

    #ECCENTRICITY
    eccentricity = dict(nx.eccentricity(net))
    print("Eccentricity: ", eccentricity)

    #NET DEGREE
    print("net degree: ", net.degree())

    #MINIMAL NODE CUT
    node_cut = nx.minimum_node_cut(net)
    print(" Minimal node cut:  ", len(node_cut))

def show_iterate_graphs(all_net):
    for n in [30, 90, 100, 200]:
        net, appto = def_net(n)
        centrality = get_cent(net, appto)
        print_bar_figure(centrality, n, all_net)
        k_core_k_component(net, n)
        grouping_measures(net)
    # comment this below if you use finger snap
    #iterate_mean_cent_visualization()

#------------------------------
# EXAMPLES
# here you can find the most used functions for the project. You can uncomment or modify what you want 
# to try the code and the possible results

topn = 60

#FULL NETWORK
#show_n_hero_network(def_net(topn)[0], def_net(topn)[1], topn) #50
#print(net)
#show_n_hero_network(net, appto, topn, True)
#print_bar_figure(centrality, topn)
#centrality = get_cent(def_net(topn)[0], def_net(topn)[1])
#print_bar_figure(centrality, topn, True)
#k_core_k_component(net, topn)
#grouping_measures(net)
#show_iterate_graphs(all_net, True)


# FINGER SNAP
print("\n", topn//2, " heroes after finger snap")
net50, appto50 = finger_snap(topn)
print(net50)
centrality50 = get_cent(net50, appto50)
show_n_hero_network(net50, appto50, topn//2, False)

print_bar_figure(centrality50, topn//2, False)
k_core_k_component(net50, topn//2)
grouping_measures(net50)



#k_core_k_component(net50, topn//2)

# ERROR IN ASSORTATIVITY 
# net, appto = def_net(topn)
# mg = nx.MultiDiGraph()
# mg = net.to_undirected()
# r = nx.degree_assortativity_coefficient(mg)
# print(f"--> {r:3.1f}")





