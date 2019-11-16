#!/usr/bin/env python
# coding: utf-8

# In[1]:

import sys
import pickle
import networkx as nx
from networkx.algorithms.distance_measures import diameter

 
G=nx.read_gpickle("github.txt")
all_nodes = list(G.nodes)

print(len(G.nodes))

degrees = sorted(G.degree, key=lambda x: x[1], reverse=False)
print("Minimum degree of a graph is: ", degrees[0]) 

# Prune the graph
# for each_node in all_nodes:
#     if G.degree(each_node) < 100:
#         G.remove_node(each_node)
# print(len(G.nodes), len(G.edges))
# print(diameter(G))

degrees = sorted(G.degree, key=lambda x: x[1], reverse=True)
print("Maximum degree of a graph is: ", degrees[0][1])

degrees = sorted(G.degree, key=lambda x: x[1], reverse=False)
print("Minimum degree of a graph is: ", degrees[0])

# TODO
# 1. Find the diameter
# print("diameter of the graph is: ", dm)

num_nodes = len(G.nodes)
print("Number of Nodes", num_nodes)

num_edges = len(G.edges)
print("Number of edges", num_edges)

print("Average Degree of the graph is: ", num_edges / num_nodes)

degrees = sorted(G.degree, key=lambda x: x[1], reverse=True)
print("Maximum degree of a graph is: ", degrees[0][1])

degrees = sorted(G.degree, key=lambda x: x[1], reverse=False)
print("Minimum degree of a graph is: ", degrees[0][1])

weakest  = max([len(c) for c in sorted(nx.weakly_connected_components(G),key=len, reverse=True)])
print("Nodes in Weakly connected component", weakest)

strongest = max([len(c) for c in sorted(nx.strongly_connected_components(G),key=len, reverse=True)])
print("Nodes in strongest connected component", strongest)


dicts = nx.clustering(G)
sums = 0
zeros = 0
for i in dicts:
    sums += dicts[i]
    if dicts[i] == 0:
        zeros += 1
average_clustering_coefficient = sums/len(G.nodes)
print("Average Clustering Coefficient", average_clustering_coefficient)
# print(dicts)

print("Weakly connected component", nx.number_weakly_connected_components(G))
print("Strongly connected component", nx.number_strongly_connected_components(G))
print("Average Clustering Coefficient", nx.average_clustering(G))


import itertools
def wedge_iterator(graph):
    for node in graph.nodes:
        neighbors = graph.neighbors(node)
        for pair in itertools.combinations(neighbors, 2):
            yield (node, pair)
            
def count_triangle(graph):
    n = 0
    for wedge in wedge_iterator(graph):
        if graph.has_edge(wedge[1][0], wedge[1][1]) or graph.has_edge(wedge[1][1], wedge[1][0]):
            n += 1
    return n

print("Number of Triangles ",count_triangle(G))
# tri = nx.triangles(G)


# In[17]:




all_cliques= nx.enumerate_all_cliques(G)
print(all_cliques)

# In[13]:


def generate_triangles(nodes):
        visited_ids = set() # mark visited node
        for node_a_id in nodes:
            temp_visited = set() # to get undirected triangles
            for node_b_id in nodes[node_a_id]:
                if node_b_id == node_a_id:
                    raise ValueError # to prevent self-loops, if your graph allows self-loops then you don't need this condition
                if node_b_id in visited_ids:
                    continue
                for node_c_id in nodes[node_b_id]:
                    if node_c_id in visited_ids:
                        continue    
                    if node_c_id in temp_visited:
                        continue
                    if node_a_id in nodes[node_c_id]:
                        yield(node_a_id, node_b_id, node_c_id)
                    else:
                        continue
                temp_visited.add(node_b_id)
            visited_ids.add(node_a_id)


# In[14]:


# cycles = list(generate_triangles(G.nodes))
# print(cycles)


# In[ ]:




