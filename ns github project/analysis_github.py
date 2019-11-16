import pickle
import networkx as nx
import csv
import operator

G=nx.read_gpickle("github.txt")


def prune_graph():
	# Prune the graph
	for each_node in all_nodes:
	    if G.degree(each_node) < 10 and G.degree(each_node) > 4:
	        G.remove_node(each_node)
	print(len(G.nodes), len(G.edges))


def print_degree():
	degrees = sorted(G.degree, key=lambda x: x[1], reverse=False)
	print("Minimum degree of a graph is: ", degrees[0])


def create_adjacency_list_file():
	with open('data.csv', 'w', newline='') as file:
		writer = csv.writer(file, delimiter=',')
		writer.writerow(['Source', 'Target'])
		for each_edge in G.edges:
			writer.writerow([each_edge[0], each_edge[1]])


def get_popular_language_by_project_count():
	language = {}
	for each_node in G.nodes:
		lan = G.nodes[str(each_node)].get('language','')
		if str(lan).lower() != 'none' and str(lan).lower() != '' and str(lan).lower() not in language:
			language[str(lan).lower()] = 1
		elif str(lan).lower() != 'none' and str(lan).lower() != '':
			language[str(lan).lower()] += 1
	
	language_list = []

	for each_language in language:
		language_list.append({'language':each_language, 'count':language[each_language]})

	sorted_language = sorted(language_list, key = lambda i: i['count'],reverse=True)  [0:10]
	with open('language_project.csv', 'w', newline='') as file: 
		writer = csv.writer(file, delimiter=',')
		writer.writerow(['Language', 'Project Count'])
		for each_language in sorted_language:
			writer.writerow([each_language['language'], each_language['count']])
	print(sorted_language)

def get_popular_language_by_commit_count():
	language = {}
	for each_node in G.nodes:
		lan = G.nodes[str(each_node)].get('language','')
		if str(lan).lower() != 'none' and str(lan).lower() != '' and str(lan).lower() not in language:
			language[str(lan).lower()] = G.nodes[str(each_node)].get('commit_count',0)
		elif str(lan).lower() != 'none' and str(lan).lower() != '':
			language[str(lan).lower()] += G.nodes[str(each_node)].get('commit_count',0)
	
	language_list = []

	for each_language in language:
		language_list.append({'language':each_language, 'count':language[each_language]})

	print(len(language_list))
	sorted_language = sorted(language_list, key = lambda i: i['count'],reverse=True) [0:10]
	with open('language_commit.csv', 'w', newline='') as file: 
		writer = csv.writer(file, delimiter=',')
		writer.writerow(['Language', 'Commit Count'])
		for each_language in sorted_language:
			writer.writerow([each_language['language'], each_language['count']])

def get_popular_language_by_contributors_count():
	language = {}
	for each_node in G.nodes:
		lan = G.nodes[str(each_node)].get('language','')
		if str(lan).lower() != 'none' and str(lan).lower() != '' and str(lan).lower() not in language:
			print(lan,  G.degree[str(each_node)])
			language[str(lan).lower()] = G.degree[str(each_node)]
		elif str(lan).lower() != 'none' and str(lan).lower() != '':
			print(lan,  G.degree[str(each_node)], "else")
			language[str(lan).lower()] += G.degree[str(each_node)]
	
	language_list = []

	for each_language in language:
		language_list.append({'language':each_language, 'count':language[each_language]})


	sorted_language = sorted(language_list, key = lambda i: i['count'],reverse=True)
	# print(sorted_language)

# get_popular_language_by_contributors_count() # Needs improvement

get_popular_language_by_project_count()
get_popular_language_by_commit_count()