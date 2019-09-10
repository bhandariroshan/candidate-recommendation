import networkx as nx
import csv
import numpy as np
# import seaborn as sns
import operator     
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import time
import random
from networkx.algorithms.centrality.degree_alg import degree_centrality
from networkx.algorithms.centrality.degree_alg import out_degree_centrality
from networkx.algorithms.centrality.degree_alg import in_degree_centrality
from networkx.algorithms.centrality import betweenness_centrality
from networkx.algorithms.centrality import betweenness_centrality, closeness_centrality
import scipy
from scipy.io import mmread
from networkx.convert_matrix import from_scipy_sparse_matrix


class FileReader(object):
    def __init__(self, filename):
        self.data = None

        with open(filename) as file:
            self.data = file.readlines()

    def clean_data(self):
        pass


class NetworkLoader(object):
    def __init__(self):
        nodes_file = 'nodes.txt'
        edges_file = 'nodes.txt'
        node_reader = FileReader(nodes_file)
        edges_reader = FileReader(edges_file)

        self.node_data = node_reader.data
        self.edges_data = edges_reader.data

        self.graph = nx.DiGraph()
        # self.create_node_file()
        self.load_network()

    def create_node_file(self):
        with open("nodes.txt", 'w') as file:
            for i in range(0, 262110):
                file.writelines(str(i) + '\n')
        print('written nodes file')

    def load_network(self):
        edges = nx.read_edgelist('edges.txt')
        nodes = nx.read_adjlist('nodes.txt')
    
        self.graph.add_edges_from(edges.edges())
        self.graph.add_nodes_from(nodes)

    def load_network_from_scipy_matrix(self):
        scipy_matrix = mmread('bcsstk18.mtx')
        self.graph = from_scipy_sparse_matrix(scipy_matrix)

    def export_networkxgraph_to_a_gephi_file(self):
         nx.write_gexf(self.graph, "test.gexf")

    def plot_a_subgraph(self):
        print('Drawing graph..')
        subgraph_nodes = [str(i) for i in range(0, 100)]
        nx.draw(self.graph.subgraph(subgraph_nodes), node_color='blue', font_size=8, font_weight='bold')

        print('Saving graph..')
        plt.savefig("graph.png", format="PNG")

    def summarize_graph(self):
        size_of_graph = self.graph.size()
        print("Size of graph is: ", size_of_graph)
        print("Degrees of graph is: ", self.find_degrees_of_graph())
        print("Max, Min, Average degree of graph is: ", self.find_average_min_and_max_degree_of_graph())
        # print("Center of Network: ", nx.center(self.graph))
        # print("Eccentricity of Network: ", nx.eccentricity(self.graph))
        print("Network centrality of the graph is: ", self.find_network_centralities())
    
    def compute_eigen_values(self):
        L = nx.normalized_laplacian_matrix(self.graph)
        e = np.linalg.eigvals(L.A)
        print("Largest eigenvalue:", max(e))
        print("Smallest eigenvalue:", min(e))
        plt.hist(e, bins=100)  # histogram with 100 bins
        plt.xlim(0, 2)  # eigenvalues between 0 and 2
        plt.savefig("eigen.png", format="PNG")

    def find_network_centralities(self):
        subgraph_nodes = [str(i) for i in range(0, 10000)]
        centralities = degree_centrality(self.graph.subgraph(subgraph_nodes))
        centralities_using_degree = sorted(centralities.items(), key=operator.itemgetter(1))

        in_degree_centralities = in_degree_centrality(self.graph.subgraph(subgraph_nodes))
        centralities_using_in_degree = sorted(in_degree_centralities.items(), key=operator.itemgetter(1))

        out_degree_centralities = out_degree_centrality(self.graph.subgraph(subgraph_nodes))
        centralities_using_out_degree = sorted(out_degree_centralities.items(), key=operator.itemgetter(1))
        print(centralities_using_in_degree[1], centralities_using_out_degree[70000])

    def find_degrees_of_graph(self):
        indegrees = [each_degree[1] for each_degree in self.graph.in_degree]
        outdegrees = [each_degree[1] for each_degree in self.graph.out_degree] 
        degree = [each_degree[1] for each_degree in self.graph.degree]

        # matplotlib histogram
        plt.hist(degree, color = 'blue', edgecolor = 'black',
                 bins = 50)

        # Add labels
        plt.title('Histogram of Degree Frequency')
        plt.xlabel('Degree')
        plt.ylabel('Frequency')

        # plt.show()
        plt.savefig('degree_distribution.png')

        # indegree histogram
        plt.hist(indegrees, color = 'blue', edgecolor = 'black',
                 bins = 50)

        # Add labels
        plt.title('Histogram of In Degree Frequency')
        plt.xlabel('In Degree (d+)')
        plt.ylabel('Frequency')

        # plt.show()
        plt.savefig('in_degree_distribution.png')

        # indegree histogram
        plt.hist(outdegrees, color = 'blue', edgecolor = 'black',
                 bins = 50)

        # Add labels
        plt.title('Histogram of Out Degree Frequency')
        plt.xlabel('Out Degree (d+)')
        plt.ylabel('Frequency')

        # plt.show()
        plt.savefig('out_degree_distribution.png')

    def get_neighbors(self, node_name):
        '''This function gives connected nodes of a node'''
        neighbors = self.graph.neighbors(str(node_name))

        return_values = []
        for each_node in neighbors:
            return_values.append((each_node, self.graph.degree(each_node)))

        return_values = sorted(return_values, key=lambda x: x[1])
        return return_values

    def find_average_min_and_max_degree_of_graph(self):
        degree_sum = 0
        max_degree = None
        min_degree = None

        for each_node in self.graph.degree:
            if min_degree is None or max_degree is None:
                min_degree  = each_node[1]
                max_degree  = each_node[1]
            
            if each_node[1] < min_degree:
                min_degree = each_node[1]

            if each_node[1] > max_degree:
                max_degree = each_node[1]

            degree_sum += each_node[1]

        average_degree = degree_sum / len(self.graph.degree)
        objects = ('Max Degree', 'Min Degree', 'Average Degree')
        y_pos = np.arange(len(objects))
        degrees = [max_degree, min_degree, average_degree]

        plt.bar(y_pos, degrees, align='center', alpha=0.5)
        plt.xticks(y_pos, objects)
        plt.ylabel('Degree')
        plt.title('Degree plot')

        # plt.show()
        plt.savefig('degree.png')
        return max_degree, min_degree, average_degree

    def implement_algorithm(
            self,
            startNode=None,
            algorithm="BFS",
            crawlsize=3500,
            network_crawl_writer=None
        ):                   

        '''This function implements an algorithm for crawling.'''

        visited = []
        nodes_processed = 0
        
        start = time.time()

        Q = self.get_neighbors(startNode)
        while (Q):
            nodes_processed += 1
            if nodes_processed > crawlsize:
                break

            # implement BFS algorithm
            if algorithm == "BFS":
                u = Q[0]
                Q = Q[1:]

            # implement DFS
            elif algorithm == "DFS":
                u = Q.pop()      

            # implement RFS
            elif algorithm == "RFS":
                index = random.randint(0, len(Q)-1)
                u = Q.pop(index)

            # add the page to visited list
            visited.append(u)

            # collect all neighbor nodes
            v = self.get_neighbors(u[0])

            for each_v in v:
                if each_v not in Q and each_v not in visited:
                    Q.append(each_v)

        end = time.time()

        suggested_products = sorted(visited, key=lambda x: x[1], reverse=True)[0:10]

        if len(visited) >= 10:
            suggested_final_products = ''
            for each_suggested_product in suggested_products:
                suggested_final_products += str(each_suggested_product[0]) + '|'

            suggested_products = suggested_products[:-1]

            network_crawl_writer.writerow(
                [
                    startNode,
                    algorithm,
                    suggested_final_products,
                    end - start
                ]
            )

    def crawl_network_to_generate_recommendation(self):
        """
            This Function iterates for:
                a. Different fixed Nodes 
                b. For 3 algorithms
                c. Stops when it has crawled fixed size or has reached leaf
                c. Finds recommdation 
        """
        algorithms = ["BFS", "DFS", "RFS"]
        
        # get start points for algorithmic traversal 
        start_nodes = [i for i in range(1, 50000, 200)]

        # Stop crawling when processed
        stop_crawl_when_processed = 1000

        fname = 'traversal' + '.csv'
        with open(fname, 'w') as csvfile:
            fieldnames = [
                'Start Node',
                "Algorithm" ,
                "Suggested Products",
                "Time to Suggest"
            ]

            tree_crawl_writer = csv.writer(
                csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)

            tree_crawl_writer.writerow(fieldnames)

            for each_start_node in start_nodes:
                # crawl all the start points for all algorithm
                # run algorithm to find highest connecting nodes for recommendation
                for each_algorith in algorithms:
                    self.implement_algorithm(
                        each_start_node,
                        each_algorith,
                        stop_crawl_when_processed,
                        tree_crawl_writer
                    )

                    print("crawling from start point:: " + 
                        str(each_start_node), ", Algorithm::" + str(each_algorith))

        bfs_avg, bfs_count = 0, 0
        dfs_avg, dfs_count = 0, 0
        rfs_avg, rfs_count = 0, 0
        with open('traversal.csv','r') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                if row and row[1] == 'BFS':
                    bfs_avg += float(row[3])
                    bfs_count += 1

                if row and row[1] == 'DFS':
                    dfs_avg += float(row[3])
                    dfs_count += 1

                if row and row[1] == 'RFS':
                    rfs_avg += float(row[3])
                    rfs_count += 1

        bfs_avg = bfs_avg / bfs_count
        dfs_avg = dfs_avg / dfs_count
        rfs_avg = rfs_avg / rfs_count

        y_pos = np.arange(len(algorithms))
        degrees = [bfs_avg, dfs_avg, rfs_avg]

        plt.bar(y_pos, degrees, align='center', alpha=0.5)
        plt.xticks(y_pos, algorithms)
        plt.ylabel('Average Time')
        plt.title('Average Time required to make 10 product suggestion')

        # plt.show()
        plt.savefig('time.png')

network_loader = NetworkLoader()
network_loader.load_network_from_scipy_matrix()
network_loader.export_networkxgraph_to_a_gephi_file()
network_loader.compute_eigen_values()
# network_loader.crawl_network_to_generate_recommendation()

'''
    1. Plot Indegree vS OutDegree
    2. Plot Network Centrality (Closeness Vs Betweeness, degree Vs Eigen)
    4. Plot Cluster
    5. Implement BFS, DFS, RFS to crwal and travel network and compare efficiency

'''
