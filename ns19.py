import networkx as nx
import csv
import numpy as np
# import seaborn as sns
import operator     
import matplotlib.pyplot as plt
plt.switch_backend('agg')

from networkx.algorithms.centrality.degree_alg import degree_centrality
from networkx.algorithms.centrality.degree_alg import out_degree_centrality
from networkx.algorithms.centrality.degree_alg import in_degree_centrality
from networkx.algorithms.centrality import betweenness_centrality
from networkx.algorithms.centrality import betweenness_centrality, closeness_centrality

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
        # self.plot_a_subgraph()
        self.summarize_graph()

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

    def plot_a_subgraph(self):
        print('Drawing graph..')
        nx.draw(self.graph, node_size=1500, node_color='blue', font_size=8, font_weight='bold')

        plt.tight_layout()
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
        
    def find_network_centralities(self):
        centralities = degree_centrality(self.graph)
        centralities_using_degree = sorted(centralities.items(), key=operator.itemgetter(1))

        in_degree_centralities = in_degree_centrality(self.graph)
        centralities_using_in_degree = sorted(in_degree_centralities.items(), key=operator.itemgetter(1))

        out_degree_centralities = out_degree_centrality(self.graph)
        centralities_using_out_degree = sorted(out_degree_centralities.items(), key=operator.itemgetter(1))
        print(centralities_using_in_degree[1], centralities_using_out_degree[70000])

        plt.plot(
            [each_centrality[0] for each_centrality in centralities_using_degree], 
            [each_centrality[1] for each_centrality in centralities_using_degree],
            'ro'
        )

        # Add labels
        plt.title('Degree Centrality Plot')
        plt.xlabel('Node')
        plt.ylabel('Centrality')

        # plt.show()
        plt.savefig('degree_centrality.png')

        # closeness_centralities = closeness_centrality(self.graph)
        # centralities_using_closeness = sorted(closeness_centralities.items(), key=operator.itemgetter(1))

        # betweenness_centralities = betweenness_centrality(self.graph)
        # centralities_using_betweenness = sorted(betweenness_centralities.items(), key=operator.itemgetter(1))
        # print(centralities_using_closeness[0], centralities_using_betweenness[0])

        # eigen_centrality = nx.eigenvector_centrality(self.graph, max_iter=10)
        # eigen_centrality = sorted(eigen_centrality.items(), key=operator.itemgetter(1))
        # print(eigen_centrality[0], centralities_using_degree[0])

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
        startNode=None,
        algorithm="BFS",
        crawlsize=3500,
        sample_files=[],
        network_crawl_writer=None
        ):                   

        '''This function implements an algorithm for crawling.'''
        # print(startNode, algorithm, crawlsize)

        visited = []
        word_list = []
        nodes_processed = 0
        total_word_count = 0

        Q = get_neighbors(startNode)
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
            v = get_neighbors(u)

            for each_v in v:
                if each_v not in Q and each_v not in visited:
                    Q.append(each_v)

            network_crawl_writer.writerow(
                [
                    startNode,
                    algorithm,
                    crawlsize,
                    u
                ]
            )



network_loader = NetworkLoader()


'''
    1. Plot Indegree vS OutDegree
    2. Plot Network Centrality (Closeness Vs Betweeness, degree Vs Eigen)
    3. Plot Graph
    4. Plot Cluster
    5. Implement BFS, DFS, RFS to crwal and travel network and compare efficiency

'''
