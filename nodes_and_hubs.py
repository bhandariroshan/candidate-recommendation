import sys
import scipy
from scipy.io import mmread
import networkx as nx
from networkx.convert_matrix import from_scipy_sparse_matrix

'''

1 G := set of pages
2 for each page p in G do
3   p.auth = 1 // p.auth is the authority score of the page p
4   p.hub = 1 // p.hub is the hub score of the page p

5 function HubsAndAuthorities(G)
6   for step from 1 to k do // run the algorithm for k steps
7     for each page p in G do  // update all authority values first
8       p.auth = 0
9       for each page q in p.incomingNeighbors do // p.incomingNeighbors is the set of pages that link to p
10         p.auth += q.hub
11     for each page p in G do  // then update all hub values
12       p.hub = 0
13       for each page r in p.outgoingNeighbors do // p.outgoingNeighbors is the set of pages that p links to
14         p.hub += r.auth

'''


class NetworkManipulator(object):
    def __init__(self, 
        matrix_file_name=None, 
        edge_file_name=None, 
        steps=None
    ):
        self.steps = steps
        self.graph = nx.DiGraph()
        
        self.matrix_file_name = matrix_file_name
        self.edge_file_name = edge_file_name

        if self.matrix_file_name:
            self.graph = self.load_network_from_scipy_matrix()

        elif self.edge_file_name:
            self.load_network_from_edgelist_file(self.edge_file_name)
        else:
            raise ('Nothing to do. Please check.')          

    def load_network_from_edgelist_file(self, edge_file_name):
        fh=open(edge_file_name, 'rb')
        self.graph = nx.read_edgelist(fh, create_using=nx.DiGraph())

    def load_network_from_scipy_matrix(self):
        scipy_matrix = mmread(self.matrix_file_name)
        self.graph = from_scipy_sparse_matrix(scipy_matrix, create_using=nx.DiGraph())

    def assign_authorities_and_hubs(self):
        for node, data in self.graph.nodes(data=True):
            data['auth'] = 1
            data['hub'] = 1

    def print_network_with_hubs_and_authorities(self):
        network = []
        for each_node, data in self.graph.nodes(data=True):
            data['node'] = each_node
            network.append(data)

        auth_network = sorted(network,  key=lambda k: k['auth'], reverse=True)[0:10]

        for each_node in auth_network:
            print(each_node['node'], each_node['auth'], each_node['hub'])

    def find_hubs_and_authorities(self):
        self.assign_authorities_and_hubs()
        for i in range(int(self.steps)):
            for node, data in self.graph.nodes(data=True):
                data['auth'] = 0
                
                neighbors = self.graph.in_edges(node, data=True)
                for in_node, self_node, in_edge_data in neighbors:
                    data['auth'] += self.graph.nodes[in_node]['hub']

            for node, data in self.graph.nodes(data=True):
                data['hub'] = 0
                
                neighbors = self.graph.out_edges(node, data=True)
                for self_node, out_node, out_edge_data in neighbors:
                    data['hub'] += self.graph.nodes[out_node]['auth']

        print("Completed updating hubs and auth for {} steps. ".format(self.steps))
        self.print_network_with_hubs_and_authorities()

if __name__ == "__main__":  
    # execute only if run as a script
    matrix_file_name = sys.argv[1]
    steps = sys.argv[2]

    nm = NetworkManipulator(
        # matrix_file_name='delaunay_n14.mtx', 
        edge_file_name=matrix_file_name,
        steps=steps
    )
    nm.find_hubs_and_authorities()