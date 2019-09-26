import sys
import scipy
import math
from scipy.io import mmread
import networkx as nx
import numpy as np
from scipy import sparse
from networkx.convert_matrix import from_scipy_sparse_matrix


class NetworkManipulator(object):
    def __init__(self, 
        edge_file_name=None, 
        steps=3,
        normalize=True,
        print_list_size=None
    ):
        self.steps = steps
        self.graph = nx.DiGraph()
        self.normalize = bool(normalize)
        self.matrix_file_name = matrix_file_name
        self.edge_file_name = edge_file_name
        self.print_list_size = print_list_size

        if self.edge_file_name:
            self.load_network_from_edgelist_file(self.edge_file_name)

        else:
            raise ('Nothing to do. Please check.')          

    def load_network_from_edgelist_file(self, edge_file_name):
        # Load the network from files
        if '.mtx' in edge_file_name or '.mm' in edge_file_name:
            self.load_network_from_scipy_matrix()
        else:
            fh=open(edge_file_name, 'rb')
            self.graph = nx.read_edgelist(fh, create_using=nx.DiGraph())

    def load_network_from_scipy_matrix(self):
        # Load the network from scipy sparse matrix
        scipy_matrix = mmread(self.matrix_file_name)
        b = scipy_matrix.todense()
        c = np.matrix(b)
        d = sparse.csr_matrix(c)
        self.graph = nx.DiGraph(c)

    def assign_authorities_and_hubs(self):
        # assign default values for hubs and authorities
        for node, data in self.graph.nodes(data=True):
            data['auth'] = 1
            data['hub'] = 1

    def print_network_with_hubs_and_authorities(self):
        network = []
        auth_network_list = ""
        hub_network_list = ""

        for each_node, data in self.graph.nodes(data=True):
            data['node'] = each_node
            network.append(data)

        auth_network = sorted(network,  key=lambda k: k['auth'], reverse=True)
        hub_network = sorted(network,  key=lambda k: k['hub'], reverse=True)

        for count, each_node in enumerate(auth_network):
            if count > self.print_list_size:
                break
            auth_network_list += str(each_node['node']) + " "

        auth_network_list = auth_network_list[:-1]

        for count, each_node in enumerate(hub_network):
            if count > self.print_list_size:
                break

            hub_network_list += str(each_node['node']) + " "

        hub_network_list = hub_network_list[:-1]

        print(auth_network_list)
        print(hub_network_list)

    def find_hubs_and_authorities(self):
        self.assign_authorities_and_hubs()
        ''' Implement HITS Algorithm. '''
        for i in range(int(self.steps)):
            norm = 0 
            for node, data in self.graph.nodes(data=True):
                data['auth'] = 0
                
                neighbors = self.graph.in_edges(node, data=True)
                for in_node, self_node, in_edge_data in neighbors:
                    data['auth'] += self.graph.nodes[in_node]['hub']

                if self.normalize:
                    norm += math.pow(data['auth'], 2) # calculate the sum of the squared auth values to normalise
            
            if self.normalize:
                norm = math.sqrt(norm)
                for node, data in self.graph.nodes(data=True):
                    data['auth'] = round(data['auth'] / norm, 3)
            
            norm = 0
            for node, data in self.graph.nodes(data=True):
                data['hub'] = 0
                
                neighbors = self.graph.out_edges(node, data=True)
                for self_node, out_node, out_edge_data in neighbors:
                    data['hub'] += self.graph.nodes[out_node]['auth']
                
                if self.normalize:
                    norm += math.pow(data['hub'], 2) # calculate the sum of the squared auth values to normalise

            if self.normalize:
                norm = math.sqrt(norm)
                for node, data in self.graph.nodes(data=True):
                    data['hub'] = round(data['hub'] / norm, 3)

        ''' Print top n values of hubs and authorities. '''
        self.print_network_with_hubs_and_authorities()


if __name__ == "__main__":  
    # execute only if run as a script
    matrix_file_name = sys.argv[1]
    try:
        steps = sys.argv[2]
    except:
        steps = 10

    try:
        normalize = int(sys.argv[3])
    except:
        normalize = True

    try:
        print_size = int(sys.argv[4])
    except:
        print_size=20

    # call network manager 
    nm = NetworkManipulator( 
        edge_file_name=matrix_file_name, # network file to load
        steps=steps, # Number of steps to iterate
        normalize=normalize, # normalize or not
        print_list_size=print_size # size of list elements to print
    )
    nm.find_hubs_and_authorities()