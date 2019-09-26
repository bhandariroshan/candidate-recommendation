import sys
import scipy
import math
from scipy.io import mmread
import networkx as nx
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
        if '.mtx' in edge_file_name or '.mm' in edge_file_name:
            self.load_network_from_scipy_matrix()
        else:
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
        auth_network_list = []
        hub_network_list = []

        for each_node, data in self.graph.nodes(data=True):
            data['node'] = each_node
            network.append(data)

        auth_network = sorted(network,  key=lambda k: k['auth'], reverse=True)
        hub_network = sorted(network,  key=lambda k: k['hub'], reverse=True)

        for each_node in auth_network:
            auth_network_list.append(each_node['node'])

        for each_node in hub_network:
            hub_network_list.append(each_node['node'])

        if self.print_list_size:
            print(auth_network_list[0:self.print_list_size])
            print(hub_network_list[0:self.print_list_size])
        else:
            print(auth_network_list)
            print(hub_network_list)

    def find_hubs_and_authorities(self):
        self.assign_authorities_and_hubs()
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
        
        self.print_network_with_hubs_and_authorities()


if __name__ == "__main__":  
    # execute only if run as a script
    matrix_file_name = sys.argv[1]
    try:
        steps = sys.argv[2]
    except:
        steps = 2
    
    try:
        normalize = int(sys.argv[3])
    except:
        normalize = True

    try:
        print_size = int(sys.argv[4])
    except:
        print_size=None

    nm = NetworkManipulator(
        # matrix_file_name='delaunay_n14.mtx', 
        edge_file_name=matrix_file_name,
        steps=steps,
        normalize=normalize,
        print_list_size=print_size
    )
    nm.find_hubs_and_authorities()