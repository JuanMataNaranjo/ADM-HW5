from collections import defaultdict, Counter
import functionality as fn
import plotly.express as plty
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from collections import deque


class Graph:

    def __init__(self):
        """
        Initialize the Graph object

        :attribute edges : dictionary of vertices (keys) to sets of vertices (values)
        :attribute induced_subgraph:
        """
        self._edges = defaultdict(set)


    @classmethod
    def from_dict(cls, dict_):
        """
        Create a new Graph object from a dictionary

        :param dict_ : dictionary of vertices (keys) to sets of vertices (values)
        :return : new Graph instance
        """
        g = cls()
        g._edges.update({k:set(v) for k, v in dict_.items()})
        vertices = list(g._edges.keys())
        for v in vertices:
            for u in g._edges[v]:
                g.add_vertex(u)     # make sure that all the vertices of the graph are in the adjacency list:
                                    # without this step, a sink node wouldn't appear in g._edges
        return g

    @property
    def n_vertices_(self):
        """
        Read-only property, number of vertices in the graph
        """
        return len(self._edges)

    @property
    def n_edges_(self):
        """
        Read-only property, number of edges in the graph
        """
        n = 0
        for v in self._edges:
            n += len(self._edges[v]) 
        return n

    @property
    def density_(self):
        """
        Read-only property, density of the graph: |E| / [ |V| * (|V| - 1) ]
        """
        n_v = self.n_vertices_
        return np.format_float_scientific(self.n_edges_ / (n_v * (n_v - 1)), precision=3)

    def get_vertices(self):
        """
        Get the vertices of the graph

        :return : list of vertices
        """
        return list(self._edges.keys())

    def get_edges(self):
        """
        Get the edges of the graph

        :return : list of (source, destination) tuples
        """
        return [(v, u) for v in self._edges.keys() for u in self._edges[v]]

    def add_edge(self, v, u):
        """
        Add an edge to the graph

        :param v : source vertex
        :param u : destination vertex

        :return : 
        """
        self._edges[v].add(u)

    def add_vertex(self, v):
        """
        Add a vertex to the graph

        :param v : new vertex
        
        :return : 
        """
        self._edges[v]


    def in_degree(self, v=None):
        """
        Compute the in-degree of a node v

        :param v : a node of the graph
        :return in_d : int
        """
        if v is None:
            degrees = defaultdict.fromkeys(self._edges.keys(), 0)
            for v in degrees:
                for u in self._edges[v]:
                    degrees[u] += 1
            return dict(degrees)
        else:
            in_d = 0
            for u in self._edges:
                if v in self._edges[u]:
                    in_d += 1
            return in_d


    def out_degree(self, v=None):
        """
        Compute the out-degree of a node v

        :param v : a node of the graph
        :return  : int
        """
        if v is None:
            degrees = dict.fromkeys(self._edges.keys(), 0)
            for v in degrees:
                degrees[v] = len(self._edges[v])
            return degrees
        return len(self._edges[v])


    def degree(self, v=None):
        """
        Compute the degree of a node v

        :param v : a node of the graph
        :return  : int
        """
        if v is None:
            in_deg = self.in_degree()
            out_deg = self.out_degree()
            degrees = dict.fromkeys(self._edges.keys(), 0)
            for v in degrees:
                degrees[v] = in_deg[v] + out_deg[v]
            return degrees
        return self.in_degree(v) + self.out_degree(v)


    def degree_distro(self, normalize=False):
        """
        Return the degree distribution of the graph

        :param normalize : normalize the degree distribution
        :return : degree d to number of vertices with degree d, defaultdict
        """
        degree_dist = Counter(self.degree().values())
        if normalize:
            n_v = self.n_vertices_
            degree_dist = {d:(degree_dist[d] / n_v) for d in degree_dist}
        return dict(degree_dist)

 
    def plot_degree_distro(self, normalize=True):
        """
        Plot the degree distribution of the graph

        :param normalize : normalize the degree distribution
        :return 
        """
        degree_dist = self.degree_distro(normalize)
        fig = plty.histogram(data_frame=pd.DataFrame(data=degree_dist.items(), columns=['Degree', 'Normalized number of nodes']),\
                            x='Degree', y='Normalized number of nodes', title='Degree distribution')
        fig.show()

    def __repr__(self):
        """
        Represent the graph as the list of its edges
        """
        return str(self.get_edges())

    @staticmethod
    def plot_graph(graph, with_labels=True, node_size=100):
        """
        Method to visualize graph or sub_graph

        :param graph: graph to be visualized
        :param with_labels: bool to add labels or not
        :param node_size: node size
        :return: Plot
        """
        g = nx.DiGraph(graph)
        plt.figure(figsize=(12, 8))
        plt.clf()
        nx.draw(g, with_labels=with_labels, node_size=node_size)
        plt.show();

    def pages_in_click(self, initial_page, num_clicks, print_=False):
        """
        Given a graph, an initial starting point and the number of clicks, how many, and which pages will we be able to
        visit?

        :param initial_page: Page we will be starting out from
        :param num_clicks: Number of clicks we are willing to do
        :param print_: Bool to visualize some of the outputs or not
        :return: Pages seen  with the given number of clicks
        """

        # This will be a list of all the pages that we are able to visit during our clicks
        pages_visited = set()
        # This will be a list of articles that we are able to reach at the ith click. We will use this list to check the
        # articles that we can reach in the i+1th click
        queue = set([initial_page])

        # Placeholder to keep track of the clicks we are doing
        clicks = 0
        # Interrupt the loop once we reach the required number of clicks
        while clicks < num_clicks:
            # List of elements that will be used for the next loop
            new_queue = set()
            # List of elements that don't have any out-node
            last_nodes = set()
            # Loop over all the pages of the current click
            for node in queue:
                if print_:
                    print(node)
                # If a given node has target node, include the out-nodes into the new_queue list
                if bool(self.edges[node]):
                    new_queue.update(self.edges[node])
                # If a given node has no target node, include it in the last_nodes list (this list will not be used to)
                # for further inspection but we will have to consider it as an article that has been seen
                else:
                    last_nodes.update([node])

            if print_:
                print(new_queue)

            # Update queue as the new pages to explore
            queue = new_queue
            # Update pages_seen with the pages that have been seen in this click (pages that still have out-nodes and
            # pages that end at that node)
            pages_visited.update(new_queue | last_nodes)
            # Update the number of clicks done
            clicks += 1

        # Return the unique pages
        return set(pages_visited)

    # TODO:  Use class methods to  construct new induced-subgraph. Issue is that the methods construct over the 
    #  self.edge and we don't  want that
    # TODO: See how to return the new class
    def generate_induced_subgraph(self, vertices):
        """
        Given a set of vertices, compute it's induced subgraph

        :param vertices: Set of vertices
        :return: Store induced sub_graph in class
        """
        induced_subgraph = defaultdict(set)
        for vertex in vertices:
            induced_subgraph[vertex] = vertices.intersection(self.edges[vertex])

        return Graph(induced_subgraph)

    # TODO: probably not needed method
    def bfs(self, start):
        """
        Function to compute the bfs at any starting point

        :param start: Initial page
        :return: Pages that can be visited from that starting point
        """
        visited = set()
        queue = deque([start])

        while queue:
            node = queue.popleft()
            if node not in visited:
                visited.update([node])
                neighbours = self.edges[node]
                for neighbour in neighbours:
                    queue.append(neighbour)
        return visited

    def min_cut(self, article1, article2):
        """
        Given two random articles, get the minimum number of edges that need to be breaken down to dut the link
        between both articles.

        This algorithm will follow the logic followed by the Edmonds-Karp Algorithm (max flow is equal to min cut),
        which is an improvement on the Ford-Fulkerson Algorithm

        :param article1: Integer representing an article (source article)
        :param article2: Integer representing an article (sink article)
        :return: Integer (number of edges)
        """
        x = self.induced_subgraph
        min_cut = 0

        return min_cut


