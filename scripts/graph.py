from collections import defaultdict
import networkx as nx
import matplotlib.pyplot as plt


class Graph:

    def __init__(self):
        """
        Initialize the Graph object

        :attribute edges : dictionary of vertices (keys) to sets of vertices (values)
        """
        self.edges = defaultdict(set)

    @classmethod
    def from_dict(cls, dict_):
        """
        Create a new Graph object from a dictionary

        :param dict_ : dictionary of vertices (keys) to sets of vertices (values)
        :return : new Graph instance
        """
        g = cls()
        g.edges.update({k: set(v) for k, v in dict_.items()})
        vertices = list(g.edges.keys())
        for v in vertices:
            for u in g.edges[v]:
                g.add_vertex(u)  # make sure that all the vertices of the graph are in the adjacency list:
                # without this step, a sink node wouldn't appear in g.edges
        return g

    @property
    def n_vertices_(self):
        """
        Read-only property, number of vertices in the graph
        """
        return len(self.edges)

    @property
    def n_edges_(self):
        """
        Read-only property, number of edges in the graph
        """
        n = 0
        for v in self.edges:
            n += len(self.edges[v])
        return n

    @property
    def density_(self):
        """
        Read-only property, density of the graph: |E| / [ |V| * (|V| - 1) ]
        """
        n_v = self.n_vertices_
        return round(self.n_edges_ / (n_v * (n_v - 1)), 3)

    def get_vertices(self):
        """
        Get the vertices of the graph

        :return : list of vertices
        """
        return list(self.edges.keys())

    def get_edges(self):
        """
        Get the edges of the graph

        :return : list of (source, destination) tuples
        """
        return [(v, u) for v in self.edges.keys() for u in self.edges[v]]

    def add_edge(self, v, u):
        """
        Add an edge to the graph

        :param v : source vertex
        :param u : destination vertex

        :return : 
        """
        self.edges[v].add(u)

    def add_vertex(self, v):
        """
        Add a vertex to the graph

        :param v : new vertex
        
        :return : 
        """
        self.edges[v]

    def __repr__(self):
        """
        Represent the graph as a string of tuples
        """
        return str(self.get_edges())

    def plot_graph(self, with_labels=True):
        """
        Method to visualize graph or sub_graph

        :return: Plot
        """

        g = nx.DiGraph(self.edges)
        plt.figure(figsize=(12, 8))
        plt.clf()
        nx.draw(g, with_labels=with_labels)
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
        pages_seen = set()
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
            # Update pages_seen with the pages that have been seen in this click
            pages_seen.update(new_queue | last_nodes)
            # Update the number of clicks done
            clicks += 1

        # Return the unique pages
        return set(pages_seen)
