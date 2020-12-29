from collections import defaultdict

class Graph:
    
    def __init__(self):
        """
        Initialize the Graph object

        :attribute edges : dictionary of vertices (keys) to sets of vertices (values)
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
        return round(self.n_edges_ / (n_v * (n_v - 1)), 3)


    def in_degree(self, v):
        """
        Compute the in-degree of a node v

        :param v : a node of the graph
        :return in_d : int
        """
        in_d = 0
        for u in self._edges:
            if v in self._edges[u]:
                in_d += 1
        return in_d


    def out_degree(self, v):
        """
        Compute the out-degree of a node v

        :param v : a node of the graph
        :return  : int
        """
        return len(self._edges[v])


    def degree(self, v):
        """
        Compute the degree of a node v

        :param v : a node of the graph
        :return  : int
        """
        return self.in_degree(v) + self.out_degree(v)


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


    def __repr__(self):
        """
        Represent the graph as the list of its edges
        """
        return str(self.get_edges())