1. Introduction
    2. Why?
    3. What?
    4. Background recommended
        5. Basic probability
            6. Sum of exclusive = 1
            7. 
        6. Basic graph theory
            7. Nodes (N) and Edges (V = (N × N))
            8. Adjacency Matrix view of graphs
            8. Directed and Undirected graphs
            9. Directed Acyclic Graphs
2. Assumptions 
    2. Fixed set of nodes
    3. Discrete time
    4. Synchronous activation
    5. Trial based temporal independence
2. Graphs: Structure
    1. Complexity of graph enumeration
        2. General directed graphs, $$2^{n^2}$$
    2. Reducing complexity:
        3. Enumeration filters 
        3. Directed Acyclic Graphs
            4. No trace (no self-loops)
            4. number of graphs
    5. Parents and children
3. Random Variables: Semantics, sampling and graphs
    4. Conditional probability distributions
    5. Conditional independence properties
    6. Graphical interpretation of conditional independence
4. Causal Graphs: Interventions
    1. Graph Surgery
    2. Causal graphs as extensions of directed graphs --- 
        1. incorporating intervention into the node set \cite{winn2012causality}
    3. Interventions as constraints on the graph set
        4. Node has no parents = node is intervened on with prior distribution equal to the 
4. Causal Bayesian NetworkX
    5. Iterator over graphs
    6. Closures for constraints
    7. 

