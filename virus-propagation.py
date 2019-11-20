# Project 5: Virus Propagation on Static Networks

# Team Members:
# 1. Amal Sony (asony)
# 2. Prayani Singh (psingh25)
# 3. Tanmaya Nanda (tnanda)

import networkx as nx
import numpy as np
import os
import matplotlib.pyplot as plt
import random
from operator import itemgetter
import scipy

def calcEffectiveStrength(b , d, lambda1):
    Cvpm = b/d
    s = np.real(lambda1 * Cvpm)
    return s

def plotStrengthVsBeta(b, d, lambda1, id):
    print('Plotting Effective Strength vs Transmission probability parameter(b), keeping healing probability parameter(d) constant')
    beta_values = np.arange(0, b+0.001, 0.001)
    s = [lambda1*(beta/d) for beta in beta_values]
    plt.figure(figsize=(12, 6))
    plt.plot(beta_values, s, "-o")
    plt.title("Effective Strength vs Transmission probability")
    plt.xlabel('Beta')
    plt.ylabel('Effective Strength')
    plotFilePath = os.getcwd() + "/output/" + "Strength-VS-Beta_"+ id +".png"
    if not os.path.exists(os.path.dirname(plotFilePath)):
        os.makedirs(os.path.dirname(plotFilePath))
    plt.savefig(plotFilePath, bbox_inches='tight')


def plotStrengthVsDelta(b, d, lambda1, id):
    print('Plotting Effective Strength vs Healing probability parameter(d), keeping transmission probability parameter(b) constant')
    delta_values = np.arange(0.01, d+0.1, 0.1)
    s = [lambda1 * (b / delta) for delta in delta_values]
    plt.figure(figsize=(12, 6))
    plt.plot(delta_values, s, "-o")
    plt.title("Effective Strength vs Healing Probability")
    plt.xlabel('Delta')
    plt.ylabel('Effective Strength')
    plotFilePath = os.getcwd() + "/output/" + "Strength-VS-Delta_"+ id +".png"
    if not os.path.exists(os.path.dirname(plotFilePath)):
        os.makedirs(os.path.dirname(plotFilePath))
    plt.savefig(plotFilePath, bbox_inches='tight')

def plotAvgNodesInfectedVsTime(graph, b, d, id):
    print('Plotting average fraction of infected nodes for 100 steps')
    simulation_output = list()
    # Run the simulation 10 times
    for i in range(10):
        f = simulation(graph, b, d)
        simulation_output.append(f)

    # Calucate average of fraction of infected nodes at every time step
    avg_fraction_of_infected_nodes = [float(sum(col)) / len(col) for col in zip(*simulation_output)]
    plt.figure(figsize=(12, 6))
    plt.plot(avg_fraction_of_infected_nodes)
    plt.title("Average Fraction of Infected Nodes Over Time")
    plt.xlabel("Time")
    plt.ylabel("Average fraction of infected nodes")
    plotFilePath = os.getcwd() + "/output/" + "Infected-Nodes-VS-Time_" + id + ".png"
    if not os.path.exists(os.path.dirname(plotFilePath)):
        os.makedirs(os.path.dirname(plotFilePath))
    plt.savefig(plotFilePath, bbox_inches='tight')

def simulation(graph,b,d):
    # Finding fraction of infected nodes at every step by running the sumulation from t = 0 to t = 100
    nodes = graph.nodes()
    fractions_of_infected_nodes = list()
    # At t=0, 1/10th of the nodes are infected
    infected_nodes = set(np.random.choice(nodes, int(len(nodes)/10), replace = False))
    fractions_of_infected_nodes.append(len(infected_nodes) / len(nodes))
    # Find number of infected nodes at the remaining 99 time steps
    for i in range(99):
        recovered_nodes=set(np.random.choice(list(infected_nodes), int(d*len(infected_nodes))))
        infected_nodes=infected_nodes.difference(recovered_nodes)
        # Finding the neighbors of the infected nodes that got infected
        for node in infected_nodes:
            neighbors=graph.neighbors(node)
            susceptible_nodes = set(neighbors).difference(infected_nodes)
            if(len(susceptible_nodes)) > 0:
                infected_nodes=infected_nodes.union(set(np.random.choice(list(susceptible_nodes), int(b*len(list(susceptible_nodes))))))
        fractions_of_infected_nodes.append(float(len(infected_nodes))/float(len(nodes)))
    return fractions_of_infected_nodes

def applyPolicy(graph, k , policy):
    if policy == "A":
        # Immunize and remove random k nodes from the network
        kRandomNodes = random.sample(range(0, nx.number_of_nodes(graph)), k)
        graph.remove_nodes_from(kRandomNodes)
    elif policy == "B":
        # Immunize and remove k highest degree nodes from the network
        kHighestDegreeNodes = [node for node,degree in sorted(graph.degree(),key=itemgetter(1),reverse=True)][:k]
        graph.remove_nodes_from(kHighestDegreeNodes)
    elif policy == "C":
        # Immunize and remove the highest degree node iteratively(k times) from the network
        for i in range(k):
            highestDegreeNode = max(graph.degree(), key=lambda node_degree: node_degree[1])[0]
            graph.remove_node(highestDegreeNode)
    elif policy == "D":
        # Immunize and remove the nodes corresponding to the k largest values in the eigen vector of the largest eigen value
        largestEigenValue, largestEigenVector = scipy.sparse.linalg.eigs( nx.to_numpy_matrix(graph), k=1, which='LM', return_eigenvectors=True)
        absValuesWithIndex = []
        for index, value in enumerate(largestEigenVector):
            absValuesWithIndex.append((index, abs(value)))
            correspondingNodes = [absValueWithIndex[0] for absValueWithIndex in sorted(absValuesWithIndex, key = lambda absValueWithIndex : absValueWithIndex[1], reverse = True)][:k]
        graph.remove_nodes_from(correspondingNodes)
    return graph

def plotStrengthVsVaccines(b, d, graph, policy):
    print('\n\nPlotting Effective Strength vs Number of Available Vaccines for policy ',policy,' network')
    n = nx.number_of_nodes(graph)
    inc = 200 if policy == "D" else 50
    k_values = [k for k in range(0, n, inc)]

    found_min = False
    s_values = []
    k_min = n

    for k in k_values:
        new_graph = applyPolicy(graph.copy(), k, policy) 
        lambda1 = scipy.sparse.linalg.eigs( nx.to_numpy_matrix(new_graph), k=1, which='LM')[0]
        s = calcEffectiveStrength(b, d, lambda1)
        if s < 1 and found_min == False:
            found_min = True
            k_min = k
        s_values.append(s)

    print("Minimum number of vaccines required to prevent epidemic: ", k_min)

    plt.figure(figsize=(12, 6))
    plt.plot(k_values, s_values, "-o")
    plt.title("Effective Strength V/S Available Vaccines")
    plt.xlabel('Available Vaccines')
    plt.ylabel('Effecive Strength')
    plt.axhline(y=1, linewidth=2, color="r")
    plt.text(50, 1.2 , 'Epidemic threshold i.e s=1')
    plotFilePath = os.getcwd() + "/output/" + "StrengthVsVaccines_policy"+ policy +".png"
    if not os.path.exists(os.path.dirname(plotFilePath)):
        os.makedirs(os.path.dirname(plotFilePath))
    plt.savefig(plotFilePath, bbox_inches='tight')

def printEffectiveStrength(s):
    print("Effective strength: ", s)
    if s > 1:
        print("The infection will spread across the network and result in an epidemic.")
    else:
        print("The virus will die quickly and will not result in an epidemic")

if __name__ == "__main__":
    # Read from the file and create a graph
    graph = nx.Graph()
    filePath = os.getcwd()+ "/static.network"
    f = open(filePath, 'r')
    next(f)
    for line in f:
        line = line.split()
        graph.add_edge(int(line[0]),int(line[1]))

    # Calculating the largest eigen value of the adjacency matrix of the network
    lambda1 = scipy.sparse.linalg.eigs( nx.to_numpy_matrix(graph), k=1, which='LM')[0]

    b = 0.20
    d = 0.70

    print("Calculating effective strength for beta = 0.20 and delta = 0.70")
    s = calcEffectiveStrength(b, d, lambda1)
    printEffectiveStrength(s)
    min_beta = np.real(d / lambda1)
    print("Minimum transmission probability parameter for the virus to cause an epidemic: ",min_beta)
    max_delta = np.real(b * lambda1)
    print("Maximum healing probability parameter for the virus to cause an epidemic: ",max_delta)
    plotStrengthVsBeta(b, d, lambda1, "case1")
    plotStrengthVsDelta(b, d, lambda1, "case1")
    # Simulation of virus propagation across network for beta = 0.20 and delta = 0.70
    plotAvgNodesInfectedVsTime(graph, b, d, "case1")

    b = 0.01
    d = 0.60

    print("\n\nCalculating effective strength for beta = 0.01 and delta = 0.60")
    s = calcEffectiveStrength(b, d, lambda1)
    printEffectiveStrength(s)
    min_beta = np.real(d / lambda1)
    print("Minimum transmission probability for the virus to cause an epidemic: ",min_beta)
    max_delta = np.real(b * lambda1)
    print("Maximum healing probability for the virus to cause an epidemic: ",max_delta)
    plotStrengthVsBeta(b, d, lambda1, "case2")
    plotStrengthVsDelta(b, d, lambda1, "case2")
    # Simulation of virus propagation across network for beta = 0.20 and delta = 0.70
    plotAvgNodesInfectedVsTime(graph, b, d, "case2")

    # Applying Immunization Policy, calculating effeective strength of the resultant network, plotting avg infected nodes over time
    k = 200
    b = 0.20
    d = 0.70

    new_graph = applyPolicy(graph.copy(), k, "A")
    print("\n\nCalculating effective strength on policy A network")  
    lambda1 = scipy.sparse.linalg.eigs( nx.to_numpy_matrix(new_graph), k=1, which='LM')[0]
    s = calcEffectiveStrength(b, d, lambda1)
    printEffectiveStrength(s)
    plotAvgNodesInfectedVsTime(new_graph, b, d, "policyA")

    new_graph = applyPolicy(graph.copy(), k, "B")
    print("\n\nCalculating effective strength on policy B network")
    lambda1 = scipy.sparse.linalg.eigs( nx.to_numpy_matrix(new_graph), k=1, which='LM')[0]
    s = calcEffectiveStrength(b, d, lambda1)
    printEffectiveStrength(s)
    plotAvgNodesInfectedVsTime(new_graph, b, d, "policyB")

    new_graph = applyPolicy(graph.copy(), k, "C")
    print("\n\nCalculating effective strength on policy C network")
    lambda1 = scipy.sparse.linalg.eigs( nx.to_numpy_matrix(new_graph), k=1, which='LM')[0]
    s = calcEffectiveStrength(b, d, lambda1)
    printEffectiveStrength(s)
    plotAvgNodesInfectedVsTime(new_graph, b, d, "policyC")


    new_graph = applyPolicy(graph.copy(), k, "D")
    print("\n\nCalculating effective strength on policy D network")
    lambda1 = scipy.sparse.linalg.eigs( nx.to_numpy_matrix(new_graph), k=1, which='LM')[0]
    s = calcEffectiveStrength(b, d, lambda1)
    printEffectiveStrength(s)
    plotAvgNodesInfectedVsTime(new_graph, b, d, "policyD")
    
    # Estimating minimum number of vaccines needed to prevent epidemic for each immunization policy
    plotStrengthVsVaccines(b, d, graph.copy(), "A")
    plotStrengthVsVaccines(b, d, graph.copy(), "B")
    plotStrengthVsVaccines(b, d, graph.copy(), "C")
    plotStrengthVsVaccines(b, d, graph.copy(), "D")
