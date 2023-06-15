# ASSIGNMENT 5
import numpy as np
import random as rd
import math
from matplotlib import pyplot as plt
rd.seed(10)
"""""
steps:
1. place ants at different nodes
2. find path for each ant
3. remove cycles
3. update phermones(tau)
4. go to step 2 until stopping criteria(number of iterations=20 times)

"""
# coordinates of each city  "pandas library doesn't work so i had to write them"
x = [0, 0.009174316, 0.064220188, 0.105504591, 0.105504591, 0.105504591, 0.185779816, 0.240825687, 0.254587155, 0.38302753, 0.394495416, 0.419724769, 0.458715603, 0.593922025, 0.729357795,
     0.731651377, 0.749999997, 0.770642198, 0.786697249, 0.811926602, 0.852217125, 0.861850152, 0.869762996, 0.871559638, 0.880733941, 0.880733941, 0.885321106, 0.908256877, 0.912270643, 0]
y = [1, 0.995412849, 0.438073402, 0.594036699, 0.706422024, 0.917431193, 0.454128438, 0.614678901, 0.396788998, 0.830275235, 0.839449537, 0.646788988, 0.470183489, 0.348509173,
     0.642201837, 0.098623857, 0.403669732, 0.495412842, 0.552752296, 0.254587155, 0.442928131, 0.493004585, 0.463761466, 0, 0.08486239, 0.268348623, 0.075688073, 0.353211012, 0.43470948, 0]


# find distance between each city and all other cities using rule of distance
def Euclidean_distance(x, y, n_cities):

    distance = np.full((n_cities, n_cities), 100, dtype='float64')

    for i in range(n_cities):
        for j in range(n_cities):
            if (i != j):
                distance[i][j] = np.round(
                    math.sqrt((x[i]-x[j])**2+(y[i]-y[j])**2), 5)
    return distance


def Eta(n_cities, distance):  # calculate eta using rule(eta[i]=1/distance[i])
    eta = np.full((n_cities, n_cities), 1000000, dtype='float64')
    for i in range(n_cities):
        for j in range(n_cities):
            eta[i][j] = 1 / distance[i][j]
    return eta


# calculate the tour length and path (select next node based on the nearest city)
def Lnn_tour_length(n_cities, Distance, start_node):
    distance = Distance.copy()
    tour_Len = 0
    path, visited = [], []
    path.append(start_node)
    visited.append(start_node)
    next_node = np.argmin(distance[start_node], axis=0)
    path.append(next_node)
    visited.append(next_node)
    while True:
        next_node = np.argmin(distance[next_node], axis=0)
        if next_node in path:
            # to avoid duplicates set the distance between the last visited node and the duplicated node = very high value  -> path[-1] means the last visited node
            distance[path[-1]][next_node] = 10000000
            continue  # to skip this iteration and start new one to find the new minimum distance
        tour_Len = tour_Len + distance[path[-1]][next_node]
        path.append(next_node)
        if next_node not in visited:
            visited.append(next_node)
        if len(visited) == n_cities:  # if all cities were visited and the salesman returned to the start city , that means he constructed a complete tour
            return path, tour_Len


# calculate the tour length and path (select the next node based on probability)
def Tour_length(n_cities, Distance, start_node, probability):
    distance = Distance.copy()
    tour_Len = 0
    path, visited = [], []
    path.append(start_node)
    visited.append(start_node)
    # select the next node based on probability
    next_node1 = np.argmax(probability[start_node])
    path.append(next_node1)
    visited.append(next_node1)

    while True:
        next_node2 = np.argmax(probability[next_node1])
        probability[next_node1][next_node2] = -100

        tour_Len = tour_Len + distance[path[-1]][next_node2]
        path.append(next_node2)

        if next_node2 not in visited:
            visited.append(next_node2)

        if len(visited) == n_cities:  # if all cities were visited and the salesman returned to the start city , that means he constructed a complete tour

            return path, tour_Len
        next_node1 = next_node2


# calculate the probability of selecting the next node
def probability_of_next_node(n_cities, tau, eta, alpha, beta):
    p = np.full((n_cities, n_cities), 0, dtype='float64')
    sum = 0
    for i in range(n_cities):
        for j in range(n_cities):
            if (i != j):
                p[i][j] = np.round(tau[i][j]**alpha + eta[i][j]**beta, 5)
                sum += p[i][j]

    for i in range(n_cities):
        for j in range(n_cities):
            if (i != j):
                p[i][j] = np.round(p[i][j]/sum, 5)

    return p


# remove cycles from a path and return acyclic path & its length
def remove_cycles(Path, n_cities, distance):
    acyclic_tour = []
    tour_length = 0
    path = Path.copy()
    l = 0
    while True:
        break_out_flag = False
        for i in range(len(path)):
            for j in range(len(path)):
                if (path[i] == path[j]) & (i != j):
                    l = j
                    path[i+1:j+1] = []
                    break_out_flag = True
                    break
            if break_out_flag:
                break
            acyclic_tour.append(path[i])

        if len(acyclic_tour) == n_cities:
            for I in range(len(acyclic_tour)-1):
                tour_length += distance[path[I]][path[I+1]]

            return acyclic_tour, tour_length
        else:
            Path.pop(l)
            path = Path.copy()

        acyclic_tour = []


# calculate the initial valaues of phermones using (nearest neigbour method)
def initial_tau(n_cities, Lnn):
    Lnn = 1/(n_cities*Lnn)
    initial_pheromones = np.full((n_cities, n_cities), Lnn, dtype='float64')
    return initial_pheromones


# using final equation to increase phermone and decrease it with evaporation
def update_tau(last_tau, path, alpha, ruo):
    for i in range(len(path)-1):
        last_tau[i][i+1] = (1-ruo) * last_tau[i][i +
                                                 1] ** alpha + last_tau[i][i+1]
    return last_tau


# to Generate m ants and place them over the cities, but make sure that no city has more than one ant.
def ACS_TSP(x, y, n_cities, m_ants, n_iteration, alpha, beta, ruo):
    distance = Euclidean_distance(x, y, n_cities)
    eta = Eta(n_cities, distance)
    start_node = np.random.randint(0, n_cities-1)
    path, tour_length = Lnn_tour_length(n_cities, distance, start_node)
    path, tour_length = remove_cycles(path, n_cities, distance)
    init_tau = initial_tau(n_cities, tour_length)
    new_tau = update_tau(init_tau, path, alpha, ruo)
    ants_init_location, shortest_tours = [], []
    ants_init_location.append(start_node)
    shortest_tours.append(tour_length)
    for i in range(n_iteration):
        # place ants at different nodes (loop until reach to node that has no ants and place ant at it)
        while True:
            start_node = np.random.randint(0, n_cities-1)
            if start_node not in ants_init_location:
                ants_init_location.append(start_node)
                break
        probability = probability_of_next_node(
            n_cities, new_tau, eta, alpha, beta)  # calculate probability of selecting next node
        path, tour_length = Tour_length(
            n_cities, distance, start_node, probability)  # find path & tour length (may has cycles)
        path, tour_length = remove_cycles(
            path, n_cities, distance)   # remove cycles
        # update phermones(tau)
        new_tau = update_tau(new_tau, path, alpha, ruo)
        # store value of shortest tour for each ant
        shortest_tours.append(tour_length)

    # to plot initial location of the cities
    print("Initial location of the cities : ", ants_init_location, "\n")
    xpoints = np.arange(1, 21, 1)
    ypoints = ants_init_location
    print("shortest tours  : ", shortest_tours)
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    font1 = {'family': 'serif', 'color': 'blue', 'size': 20}
    font2 = {'family': 'serif', 'color': 'darkred', 'size': 15}
    ax.set_xlabel('Ant number', font2)
    ax.set_ylabel('Initial location of the cities', font2)
    plt.xticks(np.arange(min(xpoints), max(xpoints)+1, 1.0))
    plt.plot(xpoints, ypoints)
    plt.title("plot", font1)
    plt.show()

    # to plot the shortest tours for each ant
    fig2 = plt.figure(figsize=(10, 10))
    ax2 = fig2.add_subplot(111)
    ax2.set_xlabel('Ant number', font2)
    ax2.set_ylabel('Shortest Tours', font2)
    plt.xticks(np.arange(min(xpoints), max(xpoints)+1, 1.0))
    plt.plot(shortest_tours)
    plt.title("plot", font1)
    plt.show()


ACS_TSP(x, y, 30, 30, 20, 0.5, 1.5, 0.5)

"""""
After implementation:

ACS_TSP(x, y, 30, 30, 20, 1.5, 3,0.5)
ACS_TSP(x, y, 30, 30, 20, 2, 1,0.2)
ACS_TSP(x, y, 30, 30, 20, 0.5, 1.5,0.1)
ACS_TSP(x, y, 30, 30, 20, 0, 1,0.9)
ACS_TSP(x, y, 30, 30, 20, 0.1, 2,0)
ACS_TSP(x, y, 30, 30, 20, 0.1, 1.5,1)


by making several runs : we can investigate that :
by increasing alpha we give a higher weigth for pheromone concentration
by increasing Beta we give a higher weigth for attractiveness
by increasing Ruo we increase the evaporation



"""
