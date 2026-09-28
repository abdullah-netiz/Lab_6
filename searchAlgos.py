import heapq
import math

import networkx as nx


# Task 1: Warehouse Robot
warehouse_locations = {
    "Receiving_Area": (0, 0),
    "Storage_B": (1, 4),
    "Storage_A": (2, 1),
    "Sorting_Area": (4, 2),
    "Inspection_Area": (5, 5),
    "Packing_Station": (7, 6),
}

warehouse_graph = {
    "Receiving_Area": {"Storage_B": 4.1, "Storage_A": 2.2},
    "Storage_B": {"Inspection_Area": 5.0, "Sorting_Area": 6.0},
    "Storage_A": {"Sorting_Area": 2.2},
    "Sorting_Area": {"Inspection_Area": 3.2, "Packing_Station": 5.0},
    "Inspection_Area": {"Packing_Station": 2.2},
    "Packing_Station": {},
}


# Task 2: Airport Baggage Handling
airport_locations = {
    "Baggage_Area": (0, 0),
    "Security": (2, 1),
    "Checkpoint": (1, 4),
    "Food_Court": (4, 2),
    "Terminal_Hall": (5, 5),
    "Departure_Gate": (7, 6),
}

airport_graph = {
    "Baggage_Area": {"Security": 2.2, "Checkpoint": 4.1},
    "Security": {"Food_Court": 2.2},
    "Checkpoint": {"Terminal_Hall": 5.0},
    "Food_Court": {"Terminal_Hall": 3.2, "Departure_Gate": 6.0},
    "Terminal_Hall": {"Departure_Gate": 3.2},
    "Departure_Gate": {},
}


# Task 3: Emergency Supply Robot
locations = {
    "Pharmacy": (0, 0),
    "Main_Corridor": (2, 1),
    "Patient_Wing": (1, 4),
    "Nursing_Station": (4, 2),
    "Laboratory": (5, 5),
    "Emergency_Ward": (8, 6),
}

hospital_graph = {
    "Pharmacy": {"Main_Corridor": 2.2, "Patient_Wing": 4.1},
    "Main_Corridor": {"Nursing_Station": 2.2},
    "Patient_Wing": {"Laboratory": 5.0},
    "Nursing_Station": {"Laboratory": 3.2, "Emergency_Ward": 5.0},
    "Laboratory": {"Emergency_Ward": 3.2},
    "Emergency_Ward": {},
}


# Task 4: Autonomous Delivery Drone
drone_locations = {
    "Distribution_Center": (0, 0),
    "Zone_A": (2, 1),
    "Zone_B": (1, 4),
    "Zone_C": (4, 2),
    "Zone_D": (5, 5),
    "Customer_Building": (7, 6),
}

drone_graph = {
    "Distribution_Center": {"Zone_A": 2.2, "Zone_B": 3.0},
    "Zone_A": {"Zone_C": 2.2},
    "Zone_B": {"Zone_D": 5.0},
    "Zone_C": {"Customer_Building": 6.0},
    "Zone_D": {"Customer_Building": 3.2},
    "Customer_Building": {},
}


def heuristic(node, goal, coordinates):
    """Return Euclidean distance from node to goal."""
    x1, y1 = coordinates[node]
    x2, y2 = coordinates[goal]
    return math.hypot(x2 - x1, y2 - y1)


def reconstruct_path(came_from, current):
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return path


def greedy_best_first_search(start, goal, graph, coordinates):
    """Run GBFS with f(n) = h(n)."""
    frontier = [(heuristic(start, goal, coordinates), 0, start)]
    came_from = {}
    visited = set()
    expansion_order = []
    counter = 1

    while frontier:
        _, _, current = heapq.heappop(frontier)
        if current in visited:
            continue
        visited.add(current)
        expansion_order.append(current)
        if current == goal:
            path = reconstruct_path(came_from, current)
            return path, path_cost(path, graph), expansion_order

        for neighbor in graph[current]:
            if neighbor not in visited:
                came_from.setdefault(neighbor, current)
                heapq.heappush(
                    frontier,
                    (heuristic(neighbor, goal, coordinates), counter, neighbor),
                )
                counter += 1

    return None, math.inf, expansion_order


def a_star_search(start, goal, graph, coordinates):
    """Run A* with f(n) = g(n) + h(n)."""
    frontier = [(heuristic(start, goal, coordinates), 0, start)]
    came_from = {}
    g_cost = {start: 0.0}
    expansion_order = []
    counter = 1

    while frontier:
        _, _, current = heapq.heappop(frontier)
        if current in expansion_order:
            continue
        expansion_order.append(current)
        if current == goal:
            path = reconstruct_path(came_from, current)
            return path, g_cost[current], expansion_order, g_cost

        for neighbor, cost in graph[current].items():
            tentative_g = g_cost[current] + cost
            if tentative_g < g_cost.get(neighbor, math.inf):
                came_from[neighbor] = current
                g_cost[neighbor] = tentative_g
                score = tentative_g + heuristic(neighbor, goal, coordinates)
                heapq.heappush(frontier, (score, counter, neighbor))
                counter += 1

    return None, math.inf, expansion_order, g_cost


def gbfs(start, goal, graph, coordinates):
    return greedy_best_first_search(start, goal, graph, coordinates)


def a_star(start, goal, graph, coordinates):
    return a_star_search(start, goal, graph, coordinates)


def weighted_a_star(start, goal, graph, coordinates, weight):
    """Run Weighted A* with f(n) = g(n) + weight * h(n)."""
    frontier = [(weight * heuristic(start, goal, coordinates), 0, start)]
    came_from = {}
    g_cost = {start: 0.0}
    expansion_order = []
    counter = 1

    while frontier:
        _, _, current = heapq.heappop(frontier)
        if current in expansion_order:
            continue
        expansion_order.append(current)
        if current == goal:
            path = reconstruct_path(came_from, current)
            return path, g_cost[current], expansion_order

        for neighbor, cost in graph[current].items():
            tentative_g = g_cost[current] + cost
            if tentative_g < g_cost.get(neighbor, math.inf):
                came_from[neighbor] = current
                g_cost[neighbor] = tentative_g
                score = tentative_g + weight * heuristic(neighbor, goal, coordinates)
                heapq.heappush(frontier, (score, counter, neighbor))
                counter += 1

    return None, math.inf, expansion_order


def path_cost(path, graph):
    if not path:
        return math.inf
    return sum(graph[source][target] for source, target in zip(path, path[1:]))


def to_networkx(graph, coordinates):
    network = nx.DiGraph()
    network.add_nodes_from((node, {"pos": position}) for node, position in coordinates.items())
    for node, neighbors in graph.items():
        for neighbor, cost in neighbors.items():
            network.add_edge(node, neighbor, weight=cost)
    return network


def verify_heuristic_condition(graph, coordinates, goal):
    checks = []
    for source, neighbors in graph.items():
        for target, cost in neighbors.items():
            left = heuristic(source, goal, coordinates)
            right = cost + heuristic(target, goal, coordinates)
            checks.append({
                "edge": f"{source} -> {target}",
                "h(n)": left,
                "cost + h(m)": right,
                "satisfied": left <= right + 1e-9,
            })
    return checks


# Backward-compatible Task 1 names.
LOCATIONS = warehouse_locations
EDGES = [
    (source, target, cost)
    for source, neighbors in warehouse_graph.items()
    for target, cost in neighbors.items()
]


def build_warehouse_graph():
    return to_networkx(warehouse_graph, warehouse_locations)


def warehouse_heuristic(location, goal="Packing_Station"):
    return heuristic(location, goal, warehouse_locations)


def heuristic_table():
    return {node: warehouse_heuristic(node) for node in warehouse_locations}


def draw_warehouse_graph(graph=None, axis=None):
    import matplotlib.pyplot as plt

    graph = graph or build_warehouse_graph()
    if axis is None:
        _, axis = plt.subplots(figsize=(10, 6))
    positions = nx.get_node_attributes(graph, "pos")
    nx.draw_networkx(graph, positions, ax=axis, node_color="#9ec5fe", node_size=1600,
                     arrows=True, arrowsize=18, font_size=8)
    nx.draw_networkx_edge_labels(graph, positions,
                                 edge_labels=nx.get_edge_attributes(graph, "weight"), ax=axis)
    axis.set_title("Warehouse Robot Graph")
    axis.set_axis_off()
    return axis
