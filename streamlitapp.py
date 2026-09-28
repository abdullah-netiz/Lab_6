import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import streamlit as st

from searchAlgos import (
    airport_graph,
    airport_locations,
    a_star,
    drone_graph,
    drone_locations,
    gbfs,
    heuristic,
    hospital_graph,
    locations,
    to_networkx,
    verify_heuristic_condition,
    warehouse_graph,
    warehouse_locations,
    weighted_a_star,
)


st.set_page_config(page_title="Search Algorithms Lab", layout="wide")
st.title("Search Algorithms for Graph Navigation")
st.write("Explore Euclidean heuristics, GBFS, A*, and Weighted A* on the supplied problem graphs.")


def draw_search_graph(graph, coordinates, path=None, title="Graph"):
    network = to_networkx(graph, coordinates)
    path_edges = set(zip(path, path[1:])) if path else set()
    node_colors = ["#f08a5d" if node in (path or []) else "#9ec5fe" for node in network.nodes]
    edge_colors = ["#e63946" if edge in path_edges else "#777777" for edge in network.edges]
    widths = [3.5 if edge in path_edges else 1.2 for edge in network.edges]

    figure, axis = plt.subplots(figsize=(10, 6))
    nx.draw_networkx_nodes(network, coordinates, node_color=node_colors, node_size=1800, ax=axis)
    nx.draw_networkx_labels(network, coordinates, font_size=8, ax=axis)
    nx.draw_networkx_edges(network, coordinates, edge_color=edge_colors, width=widths,
                           arrows=True, arrowsize=18, connectionstyle="arc3,rad=0.04", ax=axis)
    nx.draw_networkx_edge_labels(
        network, coordinates, edge_labels=nx.get_edge_attributes(network, "weight"), ax=axis
    )
    axis.set_title(title)
    axis.set_axis_off()
    figure.tight_layout()
    return figure


def show_search_result(algorithm, path, cost, expansion_order, graph, coordinates, g_cost=None):
    if path is None:
        st.error("No path was found between the selected nodes.")
        return
    st.subheader("Search Result")
    st.write(f"**Algorithm:** {algorithm}")
    st.write(f"**Solution Path:** {' -> '.join(path)}")
    st.write(f"**Total Path Cost:** {cost:.2f}")
    st.write(f"**Expansion Order:** {' -> '.join(expansion_order)}")
    st.pyplot(draw_search_graph(graph, coordinates, path, f"{algorithm} Solution Path"),
              use_container_width=True)
    if g_cost is not None:
        rows = []
        goal = path[-1]
        for node in expansion_order:
            h_value = heuristic(node, goal, coordinates)
            rows.append({"Node": node, "g(n)": g_cost[node], "h(n)": h_value,
                         "f(n)": g_cost[node] + h_value})
        st.subheader("A* Node Information")
        st.dataframe(pd.DataFrame(rows).round(2), hide_index=True, use_container_width=True)


st.sidebar.header("Interactive Search")
selected_graph = st.sidebar.selectbox("Select graph", ("Airport", "Hospital"))
if selected_graph == "Airport":
    interactive_graph = airport_graph
    interactive_locations = airport_locations
    default_start = "Baggage_Area"
    default_goal = "Departure_Gate"
else:
    interactive_graph = hospital_graph
    interactive_locations = locations
    default_start = "Pharmacy"
    default_goal = "Emergency_Ward"

interactive_nodes = list(interactive_graph)
start = st.sidebar.selectbox(
    "Select Initial Node", interactive_nodes,
    index=interactive_nodes.index(default_start),
)
goal = st.sidebar.selectbox(
    "Select Goal Node", interactive_nodes,
    index=interactive_nodes.index(default_goal),
)
algorithm = st.sidebar.selectbox("Select Search Algorithm", ("GBFS", "A*"))

if st.sidebar.button("Run Search"):
    if algorithm == "GBFS":
        path, cost, expansion_order = gbfs(start, goal, interactive_graph, interactive_locations)
        g_cost = None
    else:
        path, cost, expansion_order, g_cost = a_star(
            start, goal, interactive_graph, interactive_locations
        )
    st.header("Interactive Search Result")
    show_search_result(
        algorithm, path, cost, expansion_order, interactive_graph,
        interactive_locations, g_cost,
    )


task1, task2, task3, task4 = st.tabs(["Task 1: Warehouse", "Task 2: GBFS", "Task 3: A*", "Task 4: Weighted A*"])

with task1:
    st.header("Warehouse Robot Heuristic")
    goal = "Packing_Station"
    values = pd.DataFrame([{"Location": node, "h(n)": heuristic(node, goal, warehouse_locations)}
                           for node in warehouse_locations]).round(2)
    st.dataframe(values, hide_index=True, use_container_width=True)
    checks = pd.DataFrame(verify_heuristic_condition(warehouse_graph, warehouse_locations, goal))
    st.subheader("Heuristic Condition: h(n) <= cost(n,m) + h(m)")
    st.dataframe(checks.round(2), hide_index=True, use_container_width=True)
    st.pyplot(draw_search_graph(warehouse_graph, warehouse_locations, title="Warehouse Graph"),
              use_container_width=True)
    st.write("Euclidean distance is suitable because the coordinates represent physical locations and provide a direct estimate of remaining travel distance.")

with task2:
    st.header("Task 2: Greedy Best-First Search")
    path, cost, expansion_order = gbfs("Baggage_Area", "Departure_Gate", airport_graph, airport_locations)
    show_search_result("GBFS", path, cost, expansion_order, airport_graph, airport_locations)
    st.write("GBFS expands the node with the smallest h(n), so it follows locations that appear closest to the Departure Gate without considering the cost already travelled.")

with task3:
    st.header("Task 3: A* Emergency Supply Robot")
    path, cost, expansion_order, g_cost = a_star("Pharmacy", "Emergency_Ward", hospital_graph, locations)
    show_search_result("A*", path, cost, expansion_order, hospital_graph, locations, g_cost)
    st.write("A* combines actual cost g(n) with estimated remaining cost h(n). This lets it avoid a node that looks close to the goal when reaching that node has already been expensive.")

with task4:
    st.header("Task 4: Weighted A* Delivery Drone")
    rows = []
    results = {}
    for weight in (1, 1.5, 2, 3):
        path, cost, expansion_order = weighted_a_star(
            "Distribution_Center", "Customer_Building", drone_graph, drone_locations, weight
        )
        results[weight] = (path, cost, expansion_order)
        rows.append({"Weight w": weight, "Solution Path": " -> ".join(path),
                     "Total Cost": cost, "Nodes Expanded": len(expansion_order),
                     "Expansion Order": " -> ".join(expansion_order)})
    st.dataframe(pd.DataFrame(rows).round(2), hide_index=True, use_container_width=True)
    selected_weight = st.selectbox("Select weight to visualize", (1, 1.5, 2, 3))
    path, cost, expansion_order = results[selected_weight]
    st.pyplot(draw_search_graph(
        drone_graph, drone_locations, path,
        f"Weighted A* | w = {selected_weight} | Cost = {cost:.2f}"
    ), use_container_width=True)
    st.write("When w increases, the search trusts h(n) more than g(n). This can reduce expansions, but w > 1 does not guarantee an optimal path.")
