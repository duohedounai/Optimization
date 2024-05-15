import numpy as np
import copy

from matplotlib import pyplot as plt


def generate_distance_matrix(city_num, city_positon):
    matrix = np.zeros((city_num, city_num))
    for i in range(city_num):
        for j in range(i + 1, city_num):
            matrix[i, j] = calculate_distance(i, j, city_positon)
            matrix[j, i] = matrix[i, j]
    return matrix


def calculate_distance(i, j, city_positon):
    distance = np.sqrt(pow(city_positon[i][0] - city_positon[j][0], 2) \
                       + pow(city_positon[i][1] - city_positon[j][1], 2))
    return distance


def get_neighbors(current_route):
    # Save new solutions
    candidate_solutions = []

    # Try to generate 20 new solutions
    for solution_count in range(20):
        new_solution = copy.deepcopy(current_route)

        # Swap two cities in the old solution
        index_1 = np.random.randint(0, len(current_route))
        while True:
            index_2 = np.random.randint(0, len(current_route))
            if index_1 != index_2:
                break
        new_solution[index_1], new_solution[index_2] = new_solution[index_2], new_solution[index_1]

        if new_solution not in candidate_solutions and str(new_solution) != str(current_route):
            candidate_solutions.append(new_solution)

    return candidate_solutions


def get_route_cost(route, distance_matrix):
    total_distance = 0
    for temp_index in range(len(route)):
        former_city_id = route[temp_index]
        if temp_index == len(route) - 1:
            latter_city_id = route[0]
        else:
            latter_city_id = route[temp_index + 1]
        total_distance += distance_matrix[former_city_id, latter_city_id]
    return total_distance


def select_best_route(neighborhood, candidate_solutions_cost):
    min_cost = float('inf')
    route_selected = None

    for temp_route in neighborhood:
        temp_cost = candidate_solutions_cost[str(temp_route)]
        if temp_cost < min_cost:
            min_cost = temp_cost
            route_selected = copy.deepcopy(temp_route)

    return route_selected, min_cost


# Tabu Search Algorithm
def tabu_search_solver(tabu_length, max_iterations, city_num, distance_matrix, all_best_value_found):
    tabu_map = {}
    current_route = list(range(city_num))
    next_route = copy.deepcopy(current_route)

    # Save the historical optimal solution found
    best_route = copy.deepcopy(current_route)
    best_value = get_route_cost(best_route, distance_matrix)

    for ite_count in range(max_iterations):
        # print("iteration: " + str(ite_count))

        find_better_solution = False

        # Get feasible solutions that are not in the tabu-table
        candidate_solutions = get_neighbors(current_route)
        candidate_solutions_cost_map = {}

        # If a better solution is found than the current optimal solution,
        # the current optimal solution is updated and used as the next solution
        for temp_route in candidate_solutions:
            temp_cost = get_route_cost(temp_route, distance_matrix)
            candidate_solutions_cost_map[str(temp_route)] = temp_cost
            if temp_cost < best_value:
                find_better_solution = True
                best_route = copy.deepcopy(temp_route)
                best_value = temp_cost

                next_route = copy.deepcopy(best_route)

        # Select the best solution from the candidate_solutions
        if not find_better_solution:
            feasible_neighbors = [route for route in candidate_solutions if str(route) not in tabu_map]
            next_route, next_value = select_best_route(feasible_neighbors, candidate_solutions_cost_map)

        tabu_map[str(next_route)] = tabu_length
        current_route = copy.deepcopy(next_route)
        all_best_value_found.append(best_value)

        # Update the tabu-table
        for key, value in list(tabu_map.items()):
            if value < 0:
                del tabu_map[key]
            else:
                tabu_map[key] -= 1

    return best_route, best_value


def plot_results(all_best_value_found, city_num, city_positon, best_route):
    # Plot the cost of the optimal solution found
    x = np.arange(0, len(all_best_value_found))
    plt.title("The cost of the optimal solution found")
    plt.xlabel("Iteration")
    plt.ylabel("Travel cost")
    plt.plot(x, all_best_value_found, color='r')
    plt.show()

    # Plot the final route
    for i in range(city_num):
        plt.plot(city_positon[i][0], city_positon[i][1], 'o', color='b')
    for city_index in range(len(best_route)):
        if city_index + 1 == len(best_route):
            x = [city_positon[best_route[city_index]][0], city_positon[best_route[0]][0]]
            y = [city_positon[best_route[city_index]][1], city_positon[best_route[0]][1]]
            plt.plot(x, y, '--', color='g')
        else:
            x = [city_positon[best_route[city_index]][0], city_positon[best_route[city_index + 1]][0]]
            y = [city_positon[best_route[city_index]][1], city_positon[best_route[city_index + 1]][1]]
            plt.plot(x, y, '--', color='r')
    plt.title("The best solution found by the Tabu Search Algorithm")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()


if __name__ == "__main__":
    # Set the locations of all cities
    city_positon = [(10, 20), (5, 50), (30, 10), (60, 100), (10, 80), (25, 55), (30, 90), (70, 30), (60, 10), (60, 75),
                    (20, 20), (30, 68), (80, 25), (90, 50), (75, 65), (90, 98), (50, 50), (45, 20), (50, 80), (10, 70)]
    city_num = len(city_positon)
    print("city number: " + str(city_num))

    # Generate the distance matrix between cities
    distance_matrix = generate_distance_matrix(city_num, city_positon)
    print("city distance_matrix: \n" + str(distance_matrix))

    # Plot the locations of all cities
    for i in range(city_num):
        plt.plot(city_positon[i][0], city_positon[i][1], 'o', color='b')
    plt.title("Locations of all cities")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()

    # Set the parameters of the tabu search algorithm
    tabu_length = 10
    max_iterations = 2000

    all_best_value_found = []

    # Use taboo search to find the best path
    best_route, best_value = tabu_search_solver(tabu_length, max_iterations, city_num, distance_matrix,
                                                all_best_value_found)
    print(f"Best Route: {best_route}")
    print(f"Best Value: {best_value}")

    # Plot the results
    plot_results(all_best_value_found, city_num, city_positon, best_route)