import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# Fitness function
def objective_function(x):
    z = np.sin(np.power(1 - x[0], 2) + 2 * x[1] + np.cos(np.power(x[0], 2))) + np.power(np.sin(x[0] + x[1]), 2)
    return z


# Draw the objective function
def draw_objective_function():
    figure = plt.figure()
    ax = Axes3D(figure)

    # x, y is generated with an interval of (-10,10) and an interval of 0.1
    x = np.arange(-5, 5, 0.2)
    y = np.arange(-5, 5, 0.2)
    X, Y = np.meshgrid(x, y)

    # objective_function
    Z = np.sin(np.power(1 - X, 2) + 2 * Y + np.cos(np.power(X, 2))) + np.power(np.sin(X + Y), 2)

    plt.xlabel("x")
    plt.ylabel("y")
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap="rainbow")
    plt.show()


# Particle Swarm Optimization
def pso_solver(max_iter, dimension_num, w, c1, c2):
    # Set random seeds to ensure reproducibility
    np.random.seed(0)

    # Parameter initialization
    all_particle_positions = np.random.uniform(-5, 5, size=(particle_num, dimension_num))
    all_particle_velocities = np.random.uniform(-1, 1, size=(particle_num, dimension_num))
    best_position_each_particle = all_particle_positions.copy()
    best_fitness_each_particle = [objective_function(pos) for pos in all_particle_positions]

    # Record the fitness information during the algorithm iteration process
    best_fitness_each_iter = []
    global_best_cost_found = []

    global_best_index = np.argmax(best_fitness_each_particle)
    global_best_fitness = best_fitness_each_particle[global_best_index]

    for iter_count in range(max_iter):
        # Update particle velocity
        for i in range(len(all_particle_positions)):
            r1, r2 = np.random.rand(dimension_num), np.random.rand(dimension_num)
            all_particle_velocities[i] = w * all_particle_velocities[i] \
                                         + c1 * r1 * (best_position_each_particle[i] - all_particle_positions[i]) \
                                         + c2 * r2 * (best_position_each_particle[global_best_index] -
                                                      all_particle_positions[i])

        # Update Particle Position
        all_particle_positions += all_particle_velocities

        # Limits the particle's position coordinates to a given range
        all_particle_positions = np.clip(all_particle_positions, -5, 5)

        # Calculate the fitness of each particle
        all_particle_new_fitness = [objective_function(pos) for pos in all_particle_positions]

        # Update the optimal position and fitness of each particle
        for i in range(len(all_particle_positions)):
            if all_particle_new_fitness[i] > best_fitness_each_particle[i]:
                best_position_each_particle[i] = all_particle_positions[i].copy()
                best_fitness_each_particle[i] = all_particle_new_fitness[i]

        # Update the optimal fitness found
        if np.max(all_particle_new_fitness) > global_best_fitness:
            global_best_index = np.argmax(all_particle_new_fitness)
            global_best_fitness = all_particle_new_fitness[global_best_index]

        best_fitness_each_iter.append(np.min(all_particle_new_fitness))
        global_best_cost_found.append(global_best_fitness)

    return best_position_each_particle[global_best_index], global_best_fitness, \
           best_fitness_each_iter, global_best_cost_found


# PSO Parameter settings
# Number of particles
particle_num = 10
# Maximum number of iterations
max_iter = 200
# Inertia factor
w = 0.8
# Learning Factors (Self-Awareness)
c1 = 2.0
# Learning Factors (Social Cognition)
c2 = 2.0
# Number of dimensions
dimension_num = 2

# Draw the objective function
draw_objective_function()

# Use PSO to find the maximum value of the function
result_position, result_cost, best_fitness_each_iter, global_best_fitness_found = pso_solver(max_iter, dimension_num,
                                                                                             w, c1, c2)

# Plot the optimal fitness curve found
x = np.arange(0, max_iter, 1)
plt.plot(x, global_best_fitness_found)
plt.title("global best fitness found")
plt.show()

print("The optimal (maximum) solution found: ", result_position)
print("The fitness of the found solution: ", result_cost)
