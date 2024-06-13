from typing import Callable
from algorithm import Algorithm
import numpy as np
import random

# TODO: Graficar con promedio y programar centroide, leer articulos relacionados, investigar 2008
# TODO: 2008 y 2014 no existen articulos


def compare(method_1: Callable, method_2: Callable, upper: np.array, lower: np.array):
    """
    Compara dos metodos para el manejo de restricciones de limites y calcula su presicion
    mediante un prueba

    :param method_1: Callable, Metodo de restrccion de limites
    :param method_2: Callable, Metodo de restrccion de limites
    :param upper: np.array, Limite Superior
    :param lower: np.array, Limite Inferior
    :return: tuple, Tupla con las preciciones de cada metodo
    """

    algorithm = Algorithm()

    points = algorithm.generate(upper - 1, lower - 1)

    p1 = np.array([method_1(upper, lower, p) for p in points])
    p2 = np.array([method_2(upper, lower, p) for p in points])

    presicion_p1 = np.mean(np.abs(points - p1))
    presicion_p2 = np.mean(np.abs(points - p2))

    return (presicion_p1, presicion_p2)


class BoundaryHandler(Algorithm):
    
    """Boundary Methods Default"""

    @staticmethod
    # Metodo boundary para manejar restriccion de limites de individuo
    def saturation(upper, lower, x) -> np.ndarray:
        new_x = np.copy(x)

        for i in range(len(x)):
            if x[i] < lower[i]:
                new_x[i] = lower[i]
            elif x[i] > upper[i]:
                new_x[i] = upper[i]

        return new_x

    @staticmethod
    # Metodo de reflex para manejar restricciones  de limites dentro de los valores de un individuo
    def mirror(upper, lower, x) -> np.ndarray:
        range_width = upper - lower
        new_x = np.copy(x)

        for j in range(len(x)):
            if new_x[j] < lower[j]:
                new_x[j] = lower[j] + (lower[j] - new_x[j]) % range_width[j]
            elif new_x[j] > upper[j]:
                new_x[j] = upper[j] - (new_x[j] - upper[j]) % range_width[j]

        return new_x

    @staticmethod
    # Metodo random para manejar restricciones de limites dentro de los valores de un individuo
    def random(upper, lower, x) -> np.ndarray:
        new_x = []

        for sup, ind, inf in zip(upper, x, lower):
            if ind > sup or ind < inf:
                ind = inf + np.random.uniform(0, 1) * (sup - inf)
                new_x.append(ind)
            else:
                new_x.append(ind)

        return np.array(new_x)

    # Metodo wrapping para maenjar restricciones de limites dentro de los valores de un individuo
    def wrapping(upper, lower, x) -> np.ndarray:
        range_width = abs(upper - lower)
        new_x = np.copy(x)

        for i in range(len(x)):
            if new_x[i] < lower[i]:
                new_x[i] = upper[i] - (lower[i] - new_x[i]) % range_width[i]
            elif new_x[i] > upper[i]:
                new_x[i] = lower[i] + (new_x[i] - upper[i]) % range_width[i]

        return new_x

    """
    1999 Lampiden Mixed DE
    """

    @staticmethod
    def lampiden_replace_out_of_bounds(upper, lower, x):
        if np.any(x < lower) or np.any(x > upper):
            x = np.random.uniform(lower, upper)
        return x

    @staticmethod
    def lampiden_repeat_unitl_within_bounds(upper, lower, x):
        while np.any(x < lower) or np.any(x > upper):
            x = np.random.uniform(lower, upper)
        return x

    """
    2002 Lampiden Constraint
    """

    @staticmethod
    def lampiden_handle_boundary_constraint(upper, lower, x):
        if np.any(x < lower) or np.any(x > upper):
            return np.random.uniform(lower, upper)
        else:
            return x

    """
    2004 Zhang Handling
    """

    @staticmethod
    def zhang_periodic_mode(upper, lower, x):
        s = upper - lower
        z = np.copy(x)

        for d in range(len(x)):
            if x[d] < lower[d]:
                z[d] = upper[d] - (x[d] % s[d])
            elif x[d] > upper[d]:
                z[d] = lower[d] + (x[d] % s[d])
            else:
                z[d] = x[d]

        return z

    """
    2004 – Robinson Particle
    """

    @staticmethod
    def robinson_invisible_walls(x, objective_function, bounds):
        for i in range(len(x)):
            if x[i] < bounds[i][0] or x[i] > bounds[i][1]:
                return np.inf
        return objective_function(x)

    """
    2004 Purchia Experimentation
    """

    @staticmethod
    def purchia_proyection(upper, lower, x):
        return np.maximum(lower, np.minimum(x, upper))

    @staticmethod
    def purchia_wrapping(upper, lower, x):
        return lower + np.mod(x - lower, upper - lower)

    """ 2004 Mendes Population """

    """ 2005 Huang Hybrid """

    """ 2006 ClercConfinements """

    """
    2007 Adham Particle
    """

    @staticmethod
    def adham_clamp_position(upper, lower, x):
        return np.maximum(lower, np.minimum(x, upper))

    @staticmethod
    def adham_random_reinitialize(upper, lower, x):
        reinitialized_position = np.where(
            (x < lower) | (x > upper),
            np.random.uniform(lower, upper),
            x,
        )
        return reinitialized_position

    @staticmethod
    def adham_reflect_position(upper_bound, lower_bound, position):
        reflected_position = position.copy()
        below_lower = position < lower_bound
        above_upper = position > upper_bound

        reflected_position[below_lower] = lower_bound[below_lower] + (
            lower_bound[below_lower] - position[below_lower]
        )
        reflected_position[above_upper] = upper_bound[above_upper] - (
            position[above_upper] - upper_bound[above_upper]
        )

        return reflected_position

    @staticmethod
    def adham_shrink_position(upper_bound, lower_bound, position):
        shrunk_position = position.copy()
        shrunk_position[position < lower_bound] = lower_bound[position < lower_bound]
        shrunk_position[position > upper_bound] = upper_bound[position > upper_bound]

        return shrunk_position

    @staticmethod
    def adham_eliminate_particles(upper_bound, lower_bound, positions):
        within_bounds = (positions >= lower_bound) & (positions <= upper_bound)
        out_of_bounds = ~np.all(within_bounds, axis=1)

        new_positions = positions.copy()
        new_positions[out_of_bounds] = lower_bound + np.random.rand(
            np.sum(out_of_bounds), positions.shape[1]
        ) * (upper_bound - lower_bound)

        return new_positions

    """ 2009 Qin Algorithm"""

    @staticmethod
    def qin_random_reinitialization(upper, lower, x):
        for i in range(len(x)):
            if x[i] < lower[i] or x[i] > upper[i]:
                x[i] = np.random.uniform(lower[i], upper[i])
        return x

    @staticmethod
    def qin_boundary_repair(upper, lower, x):
        for i in range(len(x)):
            if x[i] < lower[i]:
                x[i] = lower[i]
            elif x[i] > upper[i]:
                x[i] = upper[i]
        return x

    @staticmethod
    def qin_reflection(upper, lower, x):
        for i in range(len(x)):
            if x[i] < lower[i]:
                x[i] = lower[i] + (lower[i] - x[i])
            elif x[i] > upper[i]:
                x[i] = upper[i] - (x[i] - upper[i])
        return x

    @staticmethod
    def qin_wrapping(upper, lower, x):
        for i in range(len(x)):
            if x[i] < lower[i]:
                x[i] = upper[i] - (lower[i] - x[i]) % (upper[i] - lower[i])
            elif x[i] > upper[i]:
                x[i] = lower[i] + (x[i] - upper[i]) % (upper[i] - lower[i])
        return x

    """ 2010 Methods """

    """ 2011 Chu Handling"""

    def chu_handle_bounds_random(upper, lower, x):
        for i in range(len(x)):
            if x[i] < lower[i] or x[i] > upper[i]:
                x[i] = np.random.uniform(lower[i], upper[i])
        return x

    def chu_handle_bounds_absorbing(upper, lower, x):
        for i in range(len(x)):
            if x[i] < lower[i]:
                x[i] = lower[i]
            elif x[i] > upper[i]:
                x[i] = upper[i]
        return x

    def chu_handle_bounds_reflecting(upper, lower, x):
        for i in range(len(x)):
            if x[i] < lower[i]:
                x[i] = 2 * lower[i] - x[i]
            elif x[i] > upper[i]:
                x[i] = 2 * upper[i] - x[i]
        return x

    """ 2011 Omeltschuk heterogeneous """

    @staticmethod
    def phi_static(x, f, constraints, K=1e6):
        # Evaluar si x cumple con todas las restricciones
        satisfied_constraints = [g(x) <= 0 for g in constraints["inequality"]] + [
            abs(h(x)) <= 1e-6 for h in constraints["equality"]
        ]
        s = sum(satisfied_constraints)
        l = len(constraints["inequality"])
        m = len(constraints["equality"])

        # Calcular la función penalizada
        if s == l + m:
            return f(x)
        else:
            return K * (1 - s / (l + m))

    """ 2012 Shi Experimental """

    @staticmethod
    def shi_classical_boundary_handling(upper, lower, x):
        new_x = np.where(x > upper, upper, x)
        new_x = np.where(new_x < lower, lower, new_x)
        return new_x

    @staticmethod
    def shi_deterministic_boundary_handling(x, prev_x, lower, upper):
        new_x = np.copy(x)
        new_x = np.where(x > upper, (prev_x + upper) / 2, new_x)
        new_x = np.where(x < lower, (prev_x + lower) / 2, new_x)
        return new_x

    @staticmethod
    def shi_stochastic_boundary_handling(upper, lower, particles):
        num_particles, num_dimensions = particles.shape
        for i in range(num_particles):
            for j in range(num_dimensions):
                if particles[i, j] < lower[j] or particles[i, j] > upper[j]:
                    particles[i, j] = np.random.uniform(lower[j], upper[j])
        return particles

    """ 2012 Gandomi Evolutionary """

    @staticmethod
    def evolutionary_boundary_handling(z, lb, ub, best_solution):
        a = np.random.rand()
        b = np.random.rand()

        x = np.copy(z)

        x[z < lb] = a * lb[z < lb] + (1 - a) * best_solution[z < lb]
        x[z > ub] = b * ub[z > ub] + (1 - b) * best_solution[z > ub]

        return x

    """ 2012 Han Particle """

    def escape_boundary(positions, velocities, solution_space, epsilon):
        swarm_size, dim = positions.shape
        for i in range(swarm_size):
            for d in range(dim):
                if (
                    positions[i, d] < solution_space[0]
                    or positions[i, d] > solution_space[1]
                ):
                    positions[i, d] = np.clip(
                        positions[i, d], solution_space[0], solution_space[1]
                    )
                    velocities[i, d] *= -np.random.rand()
                elif (
                    solution_space[0] + epsilon < positions[i, d] < solution_space[0]
                    or solution_space[1] < positions[i, d] < solution_space[1] - epsilon
                ):
                    velocities[i, d] *= np.random.rand()
        return positions, velocities

    """ 2013 Hewling Experimental"""

    @staticmethod
    def bounded_mirror(position, velocity, bounds):
        lower_bound, upper_bound = bounds
        new_position = position + velocity

        for i in range(len(new_position)):
            if new_position[i] < lower_bound[i]:
                new_position[i] = lower_bound[i] + (lower_bound[i] - new_position[i])
            elif new_position[i] > upper_bound[i]:
                new_position[i] = upper_bound[i] - (new_position[i] - upper_bound[i])

            # Aplicando el espejo acotado
            if new_position[i] < lower_bound[i] or new_position[i] > upper_bound[i]:
                new_position[i] = lower_bound[i] + (
                    new_position[i] % (2 * upper_bound[i] - lower_bound[i])
                )

        return new_position

    """ 2013 Padhye Boundary"""

    @staticmethod
    def inverse_parabolic_distribution(x_not, x_c, alpha=1.2):
        d = np.linalg.norm(x_c - x_not)
        a = (alpha**2 * d**2) / (np.pi / 2 + np.arctan(1 / alpha))

        def probability_distribution(x):
            return a / ((x - d) ** 2 + alpha**2 * d**2)

        x1 = np.minimum(x_not, x_c)  # Assuming x_not and x_c are vectors
        x2 = np.maximum(x_not, x_c)

        r = np.random.rand()
        x_prime = d + alpha * d * np.tan(r * np.arctan((x2 - d) / (alpha * d)))

        return x_prime

    """ 2013 Wessing Repair"""

    @staticmethod
    def wessing_projection_repair(x, lower, upper):
        return max(lower, min(x, upper))

    @staticmethod
    def reflection_repair(x, lower, upper):
        if x < lower:
            return BoundaryHandler.reflection_repair(lower + (lower - x), lower, upper)
        elif x > upper:
            return BoundaryHandler.reflection_repair(upper + (upper - x), lower, upper)
        return x

    @staticmethod
    def wessing_wrapping_reapir(upper, lower, x):
        range_ = upper - lower
        if x < lower:
            return BoundaryHandler.wrapping_reapir(x + range_, lower, upper)
        elif x > upper:
            return BoundaryHandler.wrapping_reapir(x - range_, lower, upper)
        return x

    @staticmethod
    def intersection_projection(a, b, lower, upper):
        c = lower if b < lower else upper
        s = (c - b) / (a - b)
        return b + s * (a - b)

    @staticmethod
    def intersection_reflection(a, b, lower, upper):
        c = lower if b < lower else upper
        s = (c - b) / (a - b)
        new_x = b + s * (a - b)
        return BoundaryHandler.reflection_repair(new_x, lower, upper)

    """ 2015 Gandomi Boundary"""

    @staticmethod
    def levy_flight(Lambda):
        sigma = (
            np.math.gamma(1 + Lambda)
            * np.sin(np.pi * Lambda / 2)
            / (np.math.gamma((1 + Lambda) / 2) * Lambda * 2 ** ((Lambda - 1) / 2))
        ) ** (1 / Lambda)
        u = np.random.normal(0, sigma, size=1)
        v = np.random.normal(0, 1, size=1)
        step = u / np.abs(v) ** (1 / Lambda)
        return step

    @staticmethod
    def evolutionary_boundary_handling(z, lb, ub, x_best):
        alpha = np.random.rand()
        beta = np.random.rand()
        if z < lb:
            return alpha * lb + (1 - alpha) * x_best
        elif z > ub:
            return beta * ub + (1 - beta) * x_best
        else:
            return z

    @staticmethod
    def cuckoo_search(n, dim, lb, ub, obj_func, max_iter):
        nests = np.random.rand(n, dim) * (ub - lb) + lb
        fitness = np.apply_along_axis(obj_func, 1, nests)
        best_nest = nests[np.argmin(fitness)]

        for _ in range(max_iter):
            new_nests = nests + BoundaryHandler.levy_flight(1.5) * (nests - best_nest)

            for i in range(n):
                for j in range(dim):
                    new_nests[i, j] = BoundaryHandler.evolutionary_boundary_handling(
                        new_nests[i, j], lb[j], ub[j], best_nest[j]
                    )

            new_fitness = np.apply_along_axis(obj_func, 1, new_nests)
            for i in range(n):
                if new_fitness[i] < fitness[i]:
                    nests[i] = new_nests[i]
                    fitness[i] = new_fitness[i]

            best_nest = nests[np.argmin(fitness)]

        return best_nest, np.min(fitness)

    """ 2015 Juarez Novel """

    @staticmethod
    def calculate_centroid_2015(
        population, feasible_set, infeasible_set, mutant_vector, bounds
    ):
        def random_within_bounds(bounds):
            return np.array([np.random.uniform(low, high) for low, high in bounds])

        AFS = len(feasible_set)
        if AFS > 0 and np.random.rand() > 0.5:
            wc1 = feasible_set[np.random.randint(AFS)]
        else:
            wc1 = min(
                infeasible_set,
                key=lambda x: sum(
                    np.maximum(0, np.array(bounds)[:, 0] - x)
                    + np.maximum(0, x - np.array(bounds)[:, 1])
                ),
            )

        wc2 = random_within_bounds(bounds)
        wc3 = random_within_bounds(bounds)

        centroid = (wc1 + wc2 + wc3) / 3

        for i in range(len(mutant_vector)):
            if not (bounds[i][0] <= mutant_vector[i] <= bounds[i][1]):
                mutant_vector[i] = centroid[i]

        return mutant_vector

    """ 2015 Padhye Feasibility"""

    @staticmethod
    def inverse_parabolic_method(position, parent_position, bounds, alpha=1.2):
        def get_distance(v, p):
            return np.linalg.norm(v - p)

        def calculate_new_position(xc, xp, dv, alpha):
            r = np.random.rand()
            d_prime = dv + alpha * dv * np.tan(
                r * np.arctan((bounds[1] - dv) / (alpha * dv))
            )
            return xc + d_prime * (xp - xc)

        new_positions = []
        for i in range(position.shape[0]):
            xc = position[i]
            xp = parent_position[i]
            dv = get_distance(xc, bounds[0])
            if dv > 0:
                new_pos = calculate_new_position(xc, xp, dv, alpha)
                new_positions.append(new_pos)
            else:
                new_positions.append(xc)

        return np.array(new_positions)

    """ 2015 Zhup Spacecraft """

    def boundary_handling(u, x_min, x_max):
        if u < x_min:
            return x_min + np.random.rand() * (x_max - x_min)
        elif u > x_max:
            return x_max - np.random.rand() * (x_max - u)
        return u

    """ 2016 Abdallah Solving """

    def repair_OEI(individual, best_individual, lower_bound, upper_bound):
        if lower_bound <= best_individual <= upper_bound:
            return best_individual
        elif best_individual < lower_bound:
            return lower_bound
        else:
            return upper_bound

    def repair_scaling(
        infeasible_value,
        lower_bound_old,
        upper_bound_old,
        lower_bound_new,
        upper_bound_new,
    ):
        S_old = upper_bound_old - lower_bound_old
        S_new = upper_bound_new - lower_bound_new
        return upper_bound_new - ((upper_bound_old - infeasible_value) * S_old / S_new)

    def repair_stochastic_new_boundaries(
        infeasible_value, lower_bound_new, upper_bound_new
    ):
        if infeasible_value > upper_bound_new:
            return upper_bound_new - 0.50 * np.random.random() * (
                upper_bound_new - lower_bound_new
            )
        elif infeasible_value < lower_bound_new:
            return lower_bound_new + 0.50 * np.random.random() * (
                upper_bound_new - lower_bound_new
            )

    """ 2016 Agarwl Experimental """

    def agarwl_reflect(upper_bound, lower_bound, position):
        for i in range(len(position)):
            if position[i] < lower_bound[i]:
                position[i] = lower_bound[i] + (lower_bound[i] - position[i])
            elif position[i] > upper_bound[i]:
                position[i] = upper_bound[i] - (position[i] - upper_bound[i])
        return position

    def agarwl_nearest(upper_bound, lower_bound, position):
        for i in range(len(position)):
            if position[i] < lower_bound[i]:
                position[i] = lower_bound[i]
            elif position[i] > upper_bound[i]:
                position[i] = upper_bound[i]
        return position

    def hyperbolic_2016(position, velocity, lower_bound, upper_bound):
        for i in range(len(position)):
            if velocity[i] > 0:
                velocity[i] = velocity[i] / (
                    1 + velocity[i] / (upper_bound[i] - position[i])
                )
            elif velocity[i] < 0:
                velocity[i] = velocity[i] / (
                    1 + velocity[i] / (position[i] - lower_bound[i])
                )
        return velocity

    """ 2016 Gandomi Evolutionary """
    
    """ 2018 kadavy and gandomi """
    def periodic_method(x, lower_bound, upper_bound):
        if x > upper_bound or x < lower_bound:
            return lower_bound + (x % (upper_bound - lower_bound))
        else:
            return x

 

    def probabilistic_method(x, lower_bound, upper_bound, probability=0.5):
        if x > upper_bound or x < lower_bound:
            if random.random() < probability:
                return random.uniform(lower_bound, upper_bound)
            else:
                return x
        else:
            return x


    def juarez_W_p(SFS, SIS, AFS):

        if AFS > 0 and np.random.random() > 0.5:
            return np.random.choice(SFS)
        else:
            return min(SIS, key=lambda x: np.linalg.norm(x))

    #example:
    #max resamples
    #3 * len(lower_bounds)
    def juarez_res_ran_DE_rand_1_bin(target_vector_index, population, F, lower_bounds, upper_bounds, is_valid, max_resamples):
        D = len(lower_bounds)
        NP = len(population)
        resamples = 0
        valid = False

        while not valid and resamples < 3 * D:
            r1, r2, r3 = np.random.choice(NP, 3, replace=False)
            if len(set([target_vector_index, r1, r2, r3])) != 4:
                continue
            
            mutant_vector = population[r1] + F * (population[r2] - population[r3])
            mutant_vector = np.clip(mutant_vector, lower_bounds, upper_bounds)
            resamples += 1
            is_valid(upper_bounds, lower_bounds, mutant_vector)

        if not valid:
            mutant_vector = lower_bounds + np.random.rand(D) * (upper_bounds - lower_bounds)

        return mutant_vector
    
           
    """ 2020 biedrzycki  """
    def reinitialization(m, l, u):
        return np.where((m >= l) & (m <= u), m, np.random.uniform(l, u))

    def projection(m, l, u):
        return np.where(m < l, l, np.where(m > u, u, m))

    def reflection(m, l, u):
        return np.where(m < l, 2*l - m, np.where(m > u, 2*u - m, m))

    def wrapping(m, l, u):
        return np.where(m < l, u + (m - l), np.where(m > u, l + (m - u), m))

    def transformation(m, l, u):
        a_lj = np.minimum((u - l) / 2, (1 + np.abs(l)) / 20)
        a_uj = np.minimum((u - l) / 2, (1 + np.abs(u)) / 20)
        
        mask_l = (l - a_lj <= m) & (m < l + a_lj)
        mask_u = (u - a_uj <= m) & (m < u + a_uj)
        
        m_transformed = np.where(mask_l, l + ((m - (l - a_lj))**2 / (4 * a_lj)), m)
        m_transformed = np.where(mask_u, u - ((m - (u + a_uj))**2 / (4 * a_uj)), m_transformed)
        
        return np.where((m >= l + a_lj) & (m <= u - a_uj), m, m_transformed)

    def projection_to_midpoint(m, l, u):
        midpoint = (l + u) / 2
        alpha = np.minimum((m - midpoint) / (m - l), (m - midpoint) / (u - m))
        return (1 - alpha) * midpoint + alpha * m

    def rand_base(m, l, u, o):
        return np.where(m < l, np.random.uniform(l, o), np.where(m > u, np.random.uniform(o, u), m))

    def midpoint_base(m, l, u, o):
        return np.where(m < l, (l + o) / 2, np.where(m > u, (o + u) / 2, m))

    def conservative(m, l, u, o):
        return np.where((m >= l) & (m <= u), m, o)

    def projection_to_base(m, l, u, o):
        alpha = np.minimum((m - o) / (m - l), (m - o) / (u - m))
        return (1 - alpha) * o + alpha * m
        
    
    """ 2023  """
    
    def andreaa_saturation(b, a, y):
        return np.clip(y, a, b)

    def andreaa_mirror(b, a, y):
        y_mirrored = np.where(y < a, 2 * a - y, y)
        y_mirrored = np.where(y_mirrored > b, 2 * b - y_mirrored, y_mirrored)
        return y_mirrored


    def saturation(particle, lower_bound, upper_bound):
        return np.clip(particle, lower_bound, upper_bound)

    def mirror(particle, lower_bound, upper_bound):
        particle_mirrored = np.where(particle < lower_bound, 2 * lower_bound - particle, particle)
        particle_mirrored = np.where(particle_mirrored > upper_bound, 2 * upper_bound - particle_mirrored, particle_mirrored)
        return particle_mirrored

    def vector_wise_correction(particle, lower_bound, upper_bound, R):
        alpha = np.min([
            np.where(particle < lower_bound, (R - lower_bound) / (R - particle), 1),
            np.where(particle > upper_bound, (upper_bound - R) / (particle - R), 1)
        ])
        return alpha * particle + (1 - alpha) * R

    def andreaa_uniform(b, a, y):
        return np.random.uniform(a, b, size=y.shape)
    
    def uniform(particle, lower_bound, upper_bound):
        return np.random.uniform(lower_bound, upper_bound, size=particle.shape)

    def beta(particle, lower_bound, upper_bound, mean, var):
        m = (mean - lower_bound) / (upper_bound - lower_bound)
        v = var / (upper_bound - lower_bound) ** 2
        alpha = m * ((m * (1 - m) / v) - 1)
        beta_param = alpha * ((1 - m) / m)
        return lower_bound + np.random.beta(alpha, beta_param, size=particle.shape) * (upper_bound - lower_bound)

    def exp_confined(particle, lower_bound, upper_bound, R):
        r = np.random.uniform(0, 1, size=particle.shape)
        particle_exp = np.where(
            particle < lower_bound, lower_bound - np.log(1 + r * (np.exp(lower_bound - R) - 1)),
            upper_bound + np.log(1 + (1 - r) * (np.exp(R - upper_bound) - 1))
        )
        return particle_exp

    def absorbing(y, a, b, v):
        y_absorbed = np.where(y < a, a, y)
        y_absorbed = np.where(y_absorbed > b, b, y_absorbed)
        v_absorbed = np.where((y < a) | (y > b), 0, v)
        return y_absorbed, v_absorbed

    @staticmethod
    def reflecting(y, a, b, v):
        y_reflected = np.where(y < a, a, y)
        y_reflected = np.where(y_reflected > b, b, y_reflected)
        v_reflected = np.where(y < a, -v, v)
        v_reflected = np.where(y_reflected > b, -v_reflected, v_reflected)
        return y_reflected, v_reflected

    @staticmethod
    def damping(y, a, b, v):
        y_damped = np.where(y < a, a, y)
        y_damped = np.where(y_damped > b, b, y_damped)
        v_damped = np.where(y < a, -v * np.random.uniform(0, 1), v)
        v_damped = np.where(y_damped > b, -v_damped * np.random.uniform(0, 1), v_damped)
        return y_damped, v_damped
    def absorbing(particle, lower_bound, upper_bound, velocity):
        particle_absorbed = np.where(particle < lower_bound, lower_bound, particle)
        particle_absorbed = np.where(particle_absorbed > upper_bound, upper_bound, particle_absorbed)
        velocity_absorbed = np.where((particle < lower_bound) | (particle > upper_bound), 0, velocity)
        return particle_absorbed, velocity_absorbed

    def reflecting(particle, lower_bound, upper_bound, velocity):
        particle_reflected = np.where(particle < lower_bound, lower_bound, particle)
        particle_reflected = np.where(particle_reflected > upper_bound, upper_bound, particle_reflected)
        velocity_reflected = np.where(particle < lower_bound, -velocity, velocity)
        velocity_reflected = np.where(particle_reflected > upper_bound, -velocity_reflected, velocity_reflected)
        return particle_reflected, velocity_reflected

    def damping(particle, lower_bound, upper_bound, velocity):
        particle_damped = np.where(particle < lower_bound, lower_bound, particle)
        particle_damped = np.where(particle_damped > upper_bound, upper_bound, particle_damped)
        velocity_damped = np.where(particle < lower_bound, -velocity * np.random.uniform(0, 1), velocity)
        velocity_damped = np.where(particle_damped > upper_bound, -velocity_damped * np.random.uniform(0, 1), velocity_damped)
        return particle_damped, velocity_damped

    def invisible(particle, lower_bound, upper_bound, fitness, bad_fitness_value):
        fitness_invisible = np.where((particle < lower_bound) | (particle > upper_bound), bad_fitness_value, fitness)
        return particle, fitness_invisible

    def invisible_reflecting(y, a, b, v, fitness, bad_fitness_value):
        y_invisible_reflecting, v_reflected = BoundaryHandler.reflecting(y, a, b, v)
        fitness_invisible = np.where((y < a) | (y > b), bad_fitness_value, fitness)
        return y_invisible_reflecting, v_reflected, fitness_invisible

    def invisible_damping(y, a, b, v, fitness, bad_fitness_value):
        y_invisible_damping, v_damped = BoundaryHandler.damping(y, a, b, v)
        fitness_invisible = np.where((y < a) | (y > b), bad_fitness_value, fitness)
        return y_invisible_damping, v_damped, fitness_invisible

    def andreaa_inf(b, a, y):
        y_inf = np.where(y < a, -np.inf, y)
        y_inf = np.where(y_inf > b, np.inf, y_inf)
        return y_inf

    @staticmethod
    def andreaa_nearest(b, a, y):
        y_nearest = np.where(y < a, a, y)
        y_nearest = np.where(y_nearest > b, b, y_nearest)
        return y_nearest

    def andreaa_nearest_turb(b, a, y):
        y_nearest_turb = BoundaryHandler.nearest(y, a, b)
        turbulence = np.random.normal(0, 1, size=y.shape)
        y_nearest_turb += turbulence
        return BoundaryHandler.adham_reflect_position(y_nearest_turb, a, b)

    def andreaa_random_within_bounds(b, a, y):
        y_random = np.random.uniform(a, b, size=y.shape)
        return y_random

    def andreaa_shr(y, a, b, v):
        factor = np.where(y < a, (a - y) / v, 1)
        factor = np.where(y > b, (b - y) / v, factor)
        y_shr = y + v * np.min(factor)
        return y_shr
