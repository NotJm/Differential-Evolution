import numpy as np
import random
from typing import Callable
from .algorithm import Algorithm
# from sklearn.cluster import KMeans

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


class BoundaryHandler:
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
    def reflection(upper, lower, x) -> np.ndarray:
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
    def robinson_invisible_walls(
        x: np.ndarray,
        objective_function: Callable[[np.ndarray], float],
        upper: np.ndarray,
        lower: np.ndarray,
    ) -> float:
        if np.any(x < lower) or np.any(x > upper):
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
    def adham_reflect_position(upper, lower, x):
        reflected_position = x.copy()
        below_lower = x < lower
        above_upper = x > upper

        reflected_position[below_lower] = lower[below_lower] + (
            lower[below_lower] - x[below_lower]
        )
        reflected_position[above_upper] = upper[above_upper] - (
            x[above_upper] - upper[above_upper]
        )

        return reflected_position

    @staticmethod
    def adham_shrink_position(upper, lower, x):
        shrunk_position = x.copy()
        shrunk_position[x < lower] = lower[x < lower]
        shrunk_position[x > upper] = upper[x > upper]

        return shrunk_position

    @staticmethod
    def adham_eliminate_particles(upper, lower, positions):
        within_bounds = (positions >= lower) & (positions <= upper)
        out_of_bounds = ~np.all(within_bounds, axis=1)

        new_positions = positions.copy()
        new_positions[out_of_bounds] = lower + np.random.rand(
            np.sum(out_of_bounds), positions.shape[1]
        ) * (upper - lower)

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


    """ 2012 Shi Experimental """

    @staticmethod
    def shi_classical_boundary_handling(upper, lower, x):
        new_x = np.where(x > upper, upper, x)
        new_x = np.where(new_x < lower, lower, new_x)
        return new_x

    @staticmethod
    def shi_deterministic_boundary_handling(upper, lower, x, prev_x):
        new_x = np.copy(x)
        new_x = np.where(x > upper, (prev_x + upper) / 2, new_x)
        new_x = np.where(x < lower, (prev_x + lower) / 2, new_x)
        return new_x

    @staticmethod
    def shi_stochastic_boundary_handling(upper, lower, population):
        num_particles, num_dimensions = population.shape
        for i in range(num_particles):
            for j in range(num_dimensions):
                if population[i, j] < lower[j] or population[i, j] > upper[j]:
                    population[i, j] = np.random.uniform(lower[j], upper[j])
        return population

    """ 2012 Gandomi Evolutionary """

    @staticmethod
    def evolutionary(z, lb, ub, best_solution):
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
        lower, upper = bounds
        new_position = position + velocity

        for i in range(len(new_position)):
            if new_position[i] < lower[i]:
                new_position[i] = lower[i] + (lower[i] - new_position[i])
            elif new_position[i] > upper[i]:
                new_position[i] = upper[i] - (new_position[i] - upper[i])

            # Aplicando el espejo acotado
            if new_position[i] < lower[i] or new_position[i] > upper[i]:
                new_position[i] = lower[i] + (
                    new_position[i] % (2 * upper[i] - lower[i])
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

    def repair_OEI(individual, best_individual, lower, upper):
        if lower <= best_individual <= upper:
            return best_individual
        elif best_individual < lower:
            return lower
        else:
            return upper

    def repair_scaling(
        infeasible_value,
        lower_old,
        upper_old,
        lower_new,
        upper_new,
    ):
        S_old = upper_old - lower_old
        S_new = upper_new - lower_new
        return upper_new - ((upper_old - infeasible_value) * S_old / S_new)

    def repair_stochastic_new_boundaries(
        infeasible_value, lower_new, upper_new
    ):
        if infeasible_value > upper_new:
            return upper_new - 0.50 * np.random.random() * (
                upper_new - lower_new
            )
        elif infeasible_value < lower_new:
            return lower_new + 0.50 * np.random.random() * (
                upper_new - lower_new
            )

    """ 2016 Agarwl Experimental """

    def agarwl_reflect(upper, lower, position):
        for i in range(len(position)):
            if position[i] < lower[i]:
                position[i] = lower[i] + (lower[i] - position[i])
            elif position[i] > upper[i]:
                position[i] = upper[i] - (position[i] - upper[i])
        return position

    def agarwl_nearest(upper, lower, position):
        for i in range(len(position)):
            if position[i] < lower[i]:
                position[i] = lower[i]
            elif position[i] > upper[i]:
                position[i] = upper[i]
        return position

    def hyperbolic_2016(position, velocity, lower, upper):
        for i in range(len(position)):
            if velocity[i] > 0:
                velocity[i] = velocity[i] / (
                    1 + velocity[i] / (upper[i] - position[i])
                )
            elif velocity[i] < 0:
                velocity[i] = velocity[i] / (
                    1 + velocity[i] / (position[i] - lower[i])
                )
        return velocity

    """ 2016 Gandomi Evolutionary """

    """ 2018 kadavy and gandomi """

    def periodic_method(x, lower, upper):
        if x > upper or x < lower:
            return lower + (x % (upper - lower))
        else:
            return x

    def probabilistic_method(x, lower, upper, probability=0.5):
        if x > upper or x < lower:
            if random.random() < probability:
                return random.uniform(lower, upper)
            else:
                return x
        else:
            return x
        
    

    # """ 2019 Efren Juarez Centroid """
    # def centroid_method(X, population, lower, upper, K=3):
    #     """
    #     Implementación del método Centroid para manejo de límites usando numpy.

    #     Parámetros:
    #     X (numpy.ndarray): El vector que se encuentra fuera de los límites.
    #     population (numpy.ndarray): Población actual de soluciones.
    #     lower (numpy.ndarray): Límites inferiores para cada dimensión.
    #     upper (numpy.ndarray): Límites superiores para cada dimensión.
    #     K (int): Cantidad de vectores aleatorios. Default es 3.

    #     Retorna:
    #     numpy.ndarray: Vector corregido.
    #     """
    #     NP, D = population.shape
        
    #     # Soluciones Factibles (SFS) y Soluciones No Factibles (SIS)
    #     SFS = np.array([ind for ind in population if np.all(lower <= ind) and np.all(ind <= upper)])
    #     SIS = np.array([ind for ind in population if ind not in SFS])
    #     AFS = len(SFS)

    #     if AFS > 0 and np.random.rand() > 0.5:
    #         Wp = SFS[np.random.randint(AFS)]
    #     else:
    #         if len(SIS) > 0:
    #             Wp = SIS[np.argmin([np.sum(np.maximum(0, lower - ind) + np.maximum(0, ind - upper)) for ind in SIS])]
    #         else:
    #             # Si SIS está vacío, seleccionamos aleatoriamente un vector de la población
    #             Wp = population[np.random.randint(NP)]

    #     Wr = np.empty((K, D))
    #     for i in range(K):
    #         Wi = np.copy(X)
    #         for j in range(D):
    #             if Wi[j] < lower[j] or Wi[j] > upper[j]:
    #                 Wi[j] = lower[j] + np.random.rand() * (upper[j] - lower[j])
    #         Wr[i] = Wi
        
    #     Xc = (Wp + Wr.sum(axis=0)) / (K + 1)

    #     return Xc

    
    

    def centroid(target_vector_index, population, lower, upper,  K=1):
        D = len(lower)
        NP = len(population)
        def is_valid(lower, upper, vector):
            return np.all(vector >= lower) and np.all(vector <= upper)
        
        # Seleccionar el vector de la población
        if is_valid(population[target_vector_index]):
            selected_vector = population[target_vector_index]
        else:
            selected_vector = min(population, key=lambda x: np.sum(np.maximum(0, np.maximum(lower - x, x - upper))))
        
        # Crear vectores aleatorios dentro de los límites
        random_vectors = np.array([lower + np.random.rand(D) * (upper - lower) for _ in range(K)])
        
        # Calcular el centroide
        centroid_vector = np.mean(np.vstack([selected_vector, random_vectors]), axis=0)
        
        # Aplicar los límites
        centroid_vector = np.clip(centroid_vector, lower, upper)
        
        return centroid_vector

   

    #example:
    #max resamples
    #3 * len(lower_bounds)
    def juarez_res_ran_DE_rand_1_bin(target_vector_index, population, F, lower, upper, is_valid, max_resamples=1):
        D = len(lower)
        NP = len(population)
        resamples = 0
        valid = False

        while not valid and resamples < 3 * D:
            r1, r2, r3 = np.random.choice(NP, 3, replace=False)
            if len(set([target_vector_index, r1, r2, r3])) != 4:
                continue

            mutant_vector = population[r1] + F * (population[r2] - population[r3])
            mutant_vector = np.clip(mutant_vector, lower, upper)
            resamples += 1
            is_valid(upper, lower, mutant_vector)

        if not valid:
            mutant_vector = lower + np.random.rand(D) * (
                upper - lower
            )

        return mutant_vector

    """ 2020 biedrzycki  """

    def reinitialization(m, l, u):
        return np.where((m >= l) & (m <= u), m, np.random.uniform(l, u))

    def projection(m, l, u):
        return np.where(m < l, l, np.where(m > u, u, m))

    def reflection(m, l, u):
        return np.where(m < l, 2 * l - m, np.where(m > u, 2 * u - m, m))

    def wrapping(m, l, u):
        return np.where(m < l, u + (m - l), np.where(m > u, l + (m - u), m))

    def transformation(m, l, u):
        a_lj = np.minimum((u - l) / 2, (1 + np.abs(l)) / 20)
        a_uj = np.minimum((u - l) / 2, (1 + np.abs(u)) / 20)

        mask_l = (l - a_lj <= m) & (m < l + a_lj)
        mask_u = (u - a_uj <= m) & (m < u + a_uj)

        m_transformed = np.where(mask_l, l + ((m - (l - a_lj)) ** 2 / (4 * a_lj)), m)
        m_transformed = np.where(
            mask_u, u - ((m - (u + a_uj)) ** 2 / (4 * a_uj)), m_transformed
        )

        return np.where((m >= l + a_lj) & (m <= u - a_uj), m, m_transformed)

    def projection_to_midpoint(m, l, u):
        midpoint = (l + u) / 2
        alpha = np.minimum((m - midpoint) / (m - l), (m - midpoint) / (u - m))
        return (1 - alpha) * midpoint + alpha * m

    def rand_base(m, l, u, o):
        return np.where(
            m < l, np.random.uniform(l, o), np.where(m > u, np.random.uniform(o, u), m)
        )

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

    def saturation(particle, lower, upper):
        return np.clip(particle, lower, upper)

    def reflection(particle, lower, upper):
        particle_mirrored = np.where(
            particle < lower, 2 * lower - particle, particle
        )
        particle_mirrored = np.where(
            particle_mirrored > upper,
            2 * upper - particle_mirrored,
            particle_mirrored,
        )
        return particle_mirrored

    def andreaa_vector_wise_correction(upper, lower, x, method="midpoint"):
        if method == "midpoint":
            R = (lower + upper) / 2
        else:
            raise ValueError(f"Método para calcular R no soportado: {method}")
        
        
        alpha = np.min(
            [
                np.where(x < lower, (R - lower) / (R - x), 1),
                np.where(x > upper, (upper - R) / (x - R), 1),
            ]
        )
        return alpha * x + (1 - alpha) * R

    def andreaa_uniform(b, a, y):
        return np.random.uniform(a, b, size=y.shape)
    def uniform(particle, lower_bound, upper_bound):
        return np.random.uniform(lower_bound, upper_bound, size=particle.shape)

    def andreaa_beta(individual, lower_bound, upper_bound, population):
        corrected_individual = np.copy(individual)
        population_mean = np.mean(population, axis=0)
        population_var = np.var(population, axis=0)
        
        for i in range(len(individual)):
            m_i = (population_mean[i] - lower_bound[i]) / (upper_bound[i] - lower_bound[i])
            v_i = population_var[i] / (upper_bound[i] - lower_bound[i])**2
            
            # Evitar valores de m_i en los límites
            if m_i == 0:
                m_i = 0.1
            elif m_i == 1:
                m_i = 0.9
            
            # Verificar que v_i no sea cero para evitar división por cero
            if v_i == 0:
                v_i = 1e-6
            
            # Calcular alpha_i y beta_i asegurando que sean positivos
            alpha_i = m_i * ((m_i * (1 - m_i) / v_i) - 1)
            beta_i = alpha_i * (1 - m_i) / m_i
            
            # Validar alpha_i y beta_i
            if alpha_i <= 0 or beta_i <= 0:
                alpha_i = beta_i = 1  # Asignar un valor por defecto válido
            
            # Generar valor corregido usando la distribución Beta
            corrected_value = np.random.beta(alpha_i, beta_i) * (upper_bound[i] - lower_bound[i]) + lower_bound[i]
            corrected_individual[i] = corrected_value
        
        return corrected_individual


    def exp_confined(particle, lower, upper, R):
        r = np.random.uniform(0, 1, size=particle.shape)
        particle_exp = np.where(
            particle < lower,
            lower - np.log(1 + r * (np.exp(lower - R) - 1)),
            upper + np.log(1 + (1 - r) * (np.exp(R - upper) - 1)),
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

    def absorbing(particle, lower, upper, velocity):
        particle_absorbed = np.where(particle < lower, lower, particle)
        particle_absorbed = np.where(
            particle_absorbed > upper, upper, particle_absorbed
        )
        velocity_absorbed = np.where(
            (particle < lower) | (particle > upper), 0, velocity
        )
        return particle_absorbed, velocity_absorbed

    def reflecting(particle, lower, upper, velocity):
        particle_reflected = np.where(particle < lower, lower, particle)
        particle_reflected = np.where(
            particle_reflected > upper, upper, particle_reflected
        )
        velocity_reflected = np.where(particle < lower, -velocity, velocity)
        velocity_reflected = np.where(
            particle_reflected > upper, -velocity_reflected, velocity_reflected
        )
        return particle_reflected, velocity_reflected

    def damping(particle, lower, upper, velocity):
        particle_damped = np.where(particle < lower, lower, particle)
        particle_damped = np.where(
            particle_damped > upper, upper, particle_damped
        )
        velocity_damped = np.where(
            particle < lower, -velocity * np.random.uniform(0, 1), velocity
        )
        velocity_damped = np.where(
            particle_damped > upper,
            -velocity_damped * np.random.uniform(0, 1),
            velocity_damped,
        )
        return particle_damped, velocity_damped

    def invisible(particle, lower, upper, fitness, bad_fitness_value):
        fitness_invisible = np.where(
            (particle < lower) | (particle > upper),
            bad_fitness_value,
            fitness,
        )
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

    @staticmethod
    def range_guardian(upper, lower, x, feasible_set):

        new_x = np.copy(x)

        for i in range(len(x)):
            # Reflexión
            if new_x[i] < lower[i]:
                new_x[i] = lower[i] + (lower[i] - new_x[i])
            elif new_x[i] > upper[i]:
                new_x[i] = upper[i] - (new_x[i] - upper[i])

            # Si aún está fuera de los límites, reubicar aleatoriamente
            if new_x[i] < lower[i] or new_x[i] > upper[i]:
                new_x[i] = np.random.uniform(lower[i], upper[i])

        # Ajuste basado en centroid para valores fuera de límites repetidamente
        if np.any(new_x < lower) or np.any(new_x > upper):
            centroid = np.mean(feasible_set, axis=0)
            for i in range(len(new_x)):
                if new_x[i] < lower[i] or new_x[i] > upper[i]:
                    new_x[i] = centroid[i]

        return new_x
    
    def hybrid_shrink_uniform_nearest(upper, lower, x):
        # Copiar el vector x
        new_x = np.copy(x)

        # Paso 1: Aplicar adham-shrink
        new_x[new_x < lower] = lower[new_x < lower]
        new_x[new_x > upper] = upper[new_x > upper]

        # Paso 2: Aplicar andreaa-uniform para valores fuera de límites
        out_of_bounds = (new_x < lower) | (new_x > upper)
        new_x[out_of_bounds] = np.random.uniform(lower[out_of_bounds], upper[out_of_bounds])

        # Paso 3: Aplicar agarwl-nearest para cualquier valor restante fuera de límites
        new_x[new_x < lower] = lower[new_x < lower]
        new_x[new_x > upper] = upper[new_x > upper]

        return new_x

    def probabilistic_adaptive_reflection(upper, lower, x):
        # Copiar el vector x
        new_x = np.copy(x)
        
        # Parámetro de ajuste adaptativo
        adaptation_rate = 0.5
        
        # Paso 1: Ajuste probabilístico
        for i in range(len(x)):
            if new_x[i] < lower[i] or new_x[i] > upper[i]:
                prob_adjustment = np.random.normal(loc=0, scale=1)  # Ajuste probabilístico
                new_x[i] += prob_adjustment

        # Paso 2: Ajuste adaptativo
        for i in range(len(x)):
            if new_x[i] < lower[i]:
                violation = lower[i] - new_x[i]
                new_x[i] += adaptation_rate * violation
            elif new_x[i] > upper[i]:
                violation = new_x[i] - upper[i]
                new_x[i] -= adaptation_rate * violation

        # Paso 3: Reflexión si sigue fuera de los límites
        for i in range(len(new_x)):
            if new_x[i] < lower[i]:
                new_x[i] = lower[i] + (lower[i] - new_x[i])
            elif new_x[i] > upper[i]:
                new_x[i] = upper[i] - (new_x[i] - upper[i])

        return new_x
    
    @staticmethod
    def hybrid_adaptive_centroid_boundary_handling(upper, lower, x, feasible_set):
        """
        Método de manejo de límites híbrido adaptativo con uso del centroide.

        :param upper: np.array, Limite superior
        :param lower: np.array, Limite inferior
        :param x: np.array, Vector de la solución actual
        :param feasible_set: np.array, Conjunto de soluciones factibles

        :return: np.array, Vector de la solución corregido
        """
        new_x = np.copy(x)
        range_width = upper - lower

        # Reflexión
        for i in range(len(x)):
            if new_x[i] < lower[i]:
                new_x[i] = lower[i] + (lower[i] - new_x[i]) % range_width[i]
            elif new_x[i] > upper[i]:
                new_x[i] = upper[i] - (new_x[i] - upper[i]) % range_width[i]

        # Ajuste adaptativo
        for i in range(len(x)):
            if new_x[i] < lower[i] or new_x[i] > upper[i]:
                correction_factor = np.random.uniform(0, 1)
                if new_x[i] < lower[i]:
                    new_x[i] = lower[i] + correction_factor * (upper[i] - lower[i])
                elif new_x[i] > upper[i]:
                    new_x[i] = upper[i] - correction_factor * (upper[i] - lower[i])

        # Uso del centroide
        if np.any(new_x < lower) or np.any(new_x > upper):
            centroid = np.mean(feasible_set, axis=0)
            for i in range(len(new_x)):
                if new_x[i] < lower[i] or new_x[i] > upper[i]:
                    new_x[i] = centroid[i]

        return new_x
    
    @staticmethod
    def haal_correction(x, population, lower, upper, feasible_set, K=3, history=None):
        """
        Método híbrido adaptativo con aprendizaje histórico para el manejo de límites.

        :param x: np.ndarray, Vector de la solución actual
        :param population: np.ndarray, Población actual de soluciones
        :param lower: np.ndarray, Límites inferiores para cada dimensión
        :param upper: np.ndarray, Límites superiores para cada dimensión
        :param feasible_set: np.ndarray, Conjunto de soluciones factibles
        :param K: int, Cantidad de vectores aleatorios para calcular el centroide. Default es 3.
        :param history: np.ndarray, Historia de soluciones anteriores

        :return: np.ndarray, Vector de la solución corregido
        """
        D = len(lower)
        NP = len(population)
        adaptation_factor = np.random.uniform(0.1, 0.5)

        # Función para validar si un vector está dentro de los límites
        def is_valid(vector):
            return np.all(vector >= lower) and np.all(vector <= upper)

        # Historial para aprendizaje adaptativo
        if history is None:
            history = []

        # Seleccionar el vector de la población basado en soluciones factibles
        SFS = population[np.all((population >= lower) & (population <= upper), axis=1)]
        SIS = population[np.any((population < lower) | (population > upper), axis=1)]
        AFS = len(SFS)

        if AFS > 0 and np.random.rand() > 0.5:
            Wp = SFS[np.random.randint(AFS)]
        else:
            if len(SIS) > 0:
                violations = np.sum(np.maximum(0, lower - SIS) + np.maximum(0, SIS - upper), axis=1)
                Wp = SIS[np.argmin(violations)]
            else:
                # Si SIS está vacío, seleccionamos aleatoriamente un vector de la población
                Wp = population[np.random.randint(NP)]

        # Crear vectores aleatorios dentro de los límites
        Wr = np.empty((K, D))
        for i in range(K):
            Wi = np.copy(x)
            mask_lower = Wi < lower
            mask_upper = Wi > upper
            Wi[mask_lower] = lower[mask_lower] + np.random.rand(np.sum(mask_lower)) * (upper[mask_lower] - lower[mask_lower])
            Wi[mask_upper] = lower[mask_upper] + np.random.rand(np.sum(mask_upper)) * (upper[mask_upper] - lower[mask_upper])
            Wr[i] = Wi

        # Calcular el centroide
        centroid = (Wp + Wr.sum(axis=0)) / (K + 1)

        # Aplicar corrección vectorial basada en el centroide
        for i in range(D):
            if x[i] < lower[i] or x[i] > upper[i]:
                x[i] = centroid[i]

        # Corrección vectorial adaptativa con dinamismo
        R = (lower + upper) / 2
        alpha = np.min(
            [
                np.where(x < lower, (R - lower) / (R - x), 1),
                np.where(x > upper, (upper - R) / (x - R), 1),
            ]
        )
        x = alpha * x + (1 - alpha) * R

        # Aplicar un mecanismo dinámico adicional para asegurar la corrección
        for i in range(D):
            if x[i] < lower[i]:
                x[i] += adaptation_factor * (lower[i] - x[i])
            elif x[i] > upper[i]:
                x[i] -= adaptation_factor * (x[i] - upper[i])

        # Aplicar reflexiones adaptativas
        for i in range(D):
            if x[i] < lower[i]:
                x[i] = lower[i] + (lower[i] - x[i]) % (upper[i] - lower[i])
            elif x[i] > upper[i]:
                x[i] = upper[i] - (x[i] - upper[i]) % (upper[i] - lower[i])

        # Utilizar la media de las soluciones factibles como ajuste final
        if np.any(x < lower) or np.any(x > upper):
            feasible_mean = np.mean(feasible_set, axis=0)
            for i in range(D):
                if x[i] < lower[i] or x[i] > upper[i]:
                    x[i] = feasible_mean[i]

        # Aprendizaje histórico
        history.append(x)
        if len(history) > 10:
            history.pop(0)
        
        historical_mean = np.mean(history, axis=0)
        for i in range(D):
            if x[i] < lower[i] or x[i] > upper[i]:
                x[i] = historical_mean[i]

        return x

    @staticmethod
    def dynamic_correction(x, population, lower, upper):
        """
        Método dinámico de corrección de límites que utiliza información de la población y ajuste adaptativo.

        :param x: np.ndarray, Vector de la solución actual
        :param population: np.ndarray, Población actual de soluciones
        :param lower: np.ndarray, Límites inferiores para cada dimensión
        :param upper: np.ndarray, Límites superiores para cada dimensión

        :return: np.ndarray, Vector de la solución corregido
        """
        mean_population = np.mean(population, axis=0)
        
        # Calcular la desviación normalizada y el valor de ajuste
        deviations = (x - mean_population) / np.where(np.abs(x - mean_population) == 0, 1, np.abs(x - mean_population))
        adjustments = deviations * np.minimum(np.abs(upper - mean_population), np.abs(lower - mean_population))
        
        # Aplicar corrección adaptativa
        x_corrected = mean_population + adjustments
        
        # Aplicar reflexión si sigue fuera de los límites
        x_corrected = np.where(x_corrected < lower, lower + (lower - x_corrected) % (upper - lower), x_corrected)
        x_corrected = np.where(x_corrected > upper, upper - (x_corrected - upper) % (upper - lower), x_corrected)
        
        return x_corrected
    
    def ad_correction(x, population, lower, upper, iteration, max_iterations):
        """
        Método de corrección determinístico y estocástico de límites.

        :param x: np.ndarray, Vector de la solución actual
        :param population: np.ndarray, Población actual de soluciones
        :param lower: np.ndarray, Límites inferiores para cada dimensión
        :param upper: np.ndarray, Límites superiores para cada dimensión
        :param iteration: int, Número de la iteración actual
        :param max_iterations: int, Número máximo de iteraciones

        :return: np.ndarray, Vector de la solución corregido
        """
        mean_population = np.mean(population, axis=0)
        std_population = np.std(population, axis=0)

        # Calcular la desviación normalizada y el valor de ajuste adaptativo
        deviations = (x - mean_population) / (np.abs(x - mean_population) + 1e-10)
        progress_ratio = iteration / max_iterations
        adaptive_factor = (1 - progress_ratio) * 0.5 + progress_ratio * 1.5
        adjustments = deviations * (np.minimum(np.abs(upper - mean_population), np.abs(lower - mean_population)) * adaptive_factor)

        # Aplicar corrección determinística basada en desviación estándar
        x_corrected = mean_population + adjustments
        x_corrected = np.where(np.abs(x_corrected - mean_population) > std_population, mean_population + deviations * std_population, x_corrected)

        # Aplicar reflexión si sigue fuera de los límites
        x_corrected = np.where(x_corrected < lower, lower + (lower - x_corrected) % (upper - lower), x_corrected)
        x_corrected = np.where(x_corrected > upper, upper - (x_corrected - upper) % (upper - lower), x_corrected)

        # Añadir componente estocástico controlado
        stochastic_component = np.random.uniform(-0.1, 0.1, size=x_corrected.shape) * (upper - lower)
        x_corrected = np.clip(x_corrected + stochastic_component, lower, upper)

        return x_corrected
    
    def cdebp(x, population, lower, upper, iteration, max_iterations):
        """
        Corrección Determinística y Estocástica Basada en la Población (CDEBP) a nivel vector.

        :param x: np.ndarray, Vector de la solución actual
        :param population: np.ndarray, Población actual de soluciones
        :param lower: np.ndarray, Límites inferiores para cada dimensión
        :param upper: np.ndarray, Límites superiores para cada dimensión
        :param iteration: int, Número de la iteración actual
        :param max_iterations: int, Número máximo de iteraciones

        :return: np.ndarray, Vector de la solución corregido
        """
        # Seleccionar los mejores individuos de la población
        best_individuals = population[np.argsort(np.linalg.norm(population, axis=1))[:int(len(population) * 0.2)]]
        
        # Calcular la media y desviación estándar de los mejores individuos
        mean_population = np.mean(best_individuals, axis=0)
        std_population = np.std(best_individuals, axis=0)
        
        # Calcular la desviación normalizada y el ajuste adaptativo
        deviations = (x - mean_population) / (np.linalg.norm(x - mean_population) + 1e-10)
        progress_ratio = iteration / max_iterations
        adaptive_factor = (1 - progress_ratio) * 0.5 + progress_ratio * 1.5
        adjustments = deviations * np.minimum(np.linalg.norm(upper - mean_population), np.linalg.norm(lower - mean_population)) * adaptive_factor
        
        # Aplicar la corrección adaptativa basada en la desviación estándar
        x_corrected = mean_population + adjustments
        x_corrected = np.where(np.linalg.norm(x_corrected - mean_population) > np.linalg.norm(std_population), mean_population + deviations * np.linalg.norm(std_population), x_corrected)
        
        # Aplicar reflexión si sigue fuera de los límites
        x_corrected = np.where(x_corrected < lower, lower + (lower - x_corrected) % (upper - lower), x_corrected)
        x_corrected = np.where(x_corrected > upper, upper - (x_corrected - upper) % (upper - lower), x_corrected)
        
        # Componente estocástica controlada
        x_final = np.clip(x_corrected + np.random.uniform(-0.1, 0.1, size=x.shape) * (upper - lower), lower, upper)
        
        return x_final

    @staticmethod
    def cdebp_2(x, population, lower, upper, iteration, max_iterations, best_solution):
        """
        Corrección Determinística y Estocástica Basada en la Población (CDEBP) a nivel vector con componente evolutiva.

        :param x: np.ndarray, Vector de la solución actual
        :param population: np.ndarray, Población actual de soluciones
        :param lower: np.ndarray, Límites inferiores para cada dimensión
        :param upper: np.ndarray, Límites superiores para cada dimensión
        :param iteration: int, Número de la iteración actual
        :param max_iterations: int, Número máximo de iteraciones
        :param best_solution: np.ndarray, Mejor solución actual en la población

        :return: np.ndarray, Vector de la solución corregido
        """
        # Seleccionar los mejores individuos de la población
        best_individuals = population[np.argsort(np.linalg.norm(population, axis=1))[:int(len(population) * 0.2)]]
        
        # Calcular la media y desviación estándar de los mejores individuos
        mean_population = np.mean(best_individuals, axis=0)
        std_population = np.std(best_individuals, axis=0)
        
        # Calcular la desviación normalizada y el ajuste adaptativo
        deviations = (x - mean_population) / (np.linalg.norm(x - mean_population) + 1e-10)
        progress_ratio = iteration / max_iterations
        adaptive_factor = (1 - progress_ratio) * 0.5 + progress_ratio * 1.5
        adjustments = deviations * np.minimum(np.linalg.norm(upper - mean_population), np.linalg.norm(lower - mean_population)) * adaptive_factor
        
        # Aplicar la corrección adaptativa basada en la desviación estándar
        x_corrected = mean_population + adjustments
        x_corrected = np.where(np.linalg.norm(x_corrected - mean_population) > np.linalg.norm(std_population), mean_population + deviations * np.linalg.norm(std_population), x_corrected)
        
        # Aplicar reflexión si sigue fuera de los límites
        x_corrected = np.where(x_corrected < lower, lower + (lower - x_corrected) % (upper - lower), x_corrected)
        x_corrected = np.where(x_corrected > upper, upper - (x_corrected - upper) % (upper - lower), x_corrected)
        
        # Componente evolutiva basada en el mejor individuo
        alpha = np.random.uniform(0, 1, size=x.shape)
        x_corrected = np.where(x_corrected < lower, alpha * lower + (1 - alpha) * best_solution, x_corrected)
        x_corrected = np.where(x_corrected > upper, alpha * upper + (1 - alpha) * best_solution, x_corrected)
        
        # Componente estocástica controlada
        x_final = np.clip(x_corrected + np.random.uniform(-0.1, 0.1, size=x.shape) * (upper - lower), lower, upper)
        
        return x_final
    
    @staticmethod 
    def cdebp_vector_follow_best(x, population, lower, upper, iteration, max_iterations):
        """
        Corrección Determinística y Estocástica Basada en la Población (CDEBP) a nivel vector siguiendo al mejor individuo.

        :param x: np.ndarray, Vector de la solución actual
        :param population: np.ndarray, Población actual de soluciones
        :param lower: np.ndarray, Límites inferiores para cada dimensión
        :param upper: np.ndarray, Límites superiores para cada dimensión
        :param iteration: int, Número de la iteración actual
        :param max_iterations: int, Número máximo de iteraciones

        :return: np.ndarray, Vector de la solución corregido
        """
        # Encontrar el mejor individuo en la población
        best_solution = population[np.argmin(np.linalg.norm(population - x, axis=1))]
        
        # Calcular la desviación normalizada y el ajuste adaptativo
        deviations = (x - best_solution) / (np.linalg.norm(x - best_solution) + 1e-10)
        progress_ratio = iteration / max_iterations
        adaptive_factor = (1 - progress_ratio) * 0.5 + progress_ratio * 1.5
        adjustments = deviations * np.minimum(np.linalg.norm(upper - best_solution), np.linalg.norm(lower - best_solution)) * adaptive_factor
        
        # Aplicar la corrección adaptativa
        x_corrected = best_solution + adjustments
        
        # Aplicar reflexión si sigue fuera de los límites
        x_corrected = np.where(x_corrected < lower, lower + (lower - x_corrected) % (upper - lower), x_corrected)
        x_corrected = np.where(x_corrected > upper, upper - (x_corrected - upper) % (upper - lower), x_corrected)
        
        # Componente estocástica controlada
        x_final = np.clip(x_corrected + np.random.uniform(-0.1, 0.1, size=x.shape) * (upper - lower), lower, upper)
        
        return x_final

    def cdebp_vector_follow_best_and_centroid(x, population, lower, upper, iteration, max_iterations):
        """
        Corrección Determinística y Estocástica Basada en la Población (CDEBP) a nivel vector siguiendo al mejor individuo y usando el centroide.

        :param x: np.ndarray, Vector de la solución actual
        :param population: np.ndarray, Población actual de soluciones
        :param lower: np.ndarray, Límites inferiores para cada dimensión
        :param upper: np.ndarray, Límites superiores para cada dimensión
        :param iteration: int, Número de la iteración actual
        :param max_iterations: int, Número máximo de iteraciones

        :return: np.ndarray, Vector de la solución corregido
        """
        # Encontrar el mejor individuo en la población
        best_solution = population[np.argmin(np.linalg.norm(population - x, axis=1))]
        
        # Calcular la desviación normalizada y el ajuste adaptativo
        deviations = (x - best_solution) / (np.linalg.norm(x - best_solution) + 1e-10)
        progress_ratio = iteration / max_iterations
        adaptive_factor = (1 - progress_ratio) * 0.5 + progress_ratio * 1.5
        adjustments = deviations * np.minimum(np.linalg.norm(upper - best_solution), np.linalg.norm(lower - best_solution)) * adaptive_factor
        
        # Aplicar la corrección adaptativa
        x_corrected = best_solution + adjustments
        
        # Método de corrección basado en el centroide
        K = 1  # Número de vectores aleatorios
        D = len(lower)
        random_vectors = np.array([lower + np.random.rand(D) * (upper - lower) for _ in range(K)])
        centroid_vector = np.mean(np.vstack([best_solution, random_vectors]), axis=0)
        centroid_vector = np.clip(centroid_vector, lower, upper)
        
        # Método de corrección de Andrea
        R = (lower + upper) / 2
        alpha = np.min(
            [
                np.where(x_corrected < lower, (R - lower) / (R - x_corrected), 1),
                np.where(x_corrected > upper, (upper - R) / (x_corrected - R), 1),
            ],
            axis=0
        )
        x_corrected = alpha * x_corrected + (1 - alpha) * R
        
        # Aplicar reflexión si sigue fuera de los límites
        x_corrected = np.where(x_corrected < lower, lower + (lower - x_corrected) % (upper - lower), x_corrected)
        x_corrected = np.where(x_corrected > upper, upper - (x_corrected - upper) % (upper - lower), x_corrected)
        
        # Componente estocástica controlada
        x_final = np.clip(x_corrected + np.random.uniform(-0.1, 0.1, size=x.shape) * (upper - lower), lower, upper)
        
        return x_final
    

    @staticmethod
    def cdebp_advanced(x, population, lower, upper, iteration, max_iterations, n_clusters=3):
        """
        Corrección Determinística y Estocástica Basada en la Población (CDEBP) Avanzada.

        :param x: np.ndarray, Vector de la solución actual
        :param population: np.ndarray, Población actual de soluciones
        :param lower: np.ndarray, Límites inferiores para cada dimensión
        :param upper: np.ndarray, Límites superiores para cada dimensión
        :param iteration: int, Número de la iteración actual
        :param max_iterations: int, Número máximo de iteraciones
        :param n_clusters: int, Número de clústeres para el análisis de la población

        :return: np.ndarray, Vector de la solución corregido
        """
        # Dividir la población en clústeres
        kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(population)
        labels = kmeans.labels_
        cluster_id = labels[np.argmin(np.linalg.norm(population - x, axis=1))]
        cluster_population = population[labels == cluster_id]
        
        # Calcular la media y desviación estándar del clúster
        mean_cluster = np.mean(cluster_population, axis=0)
        std_cluster = np.std(cluster_population, axis=0)
        
        # Calcular la desviación normalizada y el valor de ajuste adaptativo
        deviations = (x - mean_cluster) / (np.linalg.norm(x - mean_cluster) + 1e-10)
        progress_ratio = iteration / max_iterations
        adaptive_factor = (1 - progress_ratio) * 0.5 + progress_ratio * 1.5
        adjustments = deviations * np.minimum(np.linalg.norm(upper - mean_cluster), np.linalg.norm(lower - mean_cluster)) * adaptive_factor
        
        # Aplicar corrección adaptativa basada en desviación estándar del clúster
        x_corrected = mean_cluster + adjustments
        x_corrected = np.where(np.linalg.norm(x_corrected - mean_cluster) > np.linalg.norm(std_cluster), mean_cluster + deviations * np.linalg.norm(std_cluster), x_corrected)
        
        # Aplicar reflexión si sigue fuera de los límites
        x_corrected = np.where(x_corrected < lower, lower + (lower - x_corrected) % (upper - lower), x_corrected)
        x_corrected = np.where(x_corrected > upper, upper - (x_corrected - upper) % (upper - lower), x_corrected)
        
        # Componente estocástica controlada
        x_final = np.clip(x_corrected + np.random.uniform(-0.1, 0.1, size=x.shape) * (upper - lower), lower, upper)
        
        # Penalización dinámica para soluciones que salen del espacio frecuentemente
        if any(x_final < lower) or any(x_final > upper):
            penalty_factor = 1 + 0.1 * (iteration / max_iterations)
            x_final = mean_cluster + (x_final - mean_cluster) * penalty_factor
            x_final = np.clip(x_final, lower, upper)
        
        return x_final
    
    @staticmethod
    def scalar_compression_method(X, lower_bounds, upper_bounds, alpha=0.5):
        compressed_X = []
        for x, l, u in zip(X, lower_bounds, upper_bounds):
            if x < l or x > u:
                compressed_x = l + ((x - l) * (u - l)) / (u - l + alpha * (abs(x - u) + abs(x - l)))
            else:
                compressed_x = x
            compressed_X.append(compressed_x)
        return compressed_X
    
    
   