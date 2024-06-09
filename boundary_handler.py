from typing import Callable
from algorithm import Algorithm
import numpy as np

# TODO: Graficar con promedio y programar centroide, leer articulos relacionados, investigar 2008
# TODO: 2008 y 2014 no existen articulos

class BoundaryHandler(Algorithm):

    def compare(
        self, 
        method_1: Callable,
        method_2: Callable, 
        upper: np.array, 
        lower: np.array
    ):
        points = self.generate(upper - 1, lower - 1)

        p1 = np.array([method_1(upper, lower, p) for p in points])
        p2 = np.array(
            [method_2(upper, lower, p) for p in points]
        )

        presicion_p1 = np.mean(np.abs(points - p1))
        presicion_p2 = np.mean(
            np.abs(points - p2)
        )

        return (presicion_p1, presicion_p2)

    """Bounds Methdos Default"""

    @staticmethod
    # Metodo boundary para manejar restriccion de limites de individuo
    def boundary(superior: list, inferior: list, individuo: np.array) -> np.array:
        individuo_corregido = np.copy(individuo)

        for i in range(len(individuo)):
            if individuo[i] < inferior[i]:
                individuo_corregido[i] = inferior[i]
            elif individuo[i] > superior[i]:
                individuo_corregido[i] = superior[i]

        return individuo_corregido

    @staticmethod
    # Metodo de reflex para manejar restricciones  de limites dentro de los valores de un individuo
    def reflex(superior, inferior, individuo) -> np.array:
        range_width = superior - inferior
        individuo_corregido = np.copy(individuo)

        for j in range(len(individuo)):
            if individuo_corregido[j] < inferior[j]:
                individuo_corregido[j] = (
                    inferior[j]
                    + (inferior[j] - individuo_corregido[j]) % range_width[j]
                )
            elif individuo_corregido[j] > superior[j]:
                individuo_corregido[j] = (
                    superior[j]
                    - (individuo_corregido[j] - superior[j]) % range_width[j]
                )

        return individuo_corregido

    @staticmethod
    # Metodo random para manejar restricciones de limites dentro de los valores de un individuo
    def random(superior: list, inferior: list, individuo: np.array) -> np.array:
        individuo_corregido = []

        for sup, ind, inf in zip(superior, individuo, inferior):
            if ind > sup or ind < inf:
                ind = inf + np.random.uniform(0, 1) * (sup - inf)
                individuo_corregido.append(ind)
            else:
                individuo_corregido.append(ind)

        return np.array(individuo_corregido)

    # Metodo wrapping para maenjar restricciones de limites dentro de los valores de un individuo
    def wrapping(superior, inferior, individuo) -> np.array:
        range_width = abs(superior - inferior)
        individuo_corregido = np.copy(individuo)

        for i in range(len(individuo)):
            if individuo_corregido[i] < inferior[i]:
                individuo_corregido[i] = (
                    superior[i]
                    - (inferior[i] - individuo_corregido[i]) % range_width[i]
                )
            elif individuo_corregido[i] > superior[i]:
                individuo_corregido[i] = (
                    inferior[i]
                    + (individuo_corregido[i] - superior[i]) % range_width[i]
                )

        return individuo_corregido

    # Método del faro para atraer partículas fuera de los límites hacia el centro del espacio de búsqueda
    @staticmethod
    def faro(superior, inferior, individuo, margen=0.1):
        centro = (superior + inferior) / 2
        distancia = np.linalg.norm(individuo - centro)
        if distancia > margen:
            direccion = centro - individuo
            individuo += 0.1 * direccion  # Factor de atracción ajustable
        return individuo

    @staticmethod
    def attract(superior, inferior, individuo, atraccion=0.1):
        centro = (np.array(superior) + np.array(inferior)) / 2
        if any(individuo < inferior) or any(individuo > superior):
            direccion = centro - individuo
            individuo += atraccion * direccion
        return individuo

    @staticmethod
    def restriccion_ajuste_aptitud(
        superior, inferior, individuo, evaluar, max_intentos=100
    ):
        # Ajustar la posición hacia un punto dentro de los límites con alta aptitud
        for _ in range(max_intentos):
            nueva_posicion = np.random.uniform(inferior, superior)
            if evaluar(nueva_posicion) > evaluar(individuo):
                return nueva_posicion
        # Si no se encuentra una posición con alta aptitud, se mantiene la posición actual
        return individuo

    # Método collision para manejar colisiones entre partículas
    def collision(particles: list):
        n = len(particles)
        for i in range(n):
            for j in range(i + 1, n):
                if (
                    np.linalg.norm(particles[i] - particles[j]) < 0.1
                ):  # Definir umbral de colisión adecuado
                    # Ajustar velocidades de las partículas i y j (simulando el efecto de la colisión)
                    particles[i].velocity *= -1
                    particles[j].velocity *= -1

    """
    1999 Lampiden Mixed DE
    """
    
    @staticmethod
    def replace_out_of_bounds(upper_bound, lower_bound, value) -> float:
        if np.any(value < lower_bound) or np.any(value > upper_bound):
            value = np.random.uniform(lower_bound, upper_bound)
        return value

    @staticmethod
    def repeat_unitl_within_bounds(upper_bound, lower_bound, value):
        while np.any(value < lower_bound) or np.any(value > upper_bound):
            value = np.random.uniform(lower_bound, upper_bound)
        return value

    """
    2002 Lampiden Constraint
    """

    @staticmethod
    def handle_boundary_constraint(upper_limit, lower_limit, value):
        if np.any(value < lower_limit) or np.any(value > upper_limit):
            return np.random.uniform(lower_limit, upper_limit)
        else:
            return value

    "2004 Zhang Handling"

    @staticmethod
    def periodic_mode(u, l, x):
        s = u - l  # rango de la dimensión
        z = np.copy(x)

        for d in range(len(x)):
            if x[d] < l[d]:
                z[d] = u[d] - (x[d] % s[d])
            elif x[d] > u[d]:
                z[d] = l[d] + (x[d] % s[d])
            else:
                z[d] = x[d]

        return z
    
    """2004 – Robinson Particle"""

    @staticmethod
    # Muros Absorbentes
    def absorbing_walls(position, velocity, bounds):
        for i in range(len(position)):
            if position[i] < bounds[i][0] or position[i] > bounds[i][1]:
                velocity[i] = 0
        return velocity

    @staticmethod
    # Muros Reflectantes
    def reflecting_walls(position, velocity, bounds):
        for i in range(len(position)):
            if position[i] < bounds[i][0] or position[i] > bounds[i][1]:
                velocity[i] = -velocity[i]
        return velocity

    @staticmethod
    # Muros Invisibles
    def invisible_walls(position, fitness_function, bounds):
        for i in range(len(position)):
            if position[i] < bounds[i][0] or position[i] > bounds[i][1]:
                return (
                    np.inf
                )  # Asignar un valor muy alto al fitness para partículas fuera del límite
        return fitness_function(position)

    """2004 Purchia Experimentation"""

    @staticmethod
    def proyection(u, l, x):
        return np.maximum(l, np.minimum(x, u))

    @staticmethod
    def wrapping_purchia(u, l, x):
        return l + np.mod(x - l, u - l)
    
    
    """2004 Mendes Population"""
    
    @staticmethod
    def apply_bound_restriction(positions, velocities, lower_bounds, upper_bounds):
        for i in range(positions.shape[0]):
            for d in range(positions.shape[1]):
                if positions[i, d] < lower_bounds[d]:
                    positions[i, d] = lower_bounds[d]
                    velocities[i, d] = 0  # Opcional: Resetear velocidad a 0
                elif positions[i, d] > upper_bounds[d]:
                    positions[i, d] = upper_bounds[d]
                    velocities[i, d] = 0  # Opcional: Resetear velocidad a 0
        return positions

    """
    2005 Huang Hybrid
    """
    
    @staticmethod
    def apply_damping_boundaries(position, velocity):
        for i in range(len(position)):
            if position[i] < 0 or position[i] > 1:
                alpha = np.random.uniform(0, 1)
                velocity[i] = -alpha * velocity[i]
                position[i] = np.clip(position[i], 0, 1)
    
    """
    2006 Clerc Confinements
    """

    @staticmethod
    def no_Confinement(position, velocity):
        return position, velocity

    @staticmethod
    def no_Confinement_ArtificialLandscape(position, velocity, lower_bound, upper_bound):
        if np.any(position < lower_bound) or np.any(position > upper_bound):
            position = np.clip(position, lower_bound, upper_bound)
        return position, velocity

    @staticmethod 
    def standard(position, velocity, lower_bound, upper_bound):
        position = np.clip(position, lower_bound, upper_bound)
        velocity = np.where(position == lower_bound, -velocity, velocity)
        velocity = np.where(position == upper_bound, -velocity, velocity)
        return position, velocity

    @staticmethod
    def deterministic_back(position, velocity, lower_bound, upper_bound, gamma=0.5):
        if np.any(position < lower_bound) or np.any(position > upper_bound):
            position = np.clip(position, lower_bound, upper_bound)
            velocity *= -gamma
        return position, velocity
    
    @staticmethod
    def random_back(position, velocity, lower_bound, upper_bound):
        if np.any(position < lower_bound) or np.any(position > upper_bound):
            random_factor = np.random.uniform(0, 1, size=position.shape)
            position = np.clip(position, lower_bound, upper_bound)
            velocity *= -random_factor
        return position, velocity

    @staticmethod
    def consistent(position, velocity, lower_bound, upper_bound):
        if np.any(position < lower_bound) or np.any(position > upper_bound):
            for i in range(len(position)):
                if position[i] < lower_bound[i] or position[i] > upper_bound[i]:
                    velocity[i] = -velocity[i]
        return position, velocity

    @staticmethod
    def hyperbolic(position, velocity, lower_bound, upper_bound):
        if np.any(position < lower_bound) or np.any(position > upper_bound):
            position = np.clip(position, lower_bound, upper_bound)
            velocity = -velocity / (1 + np.abs(velocity))
        return position, velocity

    @staticmethod
    def relativity_like(position, velocity, lower_bound, upper_bound, c=1):
        if np.any(position < lower_bound) or np.any(position > upper_bound):
            position = np.clip(position, lower_bound, upper_bound)
            velocity = -velocity / (1 + (velocity**2 / c**2))
        return position, velocity

    @staticmethod
    def random_forth(position, velocity, lower_bound, upper_bound):
        if np.any(position < lower_bound) or np.any(position > upper_bound):
            random_factor = np.random.uniform(0, 1, size=position.shape)
            position = np.clip(position, lower_bound, upper_bound)
            velocity *= random_factor
        return position, velocity

    @staticmethod       
    def hybrid(position, velocity, lower_bound, upper_bound):
        position = np.clip(position, lower_bound, upper_bound)
        velocity = np.where(position == lower_bound, -velocity, velocity)
        velocity = np.where(position == upper_bound, -velocity, velocity)
        
        if np.any(position < lower_bound) or np.any(position > upper_bound):
            if np.any(position < lower_bound) or np.any(position > upper_bound):
                position = np.clip(position, lower_bound, upper_bound)
                velocity = -velocity / (1 + np.abs(velocity))
        return position, velocity
    
    """
    2007 Clerc Confinements
    """
    
    @staticmethod
    def clamp_position(position, lower_bound, upper_bound):
        return np.maximum(lower_bound, np.minimum(position, upper_bound))

    @staticmethod
    def random_reinitialize(position, lower_bound, upper_bound):
        reinitialized_position = np.where(
            (position < lower_bound) | (position > upper_bound),
            np.random.uniform(lower_bound, upper_bound),
            position
        )
        return reinitialized_position
    
    @staticmethod
    def reflect_position(position, lower_bound, upper_bound):
        reflected_position = position.copy()
        below_lower = position < lower_bound
        above_upper = position > upper_bound
        
        reflected_position[below_lower] = lower_bound[below_lower] + (lower_bound[below_lower] - position[below_lower])
        reflected_position[above_upper] = upper_bound[above_upper] - (position[above_upper] - upper_bound[above_upper])
        
        return reflected_position
    
    @staticmethod
    def shrink_position(position, lower_bound, upper_bound):
        shrunk_position = position.copy()
        shrunk_position[position < lower_bound] = lower_bound[position < lower_bound]
        shrunk_position[position > upper_bound] = upper_bound[position > upper_bound]
        
        return shrunk_position
    
    @staticmethod
    def eliminate_particles(positions, lower_bound, upper_bound):
        within_bounds = np.all((positions >= lower_bound) & (positions <= upper_bound), axis=1)
        return positions[within_bounds]
    
    """ 2009 """

    # Función de Reinicialización Aleatoria
    def random_reinitialization(x, L, U):
        for i in range(len(x)):
            if x[i] < L[i] or x[i] > U[i]:
                x[i] = np.random.uniform(L[i], U[i])
        return x


    # Función de Reparación de Bordes
    def boundary_repair(x, L, U):
        for i in range(len(x)):
            if x[i] < L[i]:
                x[i] = L[i]
            elif x[i] > U[i]:
                x[i] = U[i]
        return x

    # Función de Reflejo
    def reflection(x, L, U):
        for i in range(len(x)):
            if x[i] < L[i]:
                x[i] = L[i] + (L[i] - x[i])
            elif x[i] > U[i]:
                x[i] = U[i] - (x[i] - U[i])
        return x


    # Función de Envolvimiento
    def wrapping(x, L, U):
        for i in range(len(x)):
            if x[i] < L[i]:
                x[i] = U[i] - (L[i] - x[i]) % (U[i] - L[i])
            elif x[i] > U[i]:
                x[i] = L[i] + (x[i] - U[i]) % (U[i] - L[i])
        return x
    
    """ 2011 Chu Handling"""
    
    def handle_bounds_random_chu(upper_bound, lower_bound, position):
        for i in range(len(position)):
            if position[i] < lower_bound[i] or position[i] > upper_bound[i]:
                position[i] = np.random.uniform(lower_bound[i], upper_bound[i])
        return position
    
    
    def handle_bounds_absorbing_chu(upper_bound, lower_bound, position):
        for i in range(len(position)):
            if position[i] < lower_bound[i]:
                position[i] = lower_bound[i]
            elif position[i] > upper_bound[i]:
                position[i] = upper_bound[i]
        return position
    
    def handle_bounds_reflecting(upper_bound, lower_bound, position):
         for i in range(len(position)):
            if position[i] < lower_bound[i]:
                position[i] = 2 * lower_bound[i] - position[i]
            elif position[i] > upper_bound[i]:
                position[i] = 2 * upper_bound[i] - position[i]
         return position
    
    
    """ 2011 Omeltschuk heterogeneous """
    @staticmethod
    def phi_static(x, f, constraints, K=1e6):
        # Evaluar si x cumple con todas las restricciones
        satisfied_constraints = [g(x) <= 0 for g in constraints['inequality']] + [abs(h(x)) <= 1e-6 for h in constraints['equality']]
        s = sum(satisfied_constraints)
        l = len(constraints['inequality'])
        m = len(constraints['equality'])
        
        # Calcular la función penalizada
        if s == l + m:
            return f(x)
        else:
            return K * (1 - s / (l + m))
        
    """ 2012 Shi Experimental """
    @staticmethod
    def classical_boundary_handling(upper_bound, lower_bound, position):
        new_position = np.where(position > upper_bound, upper_bound, position)
        new_position = np.where(new_position < lower_bound, lower_bound, new_position)
        return new_position

    @staticmethod
    def deterministic_boundary_handling(position, previous_position, lower_bound, upper_bound):
        new_position = np.copy(position)
        new_position = np.where(position > upper_bound, (previous_position + upper_bound) / 2, new_position)
        new_position = np.where(position < lower_bound, (previous_position + lower_bound) / 2, new_position)
        return new_position
    
    @staticmethod
    def stochastic_boundary_handling(upper_bound, lower_bound, particles):
        num_particles, num_dimensions = particles.shape
        for i in range(num_particles):
            for j in range(num_dimensions):
                if particles[i, j] < lower_bound[j] or particles[i, j] > upper_bound[j]:
                    particles[i, j] = np.random.uniform(lower_bound[j], upper_bound[j])
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
                if positions[i, d] < solution_space[0] or positions[i, d] > solution_space[1]:
                    positions[i, d] = np.clip(positions[i, d], solution_space[0], solution_space[1])
                    velocities[i, d] *= -np.random.rand()
                elif (solution_space[0] + epsilon < positions[i, d] < solution_space[0] or
                    solution_space[1] < positions[i, d] < solution_space[1] - epsilon):
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
                new_position[i] = lower_bound[i] + (new_position[i] % (2 * upper_bound[i] - lower_bound[i]))
        
        return new_position
    
    """ 2013 Padhye Boundary"""
    @staticmethod
    def inverse_parabolic_distribution(x_not, x_c, alpha=1.2):
        d = np.linalg.norm(x_c - x_not)
        a = (alpha**2 * d**2) / (np.pi / 2 + np.arctan(1 / alpha))
        
        def probability_distribution(x):
            return a / ((x - d)**2 + alpha**2 * d**2)
        
        x1 = np.minimum(x_not, x_c)  # Assuming x_not and x_c are vectors
        x2 = np.maximum(x_not, x_c)
        
        r = np.random.rand()
        x_prime = d + alpha * d * np.tan(r * np.arctan((x2 - d) / (alpha * d)))
        
        return x_prime
    
    """ 2013 Wessing Repair"""
    @staticmethod
    def projection_repair(x, lower, upper):
        return max(lower, min(x, upper))

    @staticmethod
    def reflection_repair(x, lower, upper):
        if x < lower:
            return BoundaryHandler.reflection_repair(lower + (lower - x), lower, upper)
        elif x > upper:
            return BoundaryHandler.reflection_repair(upper + (upper - x), lower, upper)
        return x

    @staticmethod
    def wrapping_reapir(x, lower, upper):
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
        sigma = (np.math.gamma(1 + Lambda) * np.sin(np.pi * Lambda / 2) / 
                (np.math.gamma((1 + Lambda) / 2) * Lambda * 2**((Lambda - 1) / 2)))**(1 / Lambda)
        u = np.random.normal(0, sigma, size=1)
        v = np.random.normal(0, 1, size=1)
        step = u / np.abs(v)**(1 / Lambda)
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
                    new_nests[i, j] = BoundaryHandler.evolutionary_boundary_handling(new_nests[i, j], lb[j], ub[j], best_nest[j])
            
            new_fitness = np.apply_along_axis(obj_func, 1, new_nests)
            for i in range(n):
                if new_fitness[i] < fitness[i]:
                    nests[i] = new_nests[i]
                    fitness[i] = new_fitness[i]
            
            best_nest = nests[np.argmin(fitness)]
        
        return best_nest, np.min(fitness)
    
    """ 2015 Juarez Novel """
    @staticmethod
    def calculate_centroid_2015(population, feasible_set, infeasible_set, mutant_vector, bounds):
        def random_within_bounds(bounds):
            return np.array([np.random.uniform(low, high) for low, high in bounds])

        AFS = len(feasible_set)
        if AFS > 0 and np.random.rand() > 0.5:
            wc1 = feasible_set[np.random.randint(AFS)]
        else:
            wc1 = min(infeasible_set, key=lambda x: sum(np.maximum(0, np.array(bounds)[:, 0] - x) + np.maximum(0, x - np.array(bounds)[:, 1])))

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
            d_prime = dv + alpha * dv * np.tan(r * np.arctan((bounds[1] - dv) / (alpha * dv)))
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
        
    def repair_scaling(infeasible_value, lower_bound_old, upper_bound_old, lower_bound_new, upper_bound_new):
        S_old = upper_bound_old - lower_bound_old
        S_new = upper_bound_new - lower_bound_new
        return upper_bound_new - ((upper_bound_old - infeasible_value) * S_old / S_new)
    
    def repair_stochastic_new_boundaries(infeasible_value, lower_bound_new, upper_bound_new):
        if infeasible_value > upper_bound_new:
            return upper_bound_new - 0.50 * np.random.random() * (upper_bound_new - lower_bound_new)
        elif infeasible_value < lower_bound_new:
            return lower_bound_new + 0.50 * np.random.random() * (upper_bound_new - lower_bound_new)

    
    """ 2016 Agarwl Experimental """
    def reflect(position, lower_bound, upper_bound):
        for i in range(len(position)):
            if position[i] < lower_bound[i]:
                position[i] = lower_bound[i] + (lower_bound[i] - position[i])
            elif position[i] > upper_bound[i]:
                position[i] = upper_bound[i] - (position[i] - upper_bound[i])
        return position

    def nearest(position, lower_bound, upper_bound):
        for i in range(len(position)):
            if position[i] < lower_bound[i]:
                position[i] = lower_bound[i]
            elif position[i] > upper_bound[i]:
                position[i] = upper_bound[i]
        return position

    def hyperbolic_2016(position, velocity, lower_bound, upper_bound):
        for i in range(len(position)):
            if velocity[i] > 0:
                velocity[i] = velocity[i] / (1 + velocity[i] / (upper_bound[i] - position[i]))
            elif velocity[i] < 0:
                velocity[i] = velocity[i] / (1 + velocity[i] / (position[i] - lower_bound[i]))
        return velocity

    """ 2016 Gandomi Evolutionary """
    


    """ 2019 Efren Juarez """
    @staticmethod    
    def calculate_centroid(X, l, u, SFS, SIS, K):

        def generar_vector_aleatorio(l, u):
            return np.random.uniform(l, u)

        # Si X está dentro de los límites, regresamos X
        if np.all((l <= X) & (X <= u)):
            return X

        # Seleccionamos W_p de acuerdo a AFS
        AFS = len(SFS)
        if AFS > 0 and np.random.random() > 0.5:
            W_p = np.random.choice(SFS)
        else:
            W_p = min(SIS, key=lambda sol: np.sum(np.maximum(0, sol - u) + np.maximum(0, l - sol)))

        # Generamos K vectores aleatorios W_r1, ..., W_rK
        W_r = [generar_vector_aleatorio(l, u) for _ in range(K)]

        # Calculamos el centroide
        centroid = (W_p + np.sum(W_r, axis=0)) / (K + 1)

        return centroid