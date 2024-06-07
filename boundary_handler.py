from typing import Callable
from algorithm import Algorithm
import numpy as np


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

        presicion_boxcontraints = np.mean(np.abs(points - p1))
        presicion_other_boxcontraints = np.mean(
            np.abs(points - p2)
        )

        return (presicion_boxcontraints, presicion_other_boxcontraints)

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
    def handle_boundary_constraint(value, lower_limit, upper_limit):
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

    def clamp_position(position, lower_bound, upper_bound):
        """
        Clamps the particle position to the given bounds.
        
        Parameters:
        - position (np.array): The current position of the particle.
        - lower_bound (np.array): The lower bound of the search space.
        - upper_bound (np.array): The upper bound of the search space.
        
        Returns:
        - np.array: The clamped position of the particle.
        """
        return np.maximum(lower_bound, np.minimum(position, upper_bound))

    def random_reinitialize(position, lower_bound, upper_bound):
        """
        Randomly reinitializes the particle position if it exceeds the bounds.
        
        Parameters:
        - position (np.array): The current position of the particle.
        - lower_bound (np.array): The lower bound of the search space.
        - upper_bound (np.array): The upper bound of the search space.
        
        Returns:
        - np.array: The reinitialized position of the particle.
        """
        reinitialized_position = np.where(
            (position < lower_bound) | (position > upper_bound),
            np.random.uniform(lower_bound, upper_bound),
            position
        )
        return reinitialized_position
    
    def reflect_position(position, lower_bound, upper_bound):
        """
        Reflects the particle position if it exceeds the bounds.
        
        Parameters:
        - position (np.array): The current position of the particle.
        - lower_bound (np.array): The lower bound of the search space.
        - upper_bound (np.array): The upper bound of the search space.
        
        Returns:
        - np.array: The reflected position of the particle.
        """
        reflected_position = position.copy()
        below_lower = position < lower_bound
        above_upper = position > upper_bound
        
        reflected_position[below_lower] = lower_bound[below_lower] + (lower_bound[below_lower] - position[below_lower])
        reflected_position[above_upper] = upper_bound[above_upper] - (position[above_upper] - upper_bound[above_upper])
        
        return reflected_position
    
    def shrink_position(position, lower_bound, upper_bound):
        """
        Shrinks the particle position to be within the bounds proportionally.
        
        Parameters:
        - position (np.array): The current position of the particle.
        - lower_bound (np.array): The lower bound of the search space.
        - upper_bound (np.array): The upper bound of the search space.
        
        Returns:
        - np.array: The shrunk position of the particle.
        """
        shrunk_position = position.copy()
        shrunk_position[position < lower_bound] = lower_bound[position < lower_bound]
        shrunk_position[position > upper_bound] = upper_bound[position > upper_bound]
        
        return shrunk_position
    
    def eliminate_particles(positions, lower_bound, upper_bound):
        """
        Eliminates particles that exceed the bounds.
        
        Parameters:
        - positions (np.array): The current positions of the particles.
        - lower_bound (np.array): The lower bound of the search space.
        - upper_bound (np.array): The upper bound of the search space.
        
        Returns:
        - np.array: The positions of the particles that are within bounds.
        """
        within_bounds = np.all((positions >= lower_bound) & (positions <= upper_bound), axis=1)
        return positions[within_bounds]
     #2009

    # Función de Reinicialización Aleatoria
    def random_reinitialization(x, L, U):
        """
        Reinicializa aleatoriamente los componentes del vector que están fuera de los límites.

        Parámetros:
        - x (np.array): El vector actual de posiciones.
        - L (np.array): El límite inferior del espacio de búsqueda.
        - U (np.array): El límite superior del espacio de búsqueda.

        Retorna:
        - np.array: El vector de posiciones con componentes reinicializados dentro de los límites.
        """
        for i in range(len(x)):
            if x[i] < L[i] or x[i] > U[i]:
                x[i] = np.random.uniform(L[i], U[i])
        return x


    # Función de Reparación de Bordes
    def boundary_repair(x, L, U):
        """
        Ajusta los componentes del vector que están fuera de los límites a los valores de los límites.

        Parámetros:
        - x (np.array): El vector actual de posiciones.
        - L (np.array): El límite inferior del espacio de búsqueda.
        - U (np.array): El límite superior del espacio de búsqueda.

        Retorna:
        - np.array: El vector de posiciones con componentes ajustados a los límites.
        """
        for i in range(len(x)):
            if x[i] < L[i]:
                x[i] = L[i]
            elif x[i] > U[i]:
                x[i] = U[i]
        return x

    # Función de Reflejo
    def reflection(x, L, U):
        """
        Refleja los componentes del vector que están fuera de los límites dentro del rango permitido.

        Parámetros:
        - x (np.array): El vector actual de posiciones.
        - L (np.array): El límite inferior del espacio de búsqueda.
        - U (np.array): El límite superior del espacio de búsqueda.

        Retorna:
        - np.array: El vector de posiciones con componentes reflejados dentro de los límites.
        """
        for i in range(len(x)):
            if x[i] < L[i]:
                x[i] = L[i] + (L[i] - x[i])
            elif x[i] > U[i]:
                x[i] = U[i] - (x[i] - U[i])
        return x


    # Función de Envolvimiento
    def wrapping(x, L, U):
        """
        Envuelve los componentes del vector que están fuera de los límites dentro del rango permitido, similar a un comportamiento cíclico.

        Parámetros:
        - x (np.array): El vector actual de posiciones.
        - L (np.array): El límite inferior del espacio de búsqueda.
        - U (np.array): El límite superior del espacio de búsqueda.

        Retorna:
        - np.array: El vector de posiciones con componentes envueltos dentro de los límites.
        """
        for i in range(len(x)):
            if x[i] < L[i]:
                x[i] = U[i] - (L[i] - x[i]) % (U[i] - L[i])
            elif x[i] > U[i]:
                x[i] = L[i] + (x[i] - U[i]) % (U[i] - L[i])
        return x
