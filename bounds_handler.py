import numpy as np
class BoundaryHandler:
    
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
                    superior[i] - (inferior[i] - individuo_corregido[i]) % range_width[i]
                )
            elif individuo_corregido[i] > superior[i]:
                individuo_corregido[i] = (
                    inferior[i] + (individuo_corregido[i] - superior[i]) % range_width[i]
                )
                
        return individuo_corregido
    
     # Método del faro para atraer partículas fuera de los límites hacia el centro del espacio de búsqueda
    @staticmethod
    def faro(superior, inferior, individuo, margen = 0.1):
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
    def restriccion_ajuste_aptitud(superior, inferior, individuo, evaluar, max_intentos=100):
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
                if np.linalg.norm(particles[i] - particles[j]) < 0.1:  # Definir umbral de colisión adecuado
                    # Ajustar velocidades de las partículas i y j (simulando el efecto de la colisión)
                    particles[i].velocity *= -1
                    particles[j].velocity *= -1
    
    
    
    """
    1999 Lampiden Mixed DE
    """
    
    
    @staticmethod    
    def replace_out_of_bounds(value, lower_bound, upper_bound) -> float:
        if value < lower_bound and value > upper_bound:
            value = np.random.uniform(lower_bound, upper_bound)
        return value
    
    
    @staticmethod 
    def repeat_unitl_within_bounds(value, lower_bound, upper_bound):
        while value < lower_bound and value > upper_bound:
            value = np.random.uniform(lower_bound, upper_bound)
        return value
    
    """
    2002 Lampiden Constraint
    """
    
    @staticmethod
    def handle_boundary_constraint(value, lower_limit, upper_limit):
        if value < lower_limit or value > upper_limit:
            return np.random.uniform(lower_limit, upper_limit)
        else:
            return value