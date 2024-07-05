import numpy as np

class BCHM:
    
    @staticmethod
    def reflection(upper, lower, x):
        range_width = upper - lower
        new_x = np.copy(x)

        for j in range(len(x)):
            if new_x[j] < lower[j]:
                new_x[j] = lower[j] + (lower[j] - new_x[j]) % range_width[j]
            elif new_x[j] > upper[j]:
                new_x[j] = upper[j] - (new_x[j] - upper[j]) % range_width[j]

        return new_x
    
    @staticmethod
    def boundary(x, lower_bound, upper_bound):
        x_projected = np.copy(x)
        x_projected = np.where(x_projected < lower_bound, lower_bound, x_projected)
        x_projected = np.where(x_projected > upper_bound, upper_bound, x_projected)
        return x_projected
    
    @staticmethod
    def random_component(upper, lower, x):
        # Crear una máscara para identificar los elementos fuera de los límites
        mask_lower = x < lower
        mask_upper = x > upper
        # Crear una copia de x para no modificar el original
        corrected_x = np.copy(x)        
        # Asignar valores aleatorios dentro de los límites solo a los elementos fuera de los límites
        corrected_x[mask_lower] = np.random.uniform(lower[mask_lower], upper[mask_lower])
        corrected_x[mask_upper] = np.random.uniform(lower[mask_upper], upper[mask_upper])        
        return corrected_x
    
    @staticmethod
    def random_all(upper, lower, x):
        return np.random.uniform(lower, upper, size=x.shape)

    @staticmethod
    def wrapping(upper, lower, x) -> np.ndarray:
        range_width = abs(upper - lower)
        new_x = np.copy(x)

        for i in range(len(x)):
            if new_x[i] < lower[i]:
                new_x[i] = upper[i] - (lower[i] - new_x[i]) % range_width[i]
            elif new_x[i] > upper[i]:
                new_x[i] = lower[i] + (new_x[i] - upper[i]) % range_width[i]

        return new_x

    @staticmethod
    def evolutionary(z, lb, ub, best_solution):
        a = np.random.rand()
        b = np.random.rand()

        x = np.copy(z)

        x[z < lb] = a * lb[z < lb] + (1 - a) * best_solution[z < lb]
        x[z > ub] = b * ub[z > ub] + (1 - b) * best_solution[z > ub]

        return x
    
    @staticmethod
    def centroid(X, population, lower, upper, SFS, SIS, gbest_individual, K=1):
        if len(SFS) > 0 and np.random.rand() > 0.5:
            random_position_index = np.random.choice(SFS) # Elegir una posición aleatoria de SFS
            Wp = population[random_position_index] # Obtener el individuo de la población en esa posición            
        else:
            if len(SIS) > 0:
                Wp = gbest_individual
            else:                
                Wp = population[np.random.randint(population.shape[0])]

        Wr = [BCHM.random_component(upper, lower, X) for _ in range(K)]

        Wr = np.array(Wr)

        sum_components = Wp + Wr.sum(axis=0)

        result = sum_components / (K + 1)

        return result
    

    @staticmethod
    def vector_wise_correction(upper, lower, x, method="midpoint"):
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
    
    def beta(individual, lower_bound, upper_bound, population):
        corrected_individual = np.copy(individual)
        population_mean = np.mean(population, axis=0)
        population_var = np.var(population, axis=0)
        
        for i in range(len(individual)):
            m_i = (population_mean[i] - lower_bound[i]) / (upper_bound[i] - lower_bound[i])
            v_i = population_var[i] / (upper_bound[i] - lower_bound[i])**2
            
            if m_i == 0:
                m_i = 0.1
            elif m_i == 1:
                m_i = 0.9
            
            if v_i == 0:
                v_i = 1e-6
            
            alpha_i = m_i * ((m_i * (1 - m_i) / v_i) - 1)
            beta_i = alpha_i * (1 - m_i) / m_i
            
            if alpha_i <= 0 or beta_i <= 0:
                alpha_i = beta_i = 1  
            
            corrected_value = np.random.beta(alpha_i, beta_i) * (upper_bound[i] - lower_bound[i]) + lower_bound[i]
            corrected_individual[i] = corrected_value
        
        return corrected_individual


    @staticmethod
    def ADS(x, lower, upper, X_mejor, iter, max_iter, alpha_min=0.1, alpha_max=0.9, beta_min=0.1, beta_max=0.9):
        for j in range(len(x)):
            alpha = alpha_min + (alpha_max - alpha_min) * (iter / max_iter)**2
            beta = beta_min + (beta_max - beta_min) * (1 - iter / max_iter)**2
            if x[j] < lower[j]:
                x[j] = lower[j] + alpha * (lower[j] - x[j]) + beta * (X_mejor[j] - lower[j])
            elif x[j] > upper[j]:
                x[j] = upper[j] - alpha * (x[j] - upper[j]) + beta * (upper[j] - X_mejor[j])
        return x
        
    def centroid_repair(X, population, lower, upper, K=1):
        NP, D = population.shape
        
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
                Wp = population[np.random.randint(NP)]

        Wr = np.empty((K, D))
        for i in range(K):
            Wi = np.copy(X)
            mask_lower = Wi < lower
            mask_upper = Wi > upper
            Wi[mask_lower] = lower[mask_lower] + np.random.rand(np.sum(mask_lower)) * (upper[mask_lower] - lower[mask_lower])
            Wi[mask_upper] = lower[mask_upper] + np.random.rand(np.sum(mask_upper)) * (upper[mask_upper] - lower[mask_upper])
            Wr[i] = Wi
        
        Xc = (Wp + Wr.sum(axis=0)) / (K + 1)

        return Xc
    
    def adaptive_centroid   (X, population, lower, upper, generation, max_generations):
        NP, D = population.shape

        # Selecciona soluciones factibles e infactibles
        SFS = population[np.all((population >= lower) & (population <= upper), axis=1)]
        SIS = population[np.any((population < lower) | (population > upper), axis=1)]
        AFS = len(SFS)

        K = max(1, int(np.ceil((max_generations - generation) / max_generations * 5)))

        if AFS > 0 and np.random.rand() > 0.5:
            Wp = SFS[np.random.randint(AFS)]
        else:
            if len(SIS) > 0:
                violations = np.sum(np.maximum(0, lower - SIS) + np.maximum(0, SIS - upper), axis=1)
                Wp = SIS[np.argmin(violations)]
            else:
                Wp = population[np.random.randint(NP)]

        Wr = np.empty((K, D))
        for i in range(K):
            Wi = np.copy(X)
            mask_lower = Wi < lower
            mask_upper = Wi > upper
            Wi[mask_lower] = lower[mask_lower] + np.random.rand(np.sum(mask_lower)) * (upper[mask_lower] - lower[mask_lower])
            Wi[mask_upper] = lower[mask_upper] + np.random.rand(np.sum(mask_upper)) * (upper[mask_upper] - lower[mask_upper])
            Wr[i] = Wi

        Xc = (Wp + Wr.sum(axis=0)) / (K + 1)

        return Xc