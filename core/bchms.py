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
    def random(upper, lower, x):
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
    def evolutionary(z, lb, ub, x_b):
        adjusted_z = np.copy(z)

        a = np.random.rand(len(z))
        b = np.random.rand(len(z))

        below_lb = z < lb
        adjusted_z[below_lb] = a[below_lb] * lb[below_lb] + (1 - a[below_lb]) * x_b[below_lb]

        above_ub = z > ub
        adjusted_z[above_ub] = b[above_ub] * ub[above_ub] + (1 - b[above_ub]) * x_b[above_ub]

        return adjusted_z
    
    @staticmethod
    def centroid_method(X, population, lower, upper, SFS, SIS, violations , K=1):
        NP, D = population.shape
        
    
        AFS = len(SFS)
        
        if AFS > 0 and np.random.rand() > 0.5:
            Wp = SFS[np.random.randint(AFS)]
        else:
            if len(SIS) > 0:
                Wp = SIS[np.argmin(violations[violations > 0])]
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

    def res_and_rand(X, F, bounds, i, max_resamples=3):
        D = X.shape[1]  
        NP = X.shape[0]  
        lower, upper = bounds

        no_res = 0
        valid = False
        while no_res < max_resamples * D and not valid:
            indices = np.arange(NP)
            indices = np.delete(indices, i)
            r1, r2, r3 = np.random.choice(indices, 3, replace=False)

            V = X[r1] + F * (X[r2] - X[r3])

            valid = np.all((V >= lower) & (V <= upper))

            no_res += 1

        if not valid:
            V = np.clip(V, lower, upper)

        return V
    
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
        
    def new_centroid_2024(X, population, lower, upper, K=1):
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