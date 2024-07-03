import numpy as np

class BCHM:
    
    # GANDOMI 2012
    @staticmethod
    def evolutionary(z, lb, ub, x_b):
        # Inicializar el vector ajustado como una copia del vector de solución actual
        adjusted_z = np.copy(z)

        # Generar valores aleatorios entre 0 y 1
        a = np.random.rand(len(z))
        b = np.random.rand(len(z))

        # Ajustar los componentes que están por debajo del límite inferior
        below_lb = z < lb
        adjusted_z[below_lb] = a[below_lb] * lb[below_lb] + (1 - a[below_lb]) * x_b[below_lb]

        # Ajustar los componentes que están por encima del límite superior
        above_ub = z > ub
        adjusted_z[above_ub] = b[above_ub] * ub[above_ub] + (1 - b[above_ub]) * x_b[above_ub]

        return adjusted_z
    
    
    # ALEJANDRO 2024
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