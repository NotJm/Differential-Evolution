import numpy as np
from .Problem import Problem, ProblemType
from scipy.stats import ortho_group

D = 10

o = np.zeros(D)

class CEC2010_C01(Problem):
    
    SUPERIOR = np.array([10] * D)
    INFERIOR = np.array([0] * D)

    def __init__(self):
        rest_g = [self.cec2010_c01_g1, self.cec2010_c01_g2]
        super().__init__(
            ProblemType.CONSTRAINED, 
            self.SUPERIOR, self.INFERIOR, 
            rest_g, []
        )
    
    def fitness(self, individuo: np.array) -> float:
        z = individuo - o
        f_x = -np.abs((np.sum(np.cos(z)**4) - 2 * np.prod(np.cos(z)**2)) / np.sqrt(np.sum(np.arange(1, D+1) * z**2)))
        return f_x

    @staticmethod
    def cec2010_c01_g1(x):
        z = x - o
        return 0.75 - np.prod(z)  # restriccion de desigualdad <= 0
    
    @staticmethod
    def cec2010_c01_g2(x):
        z = x - o
        return np.sum(z) - 7.5 * len(z)  # restriccion de desigualdad <= 0

class CEC2010_C02(Problem):
    
    SUPERIOR = np.array([5.12] * D)
    INFERIOR = np.array([-5.12] * D)

    def __init__(self):
        rest_g = [self.cec2010_c02_g1, self.cec2010_c02_g2]
        rest_h = [self.cec2010_c02_h1]
        super().__init__(
            ProblemType.CONSTRAINED, 
            self.SUPERIOR, self.INFERIOR, 
            rest_g, rest_h
        )
    
    def fitness(self, individuo: np.array) -> float:
        z = individuo - o
        f_x = np.max(z)
        return f_x

    @staticmethod
    def cec2010_c02_g1(x):
        z = x - o
        return 10 - (1/D) * np.sum(z**2 - 10 * np.cos(2 * np.pi * z) + 10)  # restriccion de desigualdad <= 0
    
    @staticmethod
    def cec2010_c02_g2(x): 
        z = x - o
        return (1/len(z)) * np.sum(z**2 - 10 * np.cos(2 * np.pi * z) + 10) - 15  # restriccion de desigualdad <= 0
    
    @staticmethod
    def cec2010_c02_h1(x):
        z = x - o
        y = z - 0.5
        return (1/len(y)) * np.sum(y**2 - 10 * np.cos(2 * np.pi * y) + 10) - 20  # restriccion de desigualdad <= 0

class CEC2010_C03(Problem):
    
    SUPERIOR = np.array([1000] * D)
    INFERIOR = np.array([-1000] * D)

    def __init__(self):
        rest_h = [self.cec2010_c03_h]
        super().__init__(
            ProblemType.CONSTRAINED, 
            self.SUPERIOR, self.INFERIOR, 
            [], rest_h
        )
    
    def fitness(self, individuo: np.array) -> float:
        z = individuo - o
        f_x = np.sum(100 * (z[:-1]**2 - z[1:])**2 + (z[:-1] - 1)**2)
        return f_x

    @staticmethod
    def cec2010_c03_h(x):
        z = x - o
        return np.sum((z[:-1] - z[1:])**2)  # restriccion de igualdad = 0

class CEC2010_C04(Problem):
    
    SUPERIOR = np.array([50] * D)
    INFERIOR = np.array([-50] * D)

    def __init__(self):
        rest_h = [self.cec2010_c04_h1, self.cec2010_c04_h2, self.cec2010_c04_h3, self.cec2010_c04_h4]
        super().__init__(
            ProblemType.CONSTRAINED, 
            self.SUPERIOR, self.INFERIOR, 
            [], rest_h
        )
    
    def fitness(self, individuo: np.array) -> float:
        z = individuo - o
        f_x = np.max(z)
        return f_x

    @staticmethod
    def cec2010_c04_h1(x):
        z = x - o
        return (1/D) * np.sum(z * np.cos(np.sqrt(np.abs(z))))  # restriccion de igualdad = 0
    
    @staticmethod
    def cec2010_c04_h2(x):
        z = x - o
        return np.sum((z[:D//2-1] - z[1:D//2])**2)  # restriccion de igualdad = 0

    @staticmethod
    def cec2010_c04_h3(x):
        z = x - o
        return np.sum((z[D//2:D-1]**2 - z[D//2+1:]**2)**2)  # restriccion de igualdad = 0

    @staticmethod
    def cec2010_c04_h4(x):
        z = x - o
        return np.sum(z)  # restriccion de igualdad = 0


class CEC2010_C05(Problem):
    
    SUPERIOR = np.array([600] * D)
    INFERIOR = np.array([-600] * D)

    def __init__(self):
        rest_h = [self.cec2010_c05_h1, self.cec2010_c05_h2]
        super().__init__(
            ProblemType.CONSTRAINED, 
            self.SUPERIOR, self.INFERIOR, 
            [], rest_h
        )
    
    def fitness(self, individuo: np.array) -> float:
        z = individuo - o
        f_x = np.max(z)
        return f_x

    @staticmethod
    def cec2010_c05_h1(x):
        z = x - o
        return (1/D) * np.sum(-z * np.sin(np.sqrt(np.abs(z))))  # restriccion de igualdad = 0
    
    @staticmethod
    def cec2010_c05_h2(x):
        z = x - o
        return (1/D) * np.sum(-z * np.cos(0.5 * np.sqrt(np.abs(z))))  # restriccion de igualdad = 0


class CEC2010_C06(Problem):
    
    D = 10
    SUPERIOR = np.array([600] * D)
    INFERIOR = np.array([-600] * D)
    
    M = ortho_group.rvs(D)

    def __init__(self):
        rest_h = [self.cec2010_c06_h1, self.cec2010_c06_h2]
        super().__init__(
            ProblemType.CONSTRAINED, 
            self.SUPERIOR, self.INFERIOR, 
            [], rest_h
        )
    
    def fitness(self, individuo: np.array) -> float:
        z = individuo - o
        f_x = np.max(z)
        return f_x

    @staticmethod
    def cec2010_c06_h1(x):
        y = (x + 483.6106156535 - o) @ CEC2010_C06.M - 483.6106156535
        y = np.clip(y, CEC2010_C06.INFERIOR, CEC2010_C06.SUPERIOR)
        return (1/CEC2010_C06.D) * np.sum(-y * np.sin(np.sqrt(np.abs(y))))  # restricción de igualdad = 0
    
    @staticmethod
    def cec2010_c06_h2(x):
        y = (x + 483.6106156535 - o) @ CEC2010_C06.M - 483.6106156535
        y = np.clip(y, CEC2010_C06.INFERIOR, CEC2010_C06.SUPERIOR)
        return (1/CEC2010_C06.D) * np.sum(-y * np.cos(0.5 * np.sqrt(np.abs(y))))  # restricción de igualdad = 0

class CEC2010_C07(Problem):
    
    SUPERIOR = np.array([140] * D)
    INFERIOR = np.array([-140] * D)

    def __init__(self):
        rest_g = [self.cec2010_c07_g1]
        super().__init__(
            ProblemType.CONSTRAINED, 
            self.SUPERIOR, self.INFERIOR, 
            rest_g, []
        )
    
    def fitness(self, individuo: np.array) -> float:
        z = individuo + 1 - o
        f_x = np.sum(100 * (z[:-1]**2 - z[1:]**2)**2 + (z[:-1] - 1)**2)
        return f_x

    @staticmethod
    def cec2010_c07_g1(x):
        y = x - o
        return 0.5 - np.exp(-0.1 * np.sqrt((1/D) * np.sum(y**2))) - 3 * np.exp((1/D) * np.sum(np.cos(0.1 * y))) + np.exp(1)  # restriccion de desigualdad <= 0


class CEC2010_C08(Problem):
    
    SUPERIOR = np.array([140] * D)
    INFERIOR = np.array([-140] * D)
    
    M = ortho_group.rvs(D)

    def __init__(self):
        rest_g = [self.cec2010_c08_g1]
        super().__init__(
            ProblemType.CONSTRAINED, 
            self.SUPERIOR, self.INFERIOR, 
            rest_g, []
        )
    
    def fitness(self, individuo: np.array) -> float:
        z = individuo + 1 - o
        f_x = np.sum(100 * (z[:-1]**2 - z[1:]**2)**2 + (z[:-1] - 1)**2)
        return f_x

    @staticmethod
    def cec2010_c08_g1(x):
        y = (x - o) @ CEC2010_C08.M
        y = np.clip(y, CEC2010_C08.INFERIOR, CEC2010_C08.SUPERIOR)
        return 0.5 - np.exp(-0.1 * np.sqrt((1/D) * np.sum(y**2))) - 3 * np.exp((1/D) * np.sum(np.cos(0.1 * y))) + np.exp(1)  # restriccion de desigualdad <= 0

class CEC2010_C09(Problem):
    
    SUPERIOR = np.array([500] * D)
    INFERIOR = np.array([-500] * D)

    def __init__(self):
        rest_h = [self.cec2010_c09_h1]
        super().__init__(
            ProblemType.CONSTRAINED, 
            self.SUPERIOR, self.INFERIOR, 
            [], rest_h
        )
    
    def fitness(self, individuo: np.array) -> float:
        z = individuo + 1 - o
        f_x = np.sum(100 * (z[:-1]**2 - z[1:]**2)**2 + (z[:-1] - 1)**2)
        return f_x

    @staticmethod
    def cec2010_c09_h1(x):
        y = x - o
        return np.sum(y * np.sin(np.sqrt(np.abs(y)))) # restriccion de igualdad = 0


class CEC2010_C10(Problem):
    
    SUPERIOR = np.array([500] * D)
    INFERIOR = np.array([-500] * D)
    
    M = ortho_group.rvs(D)

    def __init__(self):
        rest_h = [self.cec2010_c10_h1]
        super().__init__(
            ProblemType.CONSTRAINED, 
            self.SUPERIOR, self.INFERIOR, 
            [], rest_h
        )
    
    def fitness(self, individuo: np.array) -> float:
        z = individuo + 1 - o
        f_x = np.sum(100 * (z[:-1]**2 - z[1:]**2)**2 + (z[:-1] - 1)**2)
        return f_x

    @staticmethod
    def cec2010_c10_h1(x):
        y = (x - o) @ CEC2010_C10.M
        y = np.clip(y, CEC2010_C10.INFERIOR, CEC2010_C10.SUPERIOR)
        return np.sum(y * np.sin(np.sqrt(np.abs(y)))) # restriccion de igualdad = 0


class CEC2010_C11(Problem):
    
    SUPERIOR = np.array([100] * D)
    INFERIOR = np.array([-100] * D)
    
    M = ortho_group.rvs(D)

    def __init__(self):
        rest_h = [self.cec2010_c11_h1]
        rest_g = []
        super().__init__(
            ProblemType.CONSTRAINED, 
            self.SUPERIOR, self.INFERIOR, 
            rest_g, rest_h
        )
    
    def fitness(self, individuo: np.array) -> float:
        z = (individuo - o) @ self.M
        z = np.clip(z, self.INFERIOR, self.SUPERIOR)
        f_x = (1/D) * np.sum(-z * np.cos(2 * np.sqrt(np.abs(z))))
        return f_x

    @staticmethod
    def cec2010_c11_h1(x):
        y = x + 1 - o
        return np.sum(100 * (y[:-1]**2 - y[1:]**2)**2 + (y[:-1] - 1)**2) # restriccion de igualdad = 0

class CEC2010_C12(Problem):
    
    SUPERIOR = np.array([1000] * D)
    INFERIOR = np.array([-1000] * D)

    def __init__(self):
        rest_h = [self.cec2010_c12_h1]
        rest_g = [self.cec2010_c12_g1]
        super().__init__(
            ProblemType.CONSTRAINED, 
            self.SUPERIOR, self.INFERIOR, 
            rest_g, rest_h
        )
    
    def fitness(self, individuo: np.array) -> float:
        z = individuo - o
        f_x = (1/D) * np.sum(z * np.sin(np.sqrt(np.abs(z))))
        return f_x

    @staticmethod
    def cec2010_c12_h1(x):
        z = x - o
        return np.sum((z[:-1]**2 - z[1:]**2)**2)  # restriccion de igualdad = 0

    @staticmethod
    def cec2010_c12_g1(x):
        z = x - o
        return np.sum(z - 100 * np.cos(0.1 * z) + 10)  # restriccion de desigualdad <= 0


class CEC2010_C13(Problem):
    
    SUPERIOR = np.array([500] * D)
    INFERIOR = np.array([-500] * D)

    def __init__(self):
        rest_g = [self.cec2010_c13_g1, self.cec2010_c13_g2, self.cec2010_c13_g3]
        super().__init__(
            ProblemType.CONSTRAINED, 
            self.SUPERIOR, self.INFERIOR, 
            rest_g, []
        )
    
    def fitness(self, individuo: np.array) -> float:
        z = individuo - o
        f_x = (1/D) * np.sum(-z * np.sin(np.sqrt(np.abs(z))))
        return f_x

    @staticmethod
    def cec2010_c13_g1(x):
        z = x - o
        return -50 + (1/(100*D)) * np.sum(z**2)  # restriccion de desigualdad <= 0

    @staticmethod
    def cec2010_c13_g2(x):
        z = x - o
        return (50/D) * np.sum(np.sin((1/50) * np.pi * z))  # restriccion de desigualdad <= 0

    @staticmethod
    def cec2010_c13_g3(x):
        z = x - o
        return 75 - 50 * ((1/D) * np.sum(z**2/4000) - np.prod(np.cos(z/np.sqrt(np.arange(1, D+1)))) + 1)  # restriccion de desigualdad <= 0


class CEC2010_C14(Problem):
    
    SUPERIOR = np.array([1000] * D)
    INFERIOR = np.array([-1000] * D)

    def __init__(self):
        rest_h = []
        rest_g = [self.cec2010_c14_g1, self.cec2010_c14_g2, self.cec2010_c14_g3]
        super().__init__(
            ProblemType.CONSTRAINED, 
            self.SUPERIOR, self.INFERIOR, 
            rest_g, rest_h
        )
    
    def fitness(self, individuo: np.array) -> float:
        z = individuo + 1 - o
        f_x = np.sum(100 * (z[:-1]**2 - z[1:]**2)**2 + (z[:-1] - 1)**2)
        return f_x

    @staticmethod
    def cec2010_c14_g1(x):
        y = x - o
        return np.sum(-y * np.cos(np.sqrt(np.abs(y)))) - D  # restriccion de desigualdad <= 0
    
    @staticmethod
    def cec2010_c14_g2(x):
        y = x - o
        return np.sum(y * np.cos(np.sqrt(np.abs(y)))) - D  # restriccion de desigualdad <= 0
    
    @staticmethod
    def cec2010_c14_g3(x):
        y = x - o
        return np.sum(y * np.sin(np.sqrt(np.abs(y)))) - 10 * D  # restriccion de desigualdad <= 0


class CEC2010_C15(Problem):
    
    SUPERIOR = np.array([1000] * D)
    INFERIOR = np.array([-1000] * D)
    
    M = ortho_group.rvs(D)

    def __init__(self):
        rest_h = []
        rest_g = [self.cec2010_c15_g1, self.cec2010_c15_g2, self.cec2010_c15_g3]
        super().__init__(
            ProblemType.CONSTRAINED, 
            self.SUPERIOR, self.INFERIOR, 
            rest_g, rest_h
        )
    
    def fitness(self, individuo: np.array) -> float:
        z = individuo - o
        f_x = np.max(z)
        return f_x

    @staticmethod
    def cec2010_c15_g1(x):
        y = (x - o) @ CEC2010_C15.M
        y = np.clip(y, CEC2010_C15.INFERIOR, CEC2010_C15.SUPERIOR)
        return np.sum(-y * np.cos(np.sqrt(np.abs(y)))) - D  # restriccion de desigualdad <= 0
    
    @staticmethod
    def cec2010_c15_g2(x):
        y = (x - o) @ CEC2010_C15.M
        y = np.clip(y, CEC2010_C15.INFERIOR, CEC2010_C15.SUPERIOR)
        return np.sum(y * np.cos(np.sqrt(np.abs(y)))) - D  # restriccion de desigualdad <= 0
    
    @staticmethod
    def cec2010_c15_g3(x):
        y = (x - o) @ CEC2010_C15.M
        y = np.clip(y, CEC2010_C15.INFERIOR, CEC2010_C15.SUPERIOR)
        return np.sum(y * np.sin(np.sqrt(np.abs(y)))) - 10 * D  # restriccion de desigualdad <= 0

class CEC2010_C16(Problem):
    
    SUPERIOR = np.array([10] * D)
    INFERIOR = np.array([-10] * D)

    def __init__(self):
        rest_g = [self.cec2010_c16_g1, self.cec2010_c16_g2]
        rest_h = [self.cec2010_c16_h1, self.cec2010_c16_h2]
        super().__init__(
            ProblemType.CONSTRAINED, 
            self.SUPERIOR, self.INFERIOR, 
            rest_g, rest_h
        )
    
    def fitness(self, individuo: np.array) -> float:
        z = individuo - o
        f_x = np.sum(z**2 / 4000) - np.prod(np.cos(z / np.sqrt(np.arange(1, D+1)))) + 1
        return f_x

    @staticmethod
    def cec2010_c16_g1(x):
        z = x - o
        return np.sum(z**2 - 100 * np.cos(np.pi * z) + 10)  # restriccion de desigualdad <= 0
    
    @staticmethod
    def cec2010_c16_g2(x):
        z = x - o
        return np.prod(z)  # restriccion de desigualdad <= 0

    @staticmethod
    def cec2010_c16_h1(x):
        z = x - o
        return np.sum(z * np.sin(np.sqrt(np.abs(z))))  # restriccion de igualdad = 0

    @staticmethod
    def cec2010_c16_h2(x):
        z = x - o
        return np.sum(-z * np.sin(np.sqrt(np.abs(z))))  # restriccion de igualdad = 0

class CEC2010_C17(Problem):
    
    SUPERIOR = np.array([10] * D)
    INFERIOR = np.array([-10] * D)

    def __init__(self):
        rest_g = [self.cec2010_c17_g1, self.cec2010_c17_g2]
        rest_h = [self.cec2010_c17_h1]
        super().__init__(
            ProblemType.CONSTRAINED, 
            self.SUPERIOR, self.INFERIOR, 
            rest_g, rest_h
        )
    
    def fitness(self, individuo: np.array) -> float:
        z = individuo - o
        f_x = np.sum((z[:-1] - z[1:])**2)
        return f_x

    @staticmethod
    def cec2010_c17_g1(x):
        z = x - o
        return np.prod(z)  # restriccion de desigualdad <= 0
    
    @staticmethod
    def cec2010_c17_g2(x):
        z = x - o
        return np.sum(z)  # restriccion de desigualdad <= 0

    @staticmethod
    def cec2010_c17_h1(x):
        z = x - o
        return np.sum(z * np.sin(4 * np.sqrt(np.abs(z))))  # restriccion de igualdad = 0

class CEC2010_C18(Problem):
    
    SUPERIOR = np.array([50] * D)
    INFERIOR = np.array([-50] * D)

    def __init__(self):
        rest_g = [self.cec2010_c18_g1]
        rest_h = [self.cec2010_c18_h1]
        super().__init__(
            ProblemType.CONSTRAINED, 
            self.SUPERIOR, self.INFERIOR, 
            rest_g, rest_h
        )
    
    def fitness(self, individuo: np.array) -> float:
        z = individuo - o
        f_x = np.sum((z[:-1] - z[1:])**2)
        return f_x

    @staticmethod
    def cec2010_c18_g1(x):
        z = x - o
        return (1/D) * np.sum(-z * np.sin(np.sqrt(np.abs(z))))  # restriccion de desigualdad <= 0
    
    @staticmethod
    def cec2010_c18_h1(x):
        z = x - o
        return (1/D) * np.sum(z * np.sin(np.sqrt(np.abs(z))))  # restriccion de igualdad = 0
