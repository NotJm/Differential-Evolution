import numpy as np
from .Problem import Problem, ProblemType


class CEC2010_C01(Problem):
    
    SUPERIOR = np.array([10] * 10)
    INFERIOR = np.array([0] * 10)

    def __init__(self):
        rest_g = [self.cec2010_c01_g1, self.cec2010_c01_g2]
        super().__init__(
            ProblemType.CONSTRAINED, 
            self.SUPERIOR, self.INFERIOR, 
            rest_g, []
        )
    
    def fitness(self, individuo: np.array) -> float:
        z = individuo - self.o
        f_x = -np.abs(np.sum(np.cos(z)**4) - 2 * np.prod(np.cos(z)**2))
        return f_x

    @staticmethod
    def cec2010_c01_g1(x):
        return 0.75 - np.prod(x)  # restriccion de desigualdad <= 0
    
    @staticmethod
    def cec2010_c01_g2(x):
        return np.sum(x) - 7.5 * len(x)  # restriccion de desigualdad <= 0

class CEC2010_C02(Problem):
    
    SUPERIOR = np.array([5.12] * 10)
    INFERIOR = np.array([-5.12] * 10)

    def __init__(self):
        rest_g = [self.cec2010_c02_g1, self.cec2010_c02_g2]
        super().__init__(
            ProblemType.CONSTRAINED, 
            self.SUPERIOR, self.INFERIOR, 
            rest_g, []
        )
    
    def fitness(self, individuo: np.array) -> float:
        z = individuo - self.o
        y = z - 0.5
        f_x = np.max(z)
        return f_x

    @staticmethod
    def cec2010_c02_g1(x):
        return 10 - (1/len(x)) * np.sum(x**2 - 10 * np.cos(2 * np.pi * x) + 10)  # restriccion de desigualdad <= 0
    
    @staticmethod
    def cec2010_c02_g2(x):
        return (1/len(x)) * np.sum(x**2 - 10 * np.cos(2 * np.pi * x) + 10) - 15  # restriccion de desigualdad <= 0

class CEC2010_C03(Problem):
    
    SUPERIOR = np.array([1000] * 10)
    INFERIOR = np.array([-1000] * 10)

    def __init__(self):
        rest_h = [self.cec2010_c03_h]
        super().__init__(
            ProblemType.CONSTRAINED, 
            self.SUPERIOR, self.INFERIOR, 
            [], rest_h
        )
    
    def fitness(self, individuo: np.array) -> float:
        z = individuo - self.o
        f_x = np.sum(100 * (z[:-1]**2 - z[1:])**2 + (z[:-1] - 1)**2)
        return f_x

    @staticmethod
    def cec2010_c03_h(x):
        return np.sum((x[:-1] - x[1:])**2)  # restriccion de igualdad = 0

class CEC2010_C04(Problem):
    
    SUPERIOR = np.array([50] * 10)
    INFERIOR = np.array([-50] * 10)

    def __init__(self):
        rest_h = [self.cec2010_c04_h1, self.cec2010_c04_h2, self.cec2010_c04_h3, self.cec2010_c04_h4]
        super().__init__(
            ProblemType.CONSTRAINED, 
            self.SUPERIOR, self.INFERIOR, 
            [], rest_h
        )
    
    def fitness(self, individuo: np.array) -> float:
        z = individuo - self.o
        f_x = np.max(z)
        return f_x

    @staticmethod
    def cec2010_c04_h1(x):
        return (1/len(x)) * np.sum(x * np.cos(np.sqrt(np.abs(x))))  # restriccion de igualdad = 0
    
    @staticmethod
    def cec2010_c04_h2(x):
        return np.sum((x[:-1] - x[1:])**2)  # restriccion de igualdad = 0

    @staticmethod
    def cec2010_c04_h3(x):
        return np.sum((x**2 - x[1:])**2)  # restriccion de igualdad = 0

    @staticmethod
    def cec2010_c04_h4(x):
        return np.sum(x)  # restriccion de igualdad = 0

class CEC2010_C05(Problem):
    
    SUPERIOR = np.array([600] * 10)
    INFERIOR = np.array([-600] * 10)

    def __init__(self):
        rest_h = [self.cec2010_c05_h1, self.cec2010_c05_h2]
        super().__init__(
            ProblemType.CONSTRAINED, 
            self.SUPERIOR, self.INFERIOR, 
            [], rest_h
        )
    
    def fitness(self, individuo: np.array) -> float:
        z = individuo - self.o
        f_x = np.max(z)
        return f_x

    @staticmethod
    def cec2010_c05_h1(x):
        return (1/len(x)) * np.sum(-x * np.sin(np.sqrt(np.abs(x))))  # restriccion de igualdad = 0
    
    @staticmethod
    def cec2010_c05_h2(x):
        return (1/len(x)) * np.sum(-x * np.cos(0.5 * np.sqrt(np.abs(x))))  # restriccion de igualdad = 0

class CEC2010_C06(Problem):
    
    SUPERIOR = np.array([600] * 10)
    INFERIOR = np.array([-600] * 10)

    def __init__(self):
        rest_h = [self.cec2010_c06_h1, self.cec2010_c06_h2]
        super().__init__(
            ProblemType.CONSTRAINED, 
            self.SUPERIOR, self.INFERIOR, 
            [], rest_h
        )
    
    def fitness(self, individuo: np.array) -> float:
        z = individuo - self.o
        y = (individuo + 483.6106156535 - self.o) @ self.M - 483.6106156535
        f_x = np.max(z)
        return f_x

    @staticmethod
    def cec2010_c06_h1(x):
        return (1/len(x)) * np.sum(-x * np.sin(np.sqrt(np.abs(x))))  # restriccion de igualdad = 0
    
    @staticmethod
    def cec2010_c06_h2(x):
        return (1/len(x)) * np.sum(-x * np.cos(0.5 * np.sqrt(np.abs(x))))  # restriccion de igualdad = 0

class CEC2010_C07(Problem):
    
    SUPERIOR = np.array([1400] * 10)
    INFERIOR = np.array([-1400] * 10)

    def __init__(self):
        rest_h = [self.cec2010_c07_h1]
        super().__init__(
            ProblemType.CONSTRAINED, 
            self.SUPERIOR, self.INFERIOR, 
            [], rest_h
        )
    
    def fitness(self, individuo: np.array) -> float:
        z = individuo - self.o
        f_x = np.max(z)
        return f_x

    @staticmethod
    def cec2010_c07_h1(x):
        return (1/len(x)) * np.sum(-x * np.sin(np.sqrt(np.abs(x))))  # restriccion de igualdad = 0

class CEC2010_C08(Problem):
    
    SUPERIOR = np.array([10] * 10)
    INFERIOR = np.array([-10] * 10)

    def __init__(self):
        rest_g = [self.cec2010_c08_g1]
        super().__init__(
            ProblemType.CONSTRAINED, 
            self.SUPERIOR, self.INFERIOR, 
            rest_g, []
        )
    
    def fitness(self, individuo: np.array) -> float:
        z = individuo - self.o
        f_x = np.max(z)
        return f_x

    @staticmethod
    def cec2010_c08_g1(x):
        return 10 - (1/len(x)) * np.sum(x**2 - 10 * np.cos(2 * np.pi * x) + 10)  # restriccion de desigualdad <= 0

class CEC2010_C09(Problem):
    
    SUPERIOR = np.array([500] * 10)
    INFERIOR = np.array([-500] * 10)

    def __init__(self):
        rest_g = [self.cec2010_c09_g1]
        super().__init__(
            ProblemType.CONSTRAINED, 
            self.SUPERIOR, self.INFERIOR, 
            rest_g, []
        )
    
    def fitness(self, individuo: np.array) -> float:
        z = individuo - self.o
        f_x = np.max(z)
        return f_x

    @staticmethod
    def cec2010_c09_g1(x):
        return 100 - (1/len(x)) * np.sum(x**2 - 10 * np.cos(2 * np.pi * x) + 10)  # restriccion de desigualdad <= 0

class CEC2010_C10(Problem):
    
    SUPERIOR = np.array([100] * 10)
    INFERIOR = np.array([-100] * 10)

    def __init__(self):
        rest_g = [self.cec2010_c10_g1]
        super().__init__(
            ProblemType.CONSTRAINED, 
            self.SUPERIOR, self.INFERIOR, 
            rest_g, []
        )
    
    def fitness(self, individuo: np.array) -> float:
        z = individuo - self.o
        f_x = np.max(z)
        return f_x

    @staticmethod
    def cec2010_c10_g1(x):
        return 50 - (1/len(x)) * np.sum(x**2 - 10 * np.cos(2 * np.pi * x) + 10)  # restriccion de desigualdad <= 0

class CEC2010_C11(Problem):
    
    SUPERIOR = np.array([600] * 10)
    INFERIOR = np.array([-600] * 10)

    def __init__(self):
        rest_g = [self.cec2010_c11_g1]
        super().__init__(
            ProblemType.CONSTRAINED, 
            self.SUPERIOR, self.INFERIOR, 
            rest_g, []
        )
    
    def fitness(self, individuo: np.array) -> float:
        z = individuo - self.o
        f_x = np.max(z)
        return f_x

    @staticmethod
    def cec2010_c11_g1(x):
        return 200 - (1/len(x)) * np.sum(x**2 - 10 * np.cos(2 * np.pi * x) + 10)  # restriccion de desigualdad <= 0

class CEC2010_C12(Problem):
    
    SUPERIOR = np.array([1000] * 10)
    INFERIOR = np.array([-1000] * 10)

    def __init__(self):
        rest_h = [self.cec2010_c12_h1]
        super().__init__(
            ProblemType.CONSTRAINED, 
            self.SUPERIOR, self.INFERIOR, 
            [], rest_h
        )
    
    def fitness(self, individuo: np.array) -> float:
        z = individuo - self.o
        f_x = np.max(z)
        return f_x

    @staticmethod
    def cec2010_c12_h1(x):
        return (1/len(x)) * np.sum(-x * np.sin(np.sqrt(np.abs(x))))  # restriccion de igualdad = 0

class CEC2010_C13(Problem):
    
    SUPERIOR = np.array([2000] * 10)
    INFERIOR = np.array([-2000] * 10)

    def __init__(self):
        rest_h = [self.cec2010_c13_h1]
        super().__init__(
            ProblemType.CONSTRAINED, 
            self.SUPERIOR, self.INFERIOR, 
            [], rest_h
        )
    
    def fitness(self, individuo: np.array) -> float:
        z = individuo - self.o
        f_x = np.max(z)
        return f_x

    @staticmethod
    def cec2010_c13_h1(x):
        return (1/len(x)) * np.sum(-x * np.sin(np.sqrt(np.abs(x))))  # restriccion de igualdad = 0

class CEC2010_C14(Problem):
    
    SUPERIOR = np.array([1500] * 10)
    INFERIOR = np.array([-1500] * 10)

    def __init__(self):
        rest_h = [self.cec2010_c14_h1]
        super().__init__(
            ProblemType.CONSTRAINED, 
            self.SUPERIOR, self.INFERIOR, 
            [], rest_h
        )
    
    def fitness(self, individuo: np.array) -> float:
        z = individuo - self.o
        f_x = np.max(z)
        return f_x

    @staticmethod
    def cec2010_c14_h1(x):
        return (1/len(x)) * np.sum(-x * np.sin(np.sqrt(np.abs(x))))  # restriccion de igualdad = 0

class CEC2010_C15(Problem):
    
    SUPERIOR = np.array([700] * 10)
    INFERIOR = np.array([-700] * 10)

    def __init__(self):
        rest_h = [self.cec2010_c15_h1]
        super().__init__(
            ProblemType.CONSTRAINED, 
            self.SUPERIOR, self.INFERIOR, 
            [], rest_h
        )
    
    def fitness(self, individuo: np.array) -> float:
        z = individuo - self.o
        f_x = np.max(z)
        return f_x

    @staticmethod
    def cec2010_c15_h1(x):
        return (1/len(x)) * np.sum(-x * np.sin(np.sqrt(np.abs(x))))  # restriccion de igualdad = 0

class CEC2010_C16(Problem):
    
    SUPERIOR = np.array([800] * 10)
    INFERIOR = np.array([-800] * 10)

    def __init__(self):
        rest_h = [self.cec2010_c16_h1]
        super().__init__(
            ProblemType.CONSTRAINED, 
            self.SUPERIOR, self.INFERIOR, 
            [], rest_h
        )
    
    def fitness(self, individuo: np.array) -> float:
        z = individuo - self.o
        f_x = np.max(z)
        return f_x

    @staticmethod
    def cec2010_c16_h1(x):
        return (1/len(x)) * np.sum(-x * np.sin(np.sqrt(np.abs(x))))  # restriccion de igualdad = 0

class CEC2010_C17(Problem):
    
    SUPERIOR = np.array([500] * 10)
    INFERIOR = np.array([-500] * 10)

    def __init__(self):
        rest_h = [self.cec2010_c17_h1]
        super().__init__(
            ProblemType.CONSTRAINED, 
            self.SUPERIOR, self.INFERIOR, 
            [], rest_h
        )
    
    def fitness(self, individuo: np.array) -> float:
        z = individuo - self.o
        f_x = np.max(z)
        return f_x

    @staticmethod
    def cec2010_c17_h1(x):
        return (1/len(x)) * np.sum(-x * np.sin(np.sqrt(np.abs(x))))  # restriccion de igualdad = 0

class CEC2010_C18(Problem):
    
    SUPERIOR = np.array([1000] * 10)
    INFERIOR = np.array([-1000] * 10)

    def __init__(self):
        rest_h = [self.cec2010_c18_h1]
        super().__init__(
            ProblemType.CONSTRAINED, 
            self.SUPERIOR, self.INFERIOR, 
            [], rest_h
        )
    
    def fitness(self, individuo: np.array) -> float:
        z = individuo - self.o
        f_x = np.max(z)
        return f_x

    @staticmethod
    def cec2010_c18_h1(x):
        return (1/len(x)) * np.sum(-x * np.sin(np.sqrt(np.abs(x))))  # restriccion de igualdad = 0
