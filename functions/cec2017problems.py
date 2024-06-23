import numpy as np
from .Problem import Problem, ProblemType
from scipy.stats import ortho_group

D = 10

SUPERIOR_1 = np.array([100] * D)
INFERIOR_1 = np.array([-100] * D)

SUPERIOR_2 = np.array([10] * D)
INFERIOR_2 = np.array([-10] * D)

SUPERIOR_3 = np.array([20] * D)
INFERIOR_3 = np.array([-20] * D)

SUPERIOR_4 = np.array([50] * D)
INFERIOR_4 = np.array([-50] * D)

o = np.zeros(D)

class CEC2017_C01(Problem):
    def __init__(self):
        rest_h = []
        rest_g = [self.CEC2017_C01_g1]
        super().__init__(ProblemType.CONSTRAINED, SUPERIOR_1, INFERIOR_1, rest_g, rest_h)

    def fitness(self, x: np.array) -> float:
        z = x - o
        f_x = np.sum([(np.sum(z[:i+1]))**2 for i in range(D)])
        return f_x
    
    @staticmethod
    def CEC2017_C01_g1(x):
        z = x - o
        return np.sum(z**2 - 5000 * np.cos(0.1 * np.pi * z) - 4000)

class CEC2017_C02(Problem):
    
    M = ortho_group.rvs(D)
    
    def __init__(self):
        rest_h = []
        rest_g = [self.CEC2017_C02_g1]
        super().__init__(ProblemType.CONSTRAINED, SUPERIOR_1, INFERIOR_1, rest_g, rest_h)

    def fitness(self, x: np.array) -> float:
        z = x - o
        f_x = np.sum([(np.sum(z[:i+1]))**2 for i in range(D)])
        return f_x
    
    @staticmethod
    def CEC2017_C02_g1(x):
        z = x - o
        y = CEC2017_C02.M @ z
        return np.sum(y**2 - 5000 * np.cos(0.1 * np.pi * y) - 4000)

class CEC2017_C03(Problem):
    def __init__(self):
        rest_h = [self.CEC2017_C03_h1]
        rest_g = [self.CEC2017_C03_g1]
        super().__init__(ProblemType.CONSTRAINED, SUPERIOR_1, INFERIOR_1, rest_g, rest_h)

    def fitness(self, x: np.array) -> float:
        z = x - o
        f_x = np.sum([(np.sum(z[:i+1]))**2 for i in range(D)])
        return f_x
    
    @staticmethod
    def CEC2017_C03_g1(x):
        z = x - o
        return np.sum(z**2 - 5000 * np.cos(0.1 * np.pi * z) - 4000)
    
    @staticmethod
    def CEC2017_C03_h1(x):
        z = x - o
        return np.sum(z * np.sin(0.1 * np.pi * z))

class CEC2017_C04(Problem):
    def __init__(self):
        rest_h = []
        rest_g = [self.CEC2017_C04_g1, self.CEC2017_C04_g2]
        super().__init__(ProblemType.CONSTRAINED, SUPERIOR_2, INFERIOR_2, rest_g, rest_h)

    def fitness(self, x: np.array) -> float:
        z = x - o
        f_x = np.sum(z**2 - 10 * np.cos(2 * np.pi * z) + 10)
        return f_x
    
    @staticmethod
    def CEC2017_C04_g1(x):
        z = x - o
        return - np.sum(z * np.sin(2 * z))
    
    @staticmethod
    def CEC2017_C04_g2(x):
        z = x - o
        return np.sum(z * np.sin(z))

class CEC2017_C05(Problem):
    M1 = ortho_group.rvs(D)
    M2 = ortho_group.rvs(D)
    
    def __init__(self):
        rest_h = []
        rest_g = [self.CEC2017_C05_g1, self.CEC2017_C05_g2]
        super().__init__(ProblemType.CONSTRAINED, SUPERIOR_2, INFERIOR_2, rest_g, rest_h)

    def fitness(self, x: np.array) -> float:
        z = x - o
        f_x = np.sum([100 * (z[i]**2 - z[i+1])**2 + (z[i] - 1)**2 for i in range(D-1)])
        return f_x
    
    @staticmethod
    def CEC2017_C05_g1(x):
        z = x - o
        y = CEC2017_C05.M1 @ z
        return np.sum(y**2 - 50 * np.cos(2 * np.pi * y) - 40)
    
    @staticmethod
    def CEC2017_C05_g2(x):
        z = x - o
        w = CEC2017_C05.M2 @ z
        return np.sum(w**2 - 50 * np.cos(2 * np.pi * w) - 40)
    
class CEC2017_C06(Problem):
    def __init__(self):
        rest_h = [
            self.CEC2017_C06_h1, self.CEC2017_C06_h2, 
            self.CEC2017_C06_h3, self.CEC2017_C06_h4, 
            self.CEC2017_C06_h5, self.CEC2017_C06_h6
        ]
        rest_g = []
        super().__init__(ProblemType.CONSTRAINED, SUPERIOR_3, INFERIOR_3, rest_g, rest_h)

    def fitness(self, x: np.array) -> float:
        z = x - o
        f_x = np.sum(z**2 - 10 * np.cos(2 * np.pi * z) + 10)
        return f_x
    
    @staticmethod
    def CEC2017_C06_h1(x):
        z = x - o
        return - np.sum(z * np.sin(z))
    
    @staticmethod
    def CEC2017_C06_h2(x):
        z = x - o
        return np.sum(z * np.sin(np.pi * z))
    
    @staticmethod
    def CEC2017_C06_h3(x):
        z = x - o
        return - np.sum(z * np.cos(z))
    
    @staticmethod
    def CEC2017_C06_h4(x):
        z = x - o
        return np.sum(z * np.cos(np.pi * z))
    
    @staticmethod
    def CEC2017_C06_h5(x):
        z = x - o
        return np.sum(z * np.sin(2 * np.sqrt(np.abs(z))))
    
    @staticmethod
    def CEC2017_C06_h6(x):
        z = x - o
        return - np.sum(z * np.sin(2 * np.sqrt(np.abs(z))))
    
class CEC2017_C07(Problem):
    def __init__(self):
        rest_h = [self.CEC2017_C07_h1, self.CEC2017_C07_h2]
        rest_g = []
        super().__init__(ProblemType.CONSTRAINED, SUPERIOR_4, INFERIOR_4, rest_g, rest_h)

    def fitness(self, x: np.array) -> float:
        z = x - o
        f_x = np.sum(z * np.sin(z))
        return f_x
    
    @staticmethod
    def CEC2017_C07_h1(x):
        z = x - o
        return np.sum(z - 100 * np.cos(0.5 * z) + 100)
    
    @staticmethod
    def CEC2017_C07_h2(x):
        z = x - o
        return - np.sum(z - 100 * np.cos(0.5 * z) + 100)
    
class CEC2017_C08(Problem):
    def __init__(self):
        rest_h = [self.CEC2017_C08_h1, self.CEC2017_C08_h2]
        rest_g = []
        super().__init__(ProblemType.CONSTRAINED, SUPERIOR_1, INFERIOR_1, rest_g, rest_h)

    def fitness(self, x: np.array) -> float:
        z = x - o
        f_x = np.max(z)
        return f_x
    
    @staticmethod
    def CEC2017_C08_h1(x):
        z = x - o
        y = z[::2]  # y_l = z_{2l-1} for l = 1, ..., D/2
        return np.sum([(np.sum(y[:i+1]))**2 for i in range(len(y))])
    
    @staticmethod
    def CEC2017_C08_h2(x):
        z = x - o
        w = z[1::2]  # w_l = z_{2l} for l = 1, ..., D/2
        return np.sum([(np.sum(w[:i+1]))**2 for i in range(len(w))])
    
class CEC2017_C09(Problem):
    def __init__(self):
        rest_h = [self.CEC2017_C09_h1]
        rest_g = [self.CEC2017_C09_g1]
        super().__init__(ProblemType.CONSTRAINED, SUPERIOR_2, INFERIOR_2, rest_g, rest_h)

    def fitness(self, x: np.array) -> float:
        z = x - o
        f_x = np.max(z)
        return f_x
    
    @staticmethod
    def CEC2017_C09_g1(x):
        z = x - o
        w = z[1::2]  # w_l = z_{2l} for l = 1, ..., D/2
        return np.prod(w)
    
    @staticmethod
    def CEC2017_C09_h1(x):
        z = x - o
        y = z[::2]  # y_l = z_{2l-1} for l = 1, ..., D/2
        return np.sum([(y[i]**2 - y[i+1]**2)**2 for i in range(len(y)-1)])

class CEC2017_C10(Problem):
    def __init__(self):
        rest_h = [self.CEC2017_C10_h1, self.CEC2017_C10_h2]
        rest_g = []
        super().__init__(ProblemType.CONSTRAINED, SUPERIOR_1, INFERIOR_1, rest_g, rest_h)

    def fitness(self, x: np.array) -> float:
        z = x - o
        f_x = np.max(z)
        return f_x
    
    @staticmethod
    def CEC2017_C10_h1(x):
        z = x - o
        return np.sum([(np.sum(z[:i+1]))**2 for i in range(D)])
    
    @staticmethod
    def CEC2017_C10_h2(x):
        z = x - o
        return np.sum([(z[i] - z[i+1])**2 for i in range(D-1)])
    
class CEC2017_C11(Problem):
    def __init__(self):
        rest_h = [self.CEC2017_C11_h1]
        rest_g = [self.CEC2017_C11_g1]
        super().__init__(ProblemType.CONSTRAINED, SUPERIOR_1, INFERIOR_1, rest_g, rest_h)

    def fitness(self, x: np.array) -> float:
        z = x - o
        f_x = np.sum(z)
        return f_x
    
    @staticmethod
    def CEC2017_C11_g1(x):
        z = x - o
        return np.prod(z)
    
    @staticmethod
    def CEC2017_C11_h1(x):
        z = x - o
        return np.sum([(z[i] - z[i+1])**2 for i in range(D-1)])