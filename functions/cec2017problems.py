import numpy as np
from .Problem import Problem, ProblemType
from scipy.stats import ortho_group

D = 10

o = np.zeros(D)

class CEC2017_C01(Problem):
    
    SUPERIOR = np.array([100] * D)
    INFERIOR = np.array([-100] * D)
    
    def __init__(self):
        
        rest_h = []
        rest_g = [self.CEC2017_C01_g1]
        super().__init__(ProblemType.CONSTRAINED, self.SUPERIOR, self.INFERIOR, rest_g, rest_h)

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
    
    SUPERIOR = np.array([100] * D)
    INFERIOR = np.array([-100] * D)
    
    def __init__(self):
        rest_h = []
        rest_g = [self.CEC2017_C02_g1]
        super().__init__(ProblemType.CONSTRAINED, self.SUPERIOR, self.INFERIOR, rest_g, rest_h)

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
    
    SUPERIOR = np.array([100] * D)
    INFERIOR = np.array([-100] * D)
    
    def __init__(self):
        rest_h = [self.CEC2017_C03_h1]
        rest_g = [self.CEC2017_C03_g1]
        super().__init__(ProblemType.CONSTRAINED, self.SUPERIOR, self.INFERIOR, rest_g, rest_h)

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
    
    SUPERIOR = np.array([10] * D)
    INFERIOR = np.array([-10] * D)
    
    def __init__(self):
        rest_h = []
        rest_g = [self.CEC2017_C04_g1, self.CEC2017_C04_g2]
        super().__init__(ProblemType.CONSTRAINED, self.SUPERIOR, self.INFERIOR, rest_g, rest_h)

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
    
    SUPERIOR = np.array([10] * D)
    INFERIOR = np.array([-10] * D)
    
    def __init__(self):
        rest_h = []
        rest_g = [self.CEC2017_C05_g1, self.CEC2017_C05_g2]
        super().__init__(ProblemType.CONSTRAINED, self.SUPERIOR, self.INFERIOR, rest_g, rest_h)

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
    
    SUPERIOR = np.array([20] * D)
    INFERIOR = np.array([-20] * D)
    
    def __init__(self):
        rest_h = [
            self.CEC2017_C06_h1, self.CEC2017_C06_h2, self.CEC2017_C06_h3, self.CEC2017_C06_h4, 
            self.CEC2017_C06_h5, self.CEC2017_C06_h6
        ]
        rest_g = []
        super().__init__(ProblemType.CONSTRAINED, self.SUPERIOR, self.INFERIOR, rest_g, rest_h)

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
    
    SUPERIOR = np.array([50] * D)
    INFERIOR = np.array([-50] * D)
    
    def __init__(self):
        rest_h = [self.CEC2017_C07_h1, self.CEC2017_C07_h2]
        rest_g = []
        super().__init__(ProblemType.CONSTRAINED, self.SUPERIOR, self.INFERIOR, rest_g, rest_h)

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
    
    SUPERIOR = np.array([100] * D)
    INFERIOR = np.array([-100] * D)
    
    def __init__(self):
        rest_h = [self.CEC2017_C08_h1, self.CEC2017_C08_h2]
        rest_g = []
        super().__init__(ProblemType.CONSTRAINED, self.SUPERIOR, self.INFERIOR, rest_g, rest_h)

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
    
    SUPERIOR = np.array([10] * D)
    INFERIOR = np.array([-10] * D)
    
    def __init__(self):
        rest_h = [self.CEC2017_C09_h1]
        rest_g = [self.CEC2017_C09_g1]
        super().__init__(ProblemType.CONSTRAINED, self.SUPERIOR, self.INFERIOR, rest_g, rest_h)

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
    
    SUPERIOR = np.array([100] * D)
    INFERIOR = np.array([-100] * D)
    
    def __init__(self):
        rest_h = [self.CEC2017_C10_h1, self.CEC2017_C10_h2]
        rest_g = []
        super().__init__(ProblemType.CONSTRAINED, self.SUPERIOR, self.INFERIOR, rest_g, rest_h)

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
    
    SUPERIOR = np.array([100] * D)
    INFERIOR = np.array([-100] * D)
    
    def __init__(self):
        rest_h = [self.CEC2017_C11_h1]
        rest_g = [self.CEC2017_C11_g1]
        super().__init__(ProblemType.CONSTRAINED, self.SUPERIOR, self.INFERIOR, rest_g, rest_h)

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
    
class CEC2017_C12(Problem):
    
    SUPERIOR = np.array([100] * D)
    INFERIOR = np.array([-100] * D)
    
    def __init__(self):
        rest_h = [self.CEC2017_C12_h1]
        rest_g = [self.CEC2017_C12_g1]
        super().__init__(ProblemType.CONSTRAINED, self.SUPERIOR, self.INFERIOR, rest_g, rest_h)

    def fitness(self, x: np.array) -> float:
        y = x - o
        f_x = np.sum(y**2 - 10 * np.cos(2 * np.pi * y) + 10)
        return f_x
    
    @staticmethod
    def CEC2017_C12_g1(x):
        y = x - o
        return 4 - np.sum(np.abs(y))
    
    @staticmethod
    def CEC2017_C12_h1(x):
        y = x - o
        return np.sum(y**2) - 4
    
class CEC2017_C13(Problem):
    
    SUPERIOR = np.array([100] * D)
    INFERIOR = np.array([-100] * D)
    
    def __init__(self):
        rest_h = []
        rest_g = [self.CEC2017_C13_g1, self.CEC2017_C13_g2, self.CEC2017_C13_g3]
        super().__init__(ProblemType.CONSTRAINED, self.SUPERIOR, self.INFERIOR, rest_g, rest_h)

    def fitness(self, x: np.array) -> float:
        y = x - o
        f_x = np.sum([100 * (y[i]**2 - y[i+1]**2)**2 + (y[i] - 1)**2 for i in range(D-1)])
        return f_x
    
    @staticmethod
    def CEC2017_C13_g1(x):
        y = x - o
        return np.sum(y**2 - 10 * np.cos(2 * np.pi * y) + 10) - 100
    
    @staticmethod
    def CEC2017_C13_g2(x):
        y = x - o
        return np.sum(y) - 2 * D
    
    @staticmethod
    def CEC2017_C13_g3(x):
        y = x - o
        return 5 - np.sum(y)

class CEC2017_C14(Problem):
    
    SUPERIOR = np.array([100] * D)
    INFERIOR = np.array([-100] * D)
    
    def __init__(self):
        rest_h = [self.CEC2017_C14_h]
        rest_g = [self.CEC2017_C14_g]
        super().__init__(ProblemType.CONSTRAINED, self.SUPERIOR, self.INFERIOR, rest_g, rest_h)

    def fitness(self, x: np.array) -> float:
        y = x - o
        term1 = -20 * np.exp(-0.2 * np.sqrt(np.sum(y**2) / D))
        term2 = 20 - np.exp(np.sum(np.cos(2 * np.pi * y)) / D)
        f_x = term1 + term2 + np.e
        return f_x
    
    @staticmethod
    def CEC2017_C14_g(x):
        y = x - o
        return np.sum(y[1:]**2) + 1 - np.abs(y[0])
    
    @staticmethod
    def CEC2017_C14_h(x):
        y = x - o
        return np.sum(y**2) - 4

class CEC2017_C15(Problem):
    
    SUPERIOR = np.array([100] * D)
    INFERIOR = np.array([-100] * D)
    
    def __init__(self):
        rest_h = [self.CEC2017_C15_h]
        rest_g = [self.CEC2017_C15_g]
        super().__init__(ProblemType.CONSTRAINED, self.SUPERIOR, self.INFERIOR, rest_g, rest_h)

    def fitness(self, x: np.array) -> float:
        y = x - o
        f_x = np.max(np.abs(y))
        return f_x
    
    @staticmethod
    def CEC2017_C15_g(x):
        y = x - o
        return np.sum(y**2) - 100 * D
    
    @staticmethod
    def CEC2017_C15_h(x):
        y = x - o
        f_x = np.max(np.abs(y))
        return np.cos(f_x) + np.sin(f_x)

class CEC2017_C16(Problem):
    
    SUPERIOR = np.array([100] * D)
    INFERIOR = np.array([-100] * D)
    
    def __init__(self):
        rest_h = [self.CEC2017_C16_h]
        rest_g = [self.CEC2017_C16_g]
        super().__init__(ProblemType.CONSTRAINED, self.SUPERIOR, self.INFERIOR, rest_g, rest_h)

    def fitness(self, x: np.array) -> float:
        y = x - o
        f_x = np.sum(np.abs(y))
        return f_x
    
    @staticmethod
    def CEC2017_C16_g(x):
        y = x - o
        return np.sum(y**2) - 100 * D
    
    @staticmethod
    def CEC2017_C16_h(x):
        y = x - o
        f_x = np.sum(np.abs(y))
        return (np.cos(f_x) + np.sin(f_x))**2 - np.exp(np.cos(f_x) + np.sin(f_x)) - 1 + np.exp(1)

class CEC2017_C17(Problem):
    
    SUPERIOR = np.array([100] * D)
    INFERIOR = np.array([-100] * D)
    
    def __init__(self):
        rest_h = [self.CEC2017_C17_h]
        rest_g = [self.CEC2017_C17_g]
        super().__init__(ProblemType.CONSTRAINED, self.SUPERIOR, self.INFERIOR, rest_g, rest_h)

    def fitness(self, x: np.array) -> float:
        y = x - o
        f_x = 1/4000 * np.sum(y**2) + 1 - np.prod(np.cos(y / np.sqrt(np.arange(1, D+1))))
        return f_x
    
    @staticmethod
    def CEC2017_C17_g(x):
        y = x - o
        g = 1 - np.sum(np.sign(np.abs(y)))
        for i in range(D):
            g -= np.sum(y**2) - 1
        return g
    
    @staticmethod
    def CEC2017_C17_h(x):
        y = x - o
        return np.sum(y**2) - 4 * D

class CEC2017_C18(Problem):
    
    SUPERIOR = np.array([100] * D)
    INFERIOR = np.array([-100] * D)
    
    def __init__(self):
        rest_h = [self.CEC2017_C18_h]
        rest_g = [self.CEC2017_C18_g1, self.CEC2017_C18_g2]
        super().__init__(ProblemType.CONSTRAINED, self.SUPERIOR, self.INFERIOR, rest_g, rest_h)

    def fitness(self, x: np.array) -> float:
        y = x - o
        z = np.where(np.abs(y) < 0.5, y, 0.5 * np.round(2 * y))
        f_x = np.sum(z**2 - 10 * np.cos(2 * np.pi * z) + 10)
        return f_x
    
    @staticmethod
    def CEC2017_C18_g1(x):
        y = x - o
        return 1 - np.sum(np.abs(y))
    
    @staticmethod
    def CEC2017_C18_g2(x):
        y = x - o
        return np.sum(y**2) - 100 * D
    
    @staticmethod
    def CEC2017_C18_h(x):
        y = x - o
        h1 = np.sum([100 * (y[i] - y[i+1])**2 for i in range(D-1)])
        h2 = np.prod([np.sin(np.pi * (y[i] - 1))**2 for i in range(D)])
        return h1 + h2
    
class CEC2017_C19(Problem):
    SUPERIOR = np.array([50] * D)
    INFERIOR = np.array([-50] * D)

    def __init__(self):
        rest_h = []
        rest_g = [self.CEC2017_C19_g1, self.CEC2017_C19_g2]
        super().__init__(ProblemType.CONSTRAINED, self.SUPERIOR, self.INFERIOR, rest_g, rest_h)

    def fitness(self, x: np.array) -> float:
        y = x - o
        f_x = np.sum(np.abs(y)**0.5 + 2 * np.sin(y**3))
        return f_x
    
    @staticmethod
    def CEC2017_C19_g1(x):
        y = x - o
        return np.sum(-10 * np.exp(-0.2 * np.sqrt(y[:-1]**2 + y[1:]**2))) + (D - 1) * 10 / np.exp(5)
    
    @staticmethod
    def CEC2017_C19_g2(x):
        y = x - o
        return np.sum(np.sin(2 * y)**2) - 0.5 * D

class CEC2017_C20(Problem):
    SUPERIOR = np.array([100] * D)
    INFERIOR = np.array([-100] * D)

    def __init__(self):
        rest_h = []
        rest_g = [self.CEC2017_C20_g1, self.CEC2017_C20_g2]
        super().__init__(ProblemType.CONSTRAINED, self.SUPERIOR, self.INFERIOR, rest_g, rest_h)

    def fitness(self, x: np.array) -> float:
        y = x - o
        f_x = 0.5 + np.sum(
            [(np.sin(np.sqrt(y[i]**2 + y[(i+1) % D]**2))**2 - 0.5) / 
             (1 + 0.001 * (y[i]**2 + y[(i+1) % D]**2)**2) 
            for i in range(D)]
        )
        return f_x
    
    @staticmethod
    def CEC2017_C20_g1(x):
        y = x - o
        g1 = np.cos(np.sum(y))**2 - 0.25 * np.cos(np.sum(y)) - 0.125
        return g1
    
    @staticmethod
    def CEC2017_C20_g2(x):
        y = x - o
        g2 = np.exp(np.cos(np.sum(y))) - np.exp(0.25)
        return g2


class CEC2017_C21(Problem):
    
    SUPERIOR = np.array([100] * D)
    INFERIOR = np.array([-100] * D)
    
    M = ortho_group.rvs(D)

    def __init__(self):
        rest_h = []
        rest_g = [self.CEC2017_C21_g1, self.CEC2017_C21_g2]
        super().__init__(ProblemType.CONSTRAINED, self.SUPERIOR, self.INFERIOR, rest_g, rest_h)

    def fitness(self, x: np.array) -> float:
        y = x - o
        f_x = np.sum(y**2 - 10 * np.cos(2 * np.pi * y) + 10)
        return f_x
    
    @staticmethod
    def CEC2017_C21_g1(x):
        y = x - o
        z = CEC2017_C21.M @ y
        return 4 - np.sum(np.abs(z))
    
    @staticmethod
    def CEC2017_C21_g2(x):
        y = x - o
        z = CEC2017_C21.M @ y
        return np.sum(z**2) - 4

class CEC2017_C22(Problem):
    SUPERIOR = np.array([100] * D)
    INFERIOR = np.array([-100] * D)
    
    M = ortho_group.rvs(D)

    def __init__(self):
        rest_h = []
        rest_g = [self.CEC2017_C22_g1, self.CEC2017_C22_g2, self.CEC2017_C22_g3]
        super().__init__(ProblemType.CONSTRAINED, self.SUPERIOR, self.INFERIOR, rest_g, rest_h)

    def fitness(self, x: np.array) -> float:
        y = x - o
        z = self.M @ y
        f_x = np.sum([100 * (z[i]**2 - z[i+1])**2 + (z[i] - 1)**2 for i in range(D-1)])
        return f_x
    
    @staticmethod
    def CEC2017_C22_g1(x):
        y = x - o
        z = CEC2017_C22.M @ y
        return np.sum(z**2 - 10 * np.cos(2 * np.pi * z) + 10) - 100
    
    @staticmethod
    def CEC2017_C22_g2(x):
        y = x - o
        z = CEC2017_C22.M @ y
        return np.sum(z) - 2 * D
    
    @staticmethod
    def CEC2017_C22_g3(x):
        y = x - o
        z = CEC2017_C22.M @ y
        return 5 - np.sum(z)

class CEC2017_C23(Problem):
    SUPERIOR = np.array([100] * D)
    INFERIOR = np.array([-100] * D)
    
    M = ortho_group.rvs(D)

    def __init__(self):
        rest_h = [self.CEC2017_C23_h1]
        rest_g = [self.CEC2017_C23_g1]
        super().__init__(ProblemType.CONSTRAINED, self.SUPERIOR, self.INFERIOR, rest_g, rest_h)

    def fitness(self, x: np.array) -> float:
        y = x - o
        z = self.M @ y
        term1 = -20 * np.exp(-0.2 * np.sqrt(np.sum(z**2) / D))
        term2 = 20 - np.exp(np.sum(np.cos(2 * np.pi * z) / D))
        f_x = term1 + term2 + np.e
        return f_x
    
    @staticmethod
    def CEC2017_C23_g1(x):
        y = x - o
        z = CEC2017_C23.M @ y
        return np.sum(z[1:]**2) + 1 - np.abs(z[0])
    
    @staticmethod
    def CEC2017_C23_h1(x):
        y = x - o
        z = CEC2017_C23.M @ y
        return np.sum(z**2) - 4
    
class CEC2017_C24(Problem):
    SUPERIOR = np.array([100] * D)
    INFERIOR = np.array([-100] * D)
    
    M = ortho_group.rvs(D)

    def __init__(self):
        rest_h = [self.CEC2017_C24_h1]
        rest_g = [self.CEC2017_C24_g1]
        super().__init__(ProblemType.CONSTRAINED, self.SUPERIOR, self.INFERIOR, rest_g, rest_h)

    def fitness(self, x: np.array) -> float:
        y = x - o
        z = self.M @ y
        f_x = np.max(np.abs(z))
        return f_x
    
    @staticmethod
    def CEC2017_C24_g1(x):
        y = x - o
        z = CEC2017_C24.M @ y
        return np.sum(z**2) - 100 * D
    
    @staticmethod
    def CEC2017_C24_h1(x):
        y = x - o
        z = CEC2017_C24.M @ y
        f_x = np.max(np.abs(z))
        return np.cos(f_x) + np.sin(f_x)

class CEC2017_C25(Problem):
    SUPERIOR = np.array([100] * D)
    INFERIOR = np.array([-100] * D)
    
    M = ortho_group.rvs(D)

    def __init__(self):
        rest_h = [self.CEC2017_C25_h1]
        rest_g = [self.CEC2017_C25_g1]
        super().__init__(ProblemType.CONSTRAINED, self.SUPERIOR, self.INFERIOR, rest_g, rest_h)

    def fitness(self, x: np.array) -> float:
        y = x - o
        z = self.M @ y
        f_x = np.sum(np.abs(z))
        return f_x
    
    @staticmethod
    def CEC2017_C25_g1(x):
        y = x - o
        z = CEC2017_C25.M @ y
        return np.sum(z**2) - 100 * D
    
    @staticmethod
    def CEC2017_C25_h1(x):
        y = x - o
        z = CEC2017_C25.M @ y
        f_x = np.sum(np.abs(z))
        return (np.cos(f_x) + np.sin(f_x))**2 - np.exp(np.cos(f_x) + np.sin(f_x)) - 1 + np.exp(1)

class CEC2017_C26(Problem):
    SUPERIOR = np.array([100] * D)
    INFERIOR = np.array([-100] * D)
    
    M = ortho_group.rvs(D)

    def __init__(self):
        rest_h = [self.CEC2017_C26_h1]
        rest_g = [self.CEC2017_C26_g1]
        super().__init__(ProblemType.CONSTRAINED, self.SUPERIOR, self.INFERIOR, rest_g, rest_h)

    def fitness(self, x: np.array) -> float:
        y = x - o
        z = self.M @ y
        term1 = 1 / 4000 * np.sum(z**2)
        term2 = 1 - np.prod(np.cos(z / np.sqrt(np.arange(1, D + 1))))
        f_x = term1 + term2
        return f_x
    
    @staticmethod
    def CEC2017_C26_g1(x):
        y = x - o
        z = CEC2017_C26.M @ y
        return 1 - np.sum(np.sign(np.abs(z)) - np.sum(z**2) - 1)
    
    @staticmethod
    def CEC2017_C26_h1(x):
        y = x - o
        z = CEC2017_C26.M @ y
        return np.sum(z**2) - 4 * D

class CEC2017_C27(Problem):
    SUPERIOR = np.array([100] * D)
    INFERIOR = np.array([-100] * D)
    
    M = ortho_group.rvs(D)

    def __init__(self):
        rest_h = [self.CEC2017_C27_h1]
        rest_g = [self.CEC2017_C27_g1, self.CEC2017_C27_g2]
        super().__init__(ProblemType.CONSTRAINED, self.SUPERIOR, self.INFERIOR, rest_g, rest_h)

    def fitness(self, x: np.array) -> float:
        y = x - o
        z = self.M @ y
        z = np.where(np.abs(y) < 0.5, y, 0.5 * np.round(2 * y))
        f_x = np.sum(z**2 - 10 * np.cos(2 * np.pi * z) + 10)
        return f_x
    
    @staticmethod
    def CEC2017_C27_g1(x):
        y = x - o
        z = CEC2017_C27.M @ y
        return 1 - np.sum(np.abs(z))
    
    @staticmethod
    def CEC2017_C27_g2(x):
        y = x - o
        z = CEC2017_C27.M @ y
        return np.sum(z**2) - 100 * D
    
    @staticmethod
    def CEC2017_C27_h1(x):
        y = x - o
        z = CEC2017_C27.M @ y
        return np.sum(100 * (z[:-1]**2 - z[1:]) + (np.sin(z) - 1))

class CEC2017_C28(Problem):
    SUPERIOR = np.array([50] * D)
    INFERIOR = np.array([-50] * D)
    
    M = ortho_group.rvs(D)

    def __init__(self):
        rest_h = []
        rest_g = [self.CEC2017_C28_g1, self.CEC2017_C28_g2]
        super().__init__(ProblemType.CONSTRAINED, self.SUPERIOR, self.INFERIOR, rest_g, rest_h)

    def fitness(self, x: np.array) -> float:
        y = x - o
        z = self.M @ y
        f_x = np.sum(np.abs(z)**0.5 + 2 * np.sin(z**3))
        return f_x
    
    @staticmethod
    def CEC2017_C28_g1(x):
        y = x - o
        z = CEC2017_C28.M @ y
        return np.sum(-10 * np.exp(-0.2 * np.sqrt(z[:-1]**2 + z[1:]**2))) + (D - 1) * 10 / np.exp(5)
    
    @staticmethod
    def CEC2017_C28_g2(x):
        y = x - o
        z = CEC2017_C28.M @ y
        return np.sum(np.sin(2 * z)**2) - 0.5 * D
