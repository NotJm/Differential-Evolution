import numpy as np
from .Problem import Problem, ProblemType
from scipy.stats import ortho_group

D = 10

def generate_z(x, o, transformation=None, M=None):
    z = x - o
    if transformation == 'matrix_multiply' and M is not None:
        z = z @ M
    elif transformation == 'plus_one':
        z = x + 1 - o
    return z


class CEC2010_C01(Problem):
    
    SUPERIOR = np.array([10] * D)
    INFERIOR = np.array([0] * D)
    
    o = np.array([
        0.030858718087483, -0.078632292353156, 0.048651146638038, -0.069089831066354, -0.087918542941928,
        0.088982639811141, 0.074143235639847, -0.086527593580149, -0.020616531903907, 0.055586106499231,
        0.059285954883598, -0.040671485554685, -0.087399911887693, -0.01842585125741, -0.005184912793062, 
        -0.039892037937026, 0.036509229387458, 0.026046414854433, -0.067133862936029, 0.082780189144943, 
        -0.049336722577062, 0.018503188080959, 0.051610619131255, 0.018613117768432, 0.093448598181657, 
        -0.071208840780873, -0.036535677894572, -0.03126128526933, 0.099243805247963, 0.053872445945574
    ])

    def __init__(self):
        rest_g = [self.cec2010_c01_g1, self.cec2010_c01_g2]
        super().__init__(
            ProblemType.CONSTRAINED, 
            self.SUPERIOR, self.INFERIOR, 
            rest_g, []
        )
    
    def fitness(self, individuo: np.array) -> float:
        z = generate_z(individuo, self.o[:D])
        f_x = -np.abs((np.sum(np.cos(z)**4) - 2 * np.prod(np.cos(z)**2)) / np.sqrt(np.sum(np.arange(1, D+1) * z**2)))
        return f_x

    @staticmethod
    def cec2010_c01_g1(x):
        z = generate_z(x, CEC2010_C01.o[:D])
        return 0.75 - np.prod(z)  # restriccion de desigualdad <= 0
    
    @staticmethod
    def cec2010_c01_g2(x):
        z = generate_z(x, CEC2010_C01.o[:D])
        return np.sum(z) - 7.5 * len(z)  # restriccion de desigualdad <= 0

class CEC2010_C02(Problem):
    
    SUPERIOR = np.array([5.12] * D)
    INFERIOR = np.array([-5.12] * D)
    
    o = [
        -0.066939099286697,	0.470966419894494, -0.490528349401176, -0.312203454689423, -0.124759576300523, 
        -0.247823908806285, -0.448077079941866,	0.326494954650117, 0.493435908752668, 0.061699778818925,
        -0.30251101183711, -0.274045146932175, -0.432969960330318, 0.062239193145781, -0.188163731545079,
        -0.100709842052095, -0.333528971180922, -0.496627672944882,	-0.288650116941944,	0.435648113198148,
        -0.348261107144255, 0.456550427329479, -0.286843419772511, 0.145639015401174, -0.038656025783381,
        0.333291935226012, -0.293687524888766, -0.347859473554797, -0.089300971656411, 0.142027393193559
    ]

    def __init__(self):
        rest_g = [self.cec2010_c02_g1, self.cec2010_c02_g2]
        rest_h = [self.cec2010_c02_h1]
        super().__init__(
            ProblemType.CONSTRAINED, 
            self.SUPERIOR, self.INFERIOR, 
            rest_g, rest_h
        )
    
    def fitness(self, individuo: np.array) -> float:
        z = generate_z(individuo, self.o[:D])
        f_x = np.max(z)
        return f_x
    

    @staticmethod
    def cec2010_c02_g1(x):
        z = generate_z(x, CEC2010_C02.o[:D])
        return 10 - (1/D) * np.sum(z**2 - 10 * np.cos(2 * np.pi * z) + 10)  # restriccion de desigualdad <= 0
    
    @staticmethod
    def cec2010_c02_g2(x): 
        z = generate_z(x, CEC2010_C02.o[:D])
        return (1/len(z)) * np.sum(z**2 - 10 * np.cos(2 * np.pi * z) + 10) - 15  # restriccion de desigualdad <= 0
    
    @staticmethod
    def cec2010_c02_h1(x):
        z = generate_z(x, CEC2010_C02.o[:D])
        y = z - 0.5
        return (1/len(y)) * np.sum(y**2 - 10 * np.cos(2 * np.pi * y) + 10) - 20  # restriccion de desigualdad <= 0

class CEC2010_C03(Problem):
    
    SUPERIOR = np.array([1000] * D)
    INFERIOR = np.array([-1000] * D)
    
    o = np.array([
        111.17633500088529, 92.07880492633424, 417.9818592609036, 253.16188128024302, 363.5279986597767,
        314.334093889305, 187.32739056163342, 240.4363027535162, 422.60090880560665, 327.63042902581515,
        62.04762897064405, 25.435663968682125, 360.56773191905114, 154.9226721156832, 33.161292034425806,
        177.8091733067186, 262.58198940407755, 436.9800562237075, 476.6400624069227, 331.2167787340325,
        75.205948242522, 484.33624811710115, 258.4696246506982, 419.8919566566751, 357.51468895930395,
        166.3771729386268, 47.59455935830133, 188.20606700809785, 184.7964918401363, 267.9201349178807
    ])

    def __init__(self):
        rest_h = [self.cec2010_c03_h]
        super().__init__(
            ProblemType.CONSTRAINED, 
            self.SUPERIOR, self.INFERIOR, 
            [], rest_h
        )
    
    def fitness(self, individuo: np.array) -> float:
        z = generate_z(individuo, self.o[:D])
        f_x = np.sum(100 * (z[:-1]**2 - z[1:])**2 + (z[:-1] - 1)**2)
        return f_x

    @staticmethod
    def cec2010_c03_h(x):
        z = generate_z(x, CEC2010_C03.o[:D])
        return np.sum((z[:-1] - z[1:])**2)  # restriccion de igualdad = 0

class CEC2010_C04(Problem):
    
    SUPERIOR = np.array([50] * D)
    INFERIOR = np.array([-50] * D)
    
    o = np.array([
        0.820202353727904, 5.260154140335203, -1.694610371739177, -5.589298730330406, -0.141736605495543,
        9.454675508078164, 8.795744608532939, 9.687346331423548, -3.246522827444976, 6.647399971577617,
        1.434490229836026, -0.506531215086801, 0.558594225280784, 7.919942423520642, 1.383716002673571,
        -1.520153615528276, -2.266737465474915, 6.48052999726508, -8.893207968949003, -3.528743044935322,
        6.063486037065154, -4.51585211274229, 7.320477892009357, -8.990263774675665, 9.446412007392851,
        -6.41068985463494, -9.135251626491991, 2.07763837492787, 8.051026378030816, -1.002691032064544
    ])

    def __init__(self):
        rest_h = [self.cec2010_c04_h1, self.cec2010_c04_h2, self.cec2010_c04_h3, self.cec2010_c04_h4]
        super().__init__(
            ProblemType.CONSTRAINED, 
            self.SUPERIOR, self.INFERIOR, 
            [], rest_h
        )
    
    def fitness(self, individuo: np.array) -> float:
        z = generate_z(individuo, self.o[:D])
        f_x = np.max(z)
        return f_x

    @staticmethod
    def cec2010_c04_h1(x):
        z = generate_z(x, CEC2010_C04.o[:D])
        return (1/D) * np.sum(z * np.cos(np.sqrt(np.abs(z))))  # restriccion de igualdad = 0
    
    @staticmethod
    def cec2010_c04_h2(x):
        z = generate_z(x, CEC2010_C04.o[:D])
        return np.sum((z[:D//2-1] - z[1:D//2])**2)  # restriccion de igualdad = 0

    @staticmethod
    def cec2010_c04_h3(x):
        z = generate_z(x, CEC2010_C04.o[:D])
        return np.sum((z[D//2:D-1]**2 - z[D//2+1:]**2)**2)  # restriccion de igualdad = 0

    @staticmethod
    def cec2010_c04_h4(x):
        z = generate_z(x, CEC2010_C04.o[:D])
        return np.sum(z)  # restriccion de igualdad = 0


class CEC2010_C05(Problem):
    
    SUPERIOR = np.array([600] * D)
    INFERIOR = np.array([-600] * D)
    
    o = np.array([
        72.10900225247575, 9.007673762322495, 51.86632637302316, 41.365704820161, 93.18768763916974,
        74.53341902482204, 63.745479932407655, 7.496986033468282, 56.16729598807964, 17.71630810614085,
        28.009655663065143, 29.36357615570272, 26.966653374740996, 6.892189514516317, 44.29071160734624,
        84.35803966449319, 81.16906730972529, 92.76919270133271, 3.826058034047476, 7.231864548985054,
        14.446069444832405, 46.49943418775763, 22.155722253817412, 69.11723738661682, 88.99628570349459,
        58.74823912291344, 52.265369214509846, 47.030120955005074, 53.23321779503931, 5.778976086909701
    ])

    def __init__(self):
        rest_h = [self.cec2010_c05_h1, self.cec2010_c05_h2]
        super().__init__(
            ProblemType.CONSTRAINED, 
            self.SUPERIOR, self.INFERIOR, 
            [], rest_h
        )
    
    def fitness(self, individuo: np.array) -> float:
        z = generate_z(individuo, self.o[:D])
        f_x = np.max(z)
        return f_x

    @staticmethod
    def cec2010_c05_h1(x):
        z = generate_z(x, CEC2010_C05.o[:D])
        return (1/D) * np.sum(-z * np.sin(np.sqrt(np.abs(z))))  # restriccion de igualdad = 0
    
    @staticmethod
    def cec2010_c05_h2(x):
        z = generate_z(x, CEC2010_C05.o[:D])
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
