import numpy as np
from .Problem import Problem, ProblemType

# Problem 01
class CEC2006_G01(Problem):
    
    SUPERIOR = np.array([1,1,1,1,1,1,1,1,1,100,100,100,1])
    INFERIOR = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0])

    def __init__(self):
        rest_g = [
            self.cec2006_g01_g1,self.cec2006_g01_g2,self.cec2006_g01_g3,
            self.cec2006_g01_g4,self.cec2006_g01_g5,self.cec2006_g01_g6,
            self.cec2006_g01_g7,self.cec2006_g01_g8,self.cec2006_g01_g9
        ]
        super().__init__(
            ProblemType.CONSTRAINED, #TIPO DE PROBLEMA
            self.SUPERIOR,self.INFERIOR, #LÍMITES
            rest_g,[] #RESTRICCIONES
        )
    
    def fitness(self, individuo: np.array) -> float:
        sum1 = np.sum(individuo[0:4])
        sum2 = np.sum(individuo[0:4]**2)
        sum3 = np.sum(individuo[4:13])
        f_x = 5 * sum1 - 5 * sum2 - sum3
        return f_x

    @staticmethod
    def cec2006_g01_g1(x):
        return 2 * x[0] + 2 * x[1] + x[9] + x[10] - 10  # restriccion 1 de desigualdad <= 0
    
    @staticmethod
    def cec2006_g01_g2(x):
        return 2 * x[0] + 2 * x[2] + x[9] + x[11] - 10  # restriccion 2 de desigualdad <= 0
    
    @staticmethod
    def cec2006_g01_g3(x):
        return 2 * x[1] + 2 * x[2] + x[10] + x[11] - 10 # restriccion 3 de desigualdad <= 0
    
    @staticmethod
    def cec2006_g01_g4(x):
        return -8 * x[0] + x[9] # restriccion 4 de desigualdad <= 0
    
    @staticmethod
    def cec2006_g01_g5(x):
        return -8 * x[1] + x[10] # restriccion 5 de desigualdad <= 0
    
    @staticmethod
    def cec2006_g01_g6(x):
        return -8 * x[2] + x[11] # restriccion 6 de desigualdad <= 0
    
    @staticmethod              
    def cec2006_g01_g7(x):
        return -2 * x[3] - x[4] + x[9] #restriccion 7 de desigualdad <= 0

    @staticmethod
    def cec2006_g01_g8(x):
        return -2 * x[5] - x[6] + x[10] # restriccion 8 de desigualdad <= 0

    @staticmethod
    def cec2006_g01_g9(x):
        return -2 * x[7] - x[8] + x[11] # restriccion 9 de desigualdad <= 0
    


#Notas: x es un arrelo de tamaño t= 13 y 0 ≤ xi ≤ 1(i = 1,...,9),0 ≤ xi ≤ 100(i = 10,11,12) 

    
#********************************************************************************************************************************

#Problem 02

class CEC2006_G02(Problem):
    SUPERIOR = np.array([10] * 20)
    INFERIOR = np.array([0] * 20)

    def __init__(self):
        rest_g = [
            self.cec2006_g02_g1,self.cec2006_g02_g2,
        ]
        super().__init__(
            ProblemType.CONSTRAINED, #TIPO DE PROBLEMA
            self.SUPERIOR,self.INFERIOR, #LÍMITES
            rest_g,[] #RESTRICCIONES
        )

    def fitness(self, individuo: np.array) -> float:
        sum_cos4 = np.sum(np.cos(individuo)**4)
        prod_cos2 = np.prod(np.cos(individuo)**2)
        sum_ix2 = np.sum((np.arange(1, len(individuo) + 1) * individuo**2))
        f_x = -abs((sum_cos4 - 2 * prod_cos2) / np.sqrt(sum_ix2))
        return f_x

    @staticmethod
    def cec2006_g02_g1(x):  # restriccion 1 de desigualdad <= 0
        product_x = np.prod(x)
        result = 0.75 - product_x
        return result

    @staticmethod
    def cec2006_g02_g2(x):  # restriccion 2 de desigualdad <= 0
        sum_x = np.sum(x)
        n = len(x)
        result = sum_x - 7.5 * n
        return result


#Notas: X es un arreglo de tamaño  t=20, 0<Xi<=10, (i=1,2,...,t)

#********************************************************************************************************************************

#Problem 03

class CEC2006_G03(Problem):
    SUPERIOR = np.array([1] * 10)
    INFERIOR = np.array([0] * 10)

    def __init__(self):
        rest_h = [
            self.cec2006_g03_h1
        ]
        super().__init__(
            ProblemType.CONSTRAINED, #TIPO DE PROBLEMA
            self.SUPERIOR,self.INFERIOR, #LÍMITES
            [],rest_h #RESTRICCIONES
        )

    def fitness(self, individuo: np.array) -> float:
        n = len(individuo)
        product_x = np.prod(individuo)
        f_x = -(np.sqrt(n)**n * product_x)
        return f_x

    @staticmethod
    def cec2006_g03_h1(x): # restriccion 1 de igualdad = 0
        sum_x_squared = np.sum(x**2) 
        return sum_x_squared - 1

#Notas: donde x es de tamaño t=10, 0 ≤ xi ≤ 1(i = 1,...,n).

#********************************************************************************************************************************

#Problem 04

class CEC2006_G04(Problem):
    
    SUPERIOR = np.array([102, 45, 45, 45, 45])
    INFERIOR = np.array([78, 33, 27, 27, 27])

    def __init__(self):
        rest_g = [
            self.cec2006_g04_g2,self.cec2006_g04_g1,self.cec2006_g04_g3,
            self.cec2006_g04_g4,self.cec2006_g04_g5,self.cec2006_g04_g6
        ]
        super().__init__(
            ProblemType.CONSTRAINED, #TIPO DE PROBLEMA
            self.SUPERIOR,self.INFERIOR, #LÍMITES
            rest_g,[] #RESTRICCIONES
        )

    def fitness(self, individuo: np.array) -> float:
        x = individuo
        f_x = 5.3578547 * x[2]**2 + 0.8356891 * x[0] * x[4] + 37.293239 * x[0] - 40792.141
        return f_x
    
    @staticmethod
    def cec2006_g04_g1(x): # restriccion 1 de desigualdad <= 0
        return 85.334407 + 0.0056858 * x[1] * x[4] + 0.0006262 * x[0] * x[3] - 0.0022053 * x[2] * x[4] - 92

    @staticmethod
    def cec2006_g04_g2(x): # restriccion 2 de desigualdad <= 0
        return -85.334407 - 0.0056858 * x[1] * x[4] - 0.0006262 * x[0] * x[3] + 0.0022053 * x[2] * x[4]

    @staticmethod
    def cec2006_g04_g3(x): # restriccion 3 de desigualdad <= 0
        return 80.51249 + 0.0071317 * x[1] * x[4] + 0.0029955 * x[0] * x[1] + 0.0021813 * x[2]**2 - 110

    @staticmethod
    def cec2006_g04_g4(x): # restriccion 4 de desigualdad <= 0
        return -80.51249 - 0.0071317 * x[1] * x[4] - 0.0029955 * x[0] * x[1] - 0.0021813 * x[2]**2 + 90

    @staticmethod
    def cec2006_g04_g5(x): # restriccion 5 de desigualdad <= 0
        return 9.300961 + 0.0047026 * x[2] * x[4] + 0.0012547 * x[0] * x[2] + 0.0019085 * x[2] * x[3] - 25

    @staticmethod
    def cec2006_g04_g6(x): # restriccion 6 de desigualdad <= 0
        return -9.300961 - 0.0047026 * x[2] * x[4] - 0.0012547 * x[0] * x[2] - 0.0019085 * x[2] * x[3] + 20


#Notas: donde 78 ≤ x1 ≤ 102, 33 ≤ x2 ≤ 45 y 27 ≤ xi ≤ 45(i = 3,4,5)

#********************************************************************************************************************************

#Problem 05

class CEC2006_G05(Problem):
    
    # WITH BOUNDS:
    # 0 ≤ x1 ≤ 1200, 0 ≤ x2 ≤ 1200, −0.55 ≤ x3 ≤ 0.55 y −0.55 ≤ x4 ≤ 0.55.
        
    SUPERIOR = np.array([1200, 1200, 0.55, 0.55])
    INFERIOR = np.array([0, 0, -0.55, -0.55])

    def __init__(self):
        rest_g = [
            self.cec2006_g05_g1,self.cec2006_g05_g2,
        ]
        rest_h = [
            self.cec2006_g05_h1,self.cec2006_g05_h2,self.cec2006_g05_h3
        ]
        super().__init__(
            ProblemType.CONSTRAINED, #TIPO DE PROBLEMA
            self.SUPERIOR,self.INFERIOR, #LÍMITES
            rest_g,rest_h #RESTRICCIONES
        )

    def fitness(self, individuo:np.array): 
        x = individuo
        f_x = 3 * x[0] + 0.000001* x[0]**3 + 2* x[1] + (0.000002/3) * x[1]**3
        return f_x

    @staticmethod
    def cec2006_g05_g1(x): # restriccion 1 de desigualdad <= 0
        return -x[3] + x[2] - 0.55
    
    @staticmethod
    def cec2006_g05_g2(x): # restriccion 2 de desigualdad <= 0
        return -x[2] + x[3] - 0.55
    
    @staticmethod
    def cec2006_g05_h1(x): # restriccion 3 de igualdad = 0
        return 1000 * np.sin(-x[2] - 0.25) + 1000 * np.sin(-x[3] - 0.25) + 894.8 - x[0]
    
    @staticmethod
    def cec2006_g05_h2(x): # restriccion 4 de igualdad = 0
        return 1000 * np.sin(x[2] - 0.25) + 1000 * np.sin(x[2] - x[3] - 0.25) + 894.8 - x[1]
    
    @staticmethod
    def cec2006_g05_h3(x): # restriccion 5 de igualdad = 0
        return 1000 * np.sin(x[3] - 0.25) + 1000 * np.sin(x[3] - x[2] - 0.25) + 1294.8


#********************************************************************************************************************************

#Problem 06
class CEC2006_G06(Problem):
    
    # WITH BOUNDS:
    # 13 ≤ x1 ≤ 100 y 0 ≤ x2 ≤ 100
        
    SUPERIOR = np.array([100, 100])
    INFERIOR = np.array([13, 0])

    def __init__(self):
        rest_g = [
            self.cec2006_g06_g1,self.cec2006_g06_g2,
        ]
        super().__init__(
            ProblemType.CONSTRAINED, #TIPO DE PROBLEMA
            self.SUPERIOR,self.INFERIOR, #LÍMITES
            rest_g, [] #RESTRICCIONES
        )
    
    def fitness(self, individuo:np.array): 
        x = individuo
        f_x = (x[0] -10)**3 + (x[1]-20)**3
        return f_x
    
    @staticmethod
    def cec2006_g06_g1(x): # restriccion 1 de desigualdad <= 0
        return -(x[0] -5)**2 - (x[1] - 5)**2 +100 
    
    @staticmethod
    def cec2006_g06_g2(x): # restriccion 2 de desigualdad <= 0
        return (x[0] - 6)**2 + (x[1] - 5)**2 - 82.81

#********************************************************************************************************************************

#Problem 07

import numpy as np

class CEC2006_G07(Problem):
    SUPERIOR = np.array([10] * 10)
    INFERIOR = np.array([-10] * 10)

    def __init__(self):
        rest_g = [
            self.cec2006_g07_g1,self.cec2006_g07_g2,self.cec2006_g07_g3,self.cec2006_g07_g4,
            self.cec2006_g07_g5,self.cec2006_g07_g6,self.cec2006_g07_g7,self.cec2006_g07_g8,
        ]
        super().__init__(
            ProblemType.CONSTRAINED, #TIPO DE PROBLEMA
            self.SUPERIOR, self.INFERIOR, #LÍMITES
            rest_g, [] #RESTRICCIONES
        )

    def fitness(self, individuo: np.array) -> float:
        x = individuo
        f_x = x[0]**2 + x[1]**2 + x[0]*x[1] - 14*x[0] - 16*x[1] + (x[2] - 10)**2 + 4*(x[3] - 5)**2 + (x[4] - 3)**2 + 2*(x[5] - 1)**2 + 5*x[6]**2 + 7*(x[7] - 11)**2 + 2*(x[8] - 10)**2 + (x[9] - 7)**2 + 45
        return f_x

    @staticmethod
    def cec2006_g07_g1(x):  # restriccion 1 de desigualdad <= 0
        result = -105 + 4*x[0] + 5*x[1] - 3*x[6] + 9*x[7]
        return result

    @staticmethod
    def cec2006_g07_g2(x):  # restriccion 2 de desigualdad <= 0
        result = 10*x[0] - 8*x[1] - 17*x[6] + 2*x[7]
        return result

    @staticmethod
    def cec2006_g07_g3(x):  # restriccion 3 de desigualdad <= 0
        result = -8*x[0] + 2*x[1] + 5*x[8] - 2*x[9] - 12
        return result

    @staticmethod
    def cec2006_g07_g4(x):  # restriccion 4 de desigualdad <= 0
        result = 3*(x[0] - 2)**2 + 4*(x[1] - 3)**2 + 2*x[2]**2 - 7*x[3] - 120
        return result

    @staticmethod
    def cec2006_g07_g5(x):  # restriccion 5 de desigualdad <= 0
        result = 5*x[0]**2 + 8*x[1] + (x[2] - 6)**2 - 2*x[3] - 40
        return result

    @staticmethod
    def cec2006_g07_g6(x):  # restriccion 6 de desigualdad <= 0
        result = x[0]**2 + 2*(x[1] - 2)**2 - 2*x[0]*x[1] + 14*x[4] - 6*x[5]
        return result

    @staticmethod
    def cec2006_g07_g7(x):  # restriccion 7 de desigualdad <= 0
        result = 0.5*(x[0] - 8)**2 + 2*(x[1] - 4)**2 + 3*x[4]**2 - x[5] - 30
        return result

    @staticmethod
    def cec2006_g07_g8(x):  # restriccion 8 de desigualdad <= 0
        result = -3*x[0] + 6*x[1] + 12*(x[8] - 8)**2 - 7*x[9]
        return result
    
    #Notas: donde −10 ≤ xi ≤ 10(i = 1,...,10)
    
#********************************************************************************************************************************

#Problem 08

class CEC2006_G08(Problem):
    SUPERIOR = np.array([10, 10])
    INFERIOR = np.array([0, 0])

    def __init__(self):
        rest_g = [
            self.cec2006_g08_g1,self.cec2006_g08_g2,
        ]
        super().__init__(
            ProblemType.CONSTRAINED, #TIPO DE PROBLEMA
            self.SUPERIOR, self.INFERIOR, #LÍMITES
            rest_g, [] #RESTRICCIONES
        )

    def fitness(self, individuo: np.array) -> float:
        x = individuo
        f_x = - (np.sin(2 * np.pi * x[0])**3 * np.sin(2 * np.pi * x[1])) / (x[0]**3 * (x[0] + x[1]))
        return f_x

    @staticmethod
    def cec2006_g08_g1(x):  # restriccion 1 de desigualdad <= 0
        result = x[0]**2 - x[1] + 1
        return result

    @staticmethod
    def cec2006_g08_g2(x):  # restriccion 2 de desigualdad <= 0
        result = 1 - x[0] + (x[1] - 4)**2
        return result
    
#********************************************************************************************************************************

#Problem 9

class CEC2006_G09(Problem):
    SUPERIOR = np.array([10] * 7)
    INFERIOR = np.array([-10] * 7)

    def __init__(self):
        rest_g = [
            self.cec2006_g09_g1,self.cec2006_g09_g2,self.cec2006_g09_g3,self.cec2006_g09_g4,
        ]
        super().__init__(
            ProblemType.CONSTRAINED, #TIPO DE PROBLEMA
            self.SUPERIOR, self.INFERIOR, #LÍMITES
            rest_g, [] #RESTRICCIONES
        )

    def fitness(self, individuo: np.array) -> float:
        x = individuo
        f_x = (x[0] - 10)**2 + 5*(x[1] - 12)**2 + x[2]**4 + 3*(x[3] - 11)**2 + 10*x[4]**6 + 7*x[5]**2 + x[6]**4 - 4*x[5]*x[6] - 10*x[5] - 8*x[6]
        return f_x

    @staticmethod
    def cec2006_g09_g1(x):  # restriccion 1 de desigualdad <= 0
        result = -127 + 2*x[0]**2 + 3*x[1]**4 + x[2] + 4*x[3]**2 + 5*x[4]
        return result

    @staticmethod
    def cec2006_g09_g2(x):  # restriccion 2 de desigualdad <= 0
        result = -282 + 7*x[0] + 3*x[1] + 10*x[2]**2 + x[3] - x[4]
        return result

    @staticmethod
    def cec2006_g09_g3(x):  # restriccion 3 de desigualdad <= 0
        result = -196 + 23*x[0] + x[1]**2 + 6*x[5]**2 - 8*x[6]
        return result

    @staticmethod
    def cec2006_g09_g4(x):  # restriccion 4 de desigualdad <= 0
        result = 4*x[0]**2 + x[1]**2 - 3*x[0]*x[1] + 2*x[2]**2 + 5*x[5] - 11*x[6]
        return result

class CEC2006_G10(Problem):
    
    SUPERIOR = np.array([10000, 10000, 10000, 1000, 1000, 1000, 1000, 1000])
    INFERIOR = np.array([100, 1000, 1000, 10, 10, 10, 10, 10])

    def __init__(self):
        rest_g = [self.cec2006_g10_g1, self.cec2006_g10_g2, self.cec2006_g10_g3,
                  self.cec2006_g10_g4, self.cec2006_g10_g5, self.cec2006_g10_g6]
        super().__init__(
            ProblemType.CONSTRAINED,
            self.SUPERIOR, self.INFERIOR,
            rest_g, []
        )
    
    def fitness(self, individuo: np.array) -> float:
        f_x = np.sum(individuo[:3])
        return f_x

    @staticmethod
    def cec2006_g10_g1(x):
        return -1 + 0.0025 * (x[3] + x[5])

    @staticmethod
    def cec2006_g10_g2(x):
        return -1 + 0.0025 * (x[4] + x[6] - x[3])

    @staticmethod
    def cec2006_g10_g3(x):
        return -1 + 0.01 * (x[7] - x[4])

    @staticmethod
    def cec2006_g10_g4(x):
        return -x[0] * x[5] + 833.33252 * x[3] + 100 * x[0] - 83333.333

    @staticmethod
    def cec2006_g10_g5(x):
        return -x[1] * x[6] + 1250 * x[4] + x[1] * x[3] - 1250 * x[3]

    @staticmethod
    def cec2006_g10_g6(x):
        return -x[2] * x[7] + 1250000 + x[2] * x[4] - 2500 * x[4]


class CEC2006_G11(Problem):
    
    SUPERIOR = np.array([1, 1])
    INFERIOR = np.array([-1, -1])

    def __init__(self):
        rest_h = [self.cec2006_g11_h1]
        super().__init__(
            ProblemType.CONSTRAINED,
            self.SUPERIOR, self.INFERIOR,
            [], rest_h
        )
    
    def fitness(self, individuo: np.array) -> float:
        f_x = individuo[0] ** 2 + (individuo[1] - 1) ** 2
        return f_x

    @staticmethod
    def cec2006_g11_h1(x):
        return x[1] - x[0] ** 2


class CEC2006_G12(Problem):
    
    SUPERIOR = np.array([10, 10, 10])
    INFERIOR = np.array([0, 0, 0])

    def __init__(self):
        rest_g = [self.cec2006_g12_g1]
        super().__init__(
            ProblemType.CONSTRAINED,
            self.SUPERIOR, self.INFERIOR,
            rest_g, []
        )
    
    def fitness(self, individuo: np.array) -> float:
        f_x = -(100 - (individuo[0] - 5) ** 2 - (individuo[1] - 5) ** 2 - (individuo[2] - 5) ** 2) / 100
        return f_x

    @staticmethod
    def cec2006_g12_g1(x):
        p, q, r = 1, 2, 3  # Example values, adjust according to the specific case
        return (x[0] - p) ** 2 + (x[1] - q) ** 2 + (x[2] - r) ** 2 - 0.0625


class CEC2006_G13(Problem):
    
    SUPERIOR = np.array([2.3, 2.3, 3.2, 3.2, 3.2])
    INFERIOR = np.array([-2.3, -2.3, -3.2, -3.2, -3.2])

    def __init__(self):
        rest_h = [self.cec2006_g13_h1, self.cec2006_g13_h2, self.cec2006_g13_h3]
        super().__init__(
            ProblemType.CONSTRAINED,
            self.SUPERIOR, self.INFERIOR,
            [], rest_h
        )
    
    def fitness(self, individuo: np.array) -> float:
        f_x = np.exp(np.prod(individuo))
        return f_x

    @staticmethod
    def cec2006_g13_h1(x):
        return np.sum(x ** 2) - 10

    @staticmethod
    def cec2006_g13_h2(x):
        return x[1] * x[2] - 5 * x[3] * x[4]

    @staticmethod
    def cec2006_g13_h3(x):
        return x[0] ** 3 + x[1] ** 3 + 1


class CEC2006_G14(Problem):
    
    SUPERIOR = np.array([10] * 10)
    INFERIOR = np.array([0] * 10)

    def __init__(self):
        rest_h = [self.cec2006_g14_h1, self.cec2006_g14_h2, self.cec2006_g14_h3]
        super().__init__(
            ProblemType.CONSTRAINED,
            self.SUPERIOR, self.INFERIOR,
            [], rest_h
        )
    
    def fitness(self, individuo: np.array) -> float:
        ci = np.array([-6.089, -17.164, -34.054, -5.914, -24.721, -14.986, -24.1, -10.708, -26.662, -22.179])
        f_x = np.sum(individuo * (ci + np.log(individuo / np.sum(individuo))))
        return f_x

    @staticmethod
    def cec2006_g14_h1(x):
        return x[0] + 2 * x[1] + 2 * x[2] + x[5] + x[9] - 2

    @staticmethod
    def cec2006_g14_h2(x):
        return x[3] + 2 * x[4] + x[5] + x[6] - 1

    @staticmethod
    def cec2006_g14_h3(x):
        return x[2] + x[6] + x[7] + 2 * x[8] + x[9] - 1


class CEC2006_G15(Problem):
    
    SUPERIOR = np.array([10, 10, 10])
    INFERIOR = np.array([0, 0, 0])

    def __init__(self):
        rest_h = [self.cec2006_g15_h1, self.cec2006_g15_h2]
        super().__init__(
            ProblemType.CONSTRAINED,
            self.SUPERIOR, self.INFERIOR,
            [], rest_h
        )
    
    def fitness(self, individuo: np.array) -> float:
        f_x = 1000 - individuo[0] ** 2 - 2 * individuo[1] ** 2 - individuo[2] ** 2 - individuo[0] * individuo[1] - individuo[0] * individuo[2]
        return f_x

    @staticmethod
    def cec2006_g15_h1(x):
        return x[0] ** 2 + x[1] ** 2 + x[2] ** 2 - 25

    @staticmethod
    def cec2006_g15_h2(x):
        return 8 * x[0] + 14 * x[1] + 7 * x[2] - 56

class CEC2006_G16(Problem):
    
    SUPERIOR = np.array([906.3855, 288.88, 134.75, 287.0966, 84.1988])
    INFERIOR = np.array([704.4148, 68.6, 0, 193, 25])

    def __init__(self):
        rest_g = [
            self.cec2006_g16_g1, self.cec2006_g16_g2, self.cec2006_g16_g3,
            self.cec2006_g16_g4, self.cec2006_g16_g5, self.cec2006_g16_g6,
            self.cec2006_g16_g7, self.cec2006_g16_g8, self.cec2006_g16_g9,
            self.cec2006_g16_g10, self.cec2006_g16_g11, self.cec2006_g16_g12,
            self.cec2006_g16_g13, self.cec2006_g16_g14, self.cec2006_g16_g15,
            self.cec2006_g16_g16, self.cec2006_g16_g17, self.cec2006_g16_g18,
            self.cec2006_g16_g19, self.cec2006_g16_g20, self.cec2006_g16_g21,
            self.cec2006_g16_g22, self.cec2006_g16_g23, self.cec2006_g16_g24,
            self.cec2006_g16_g25, self.cec2006_g16_g26, self.cec2006_g16_g27,
            self.cec2006_g16_g28, self.cec2006_g16_g29, self.cec2006_g16_g30,
            self.cec2006_g16_g31, self.cec2006_g16_g32, self.cec2006_g16_g33,
            self.cec2006_g16_g34, self.cec2006_g16_g35, self.cec2006_g16_g36,
            self.cec2006_g16_g37, self.cec2006_g16_g38
        ]
        super().__init__(
            ProblemType.CONSTRAINED,
            self.SUPERIOR, self.INFERIOR,
            rest_g, []
        )
    
    def fitness(self, individuo: np.array) -> float:
        y = self.compute_y(individuo)
        f_x = (
            0.000117 * y[13] + 0.1365 + 0.00002358 * y[12] + 0.000001502 * y[15] + 0.0321 * y[11]
            + 0.004324 * y[4] + 0.0001 * y[14] + 37.48 * y[1] - 0.0000005843 * y[16]
        )
        return f_x

    @staticmethod
    def compute_y(x):
        y = np.zeros(17)
        y[0] = x[1] + x[2] + 41.6
        y[1] = 0.024 * x[3] - 4.62
        y[2] = 12.5 / y[1] + 12
        y[3] = 0.0003535 * (x[0] ** 2) + 0.5311 * x[0] + 0.08705 * y[2] * x[0]
        y[4] = 0.052 * x[0] + 78 + 0.002377 * y[2] * x[0]
        y[5] = y[3] / y[4]
        y[6] = 19 * y[5]
        y[7] = 0.04782 * (x[0] - y[5]) + 0.1956 * ((x[0] - y[5]) ** 2) / x[1] + 0.6376 * y[6] + 1.594 * y[5]
        y[8] = 100 * x[1]
        y[9] = x[0] - y[5] - y[6]
        y[10] = 0.950 - y[7] / y[8]
        y[11] = y[9] * y[10]
        y[12] = x[0] - y[11] - y[6] - y[5]
        y[13] = (y[11] + y[6]) * 0.995
        y[14] = y[13] / y[0]
        y[15] = y[13] / 3798
        y[16] = y[14] - 0.0663 * y[14] / y[15] - 0.3153
        return y

    @staticmethod
    def cec2006_g16_g1(x):
        return 0.28 * x[4] - 0.72 * x[3]

    @staticmethod
    def cec2006_g16_g2(x):
        return x[2] - 1.5 * x[1]

    @staticmethod
    def cec2006_g16_g3(x):
        return 3496 * x[1] - 21

    @staticmethod
    def cec2006_g16_g4(x):
        return 110.6 + x[0] - 62212 * x[2]

    @staticmethod
    def cec2006_g16_g5(x):
        return 213.1 - x[0]

    @staticmethod
    def cec2006_g16_g6(x):
        return x[0] - 405.23

    @staticmethod
    def cec2006_g16_g7(x):
        return 17.505 - x[1]

    @staticmethod
    def cec2006_g16_g8(x):
        return x[1] - 1053.6667

    @staticmethod
    def cec2006_g16_g9(x):
        return 11.275 - x[2]

    @staticmethod
    def cec2006_g16_g10(x):
        return x[2] - 35.03

    @staticmethod
    def cec2006_g16_g11(x):
        return 214.228 - x[3]

    @staticmethod
    def cec2006_g16_g12(x):
        return x[3] - 665.585

    @staticmethod
    def cec2006_g16_g13(x):
        return 7.458 - x[4]

    @staticmethod
    def cec2006_g16_g14(x):
        return x[4] - 584.463

    @staticmethod
    def cec2006_g16_g15(x):
        return 0.961 - x[5]

    @staticmethod
    def cec2006_g16_g16(x):
        return x[5] - 265.916

    @staticmethod
    def cec2006_g16_g17(x):
        return 1.612 - x[6]

    @staticmethod
    def cec2006_g16_g18(x):
        return x[6] - 7.046

    @staticmethod
    def cec2006_g16_g19(x):
        return 0.146 - x[7]

    @staticmethod
    def cec2006_g16_g20(x):
        return x[7] - 0.222

    @staticmethod
    def cec2006_g16_g21(x):
        return 107.99 - x[8]

    @staticmethod
    def cec2006_g16_g22(x):
        return x[8] - 273.366

    @staticmethod
    def cec2006_g16_g23(x):
        return 922.693 - x[9]

    @staticmethod
    def cec2006_g16_g24(x):
        return x[9] - 1286.105

    @staticmethod
    def cec2006_g16_g25(x):
        return 926.832 - x[10]

    @staticmethod
    def cec2006_g16_g26(x):
        return x[10] - 1444.046

    @staticmethod
    def cec2006_g16_g27(x):
        return 18.766 - x[11]

    @staticmethod
    def cec2006_g16_g28(x):
        return x[11] - 537.141

    @staticmethod
    def cec2006_g16_g29(x):
        return 1072.163 - x[12]

    @staticmethod
    def cec2006_g16_g30(x):
        return x[12] - 3247.039

    @staticmethod
    def cec2006_g16_g31(x):
        return 8961.448 - x[13]

    @staticmethod
    def cec2006_g16_g32(x):
        return x[13] - 26844.086

    @staticmethod
    def cec2006_g16_g33(x):
        return 0.063 - x[14]

    @staticmethod
    def cec2006_g16_g34(x):
        return x[14] - 0.386

    @staticmethod
    def cec2006_g16_g35(x):
        return 71084.33 - x[15]

    @staticmethod
    def cec2006_g16_g36(x):
        return -140000 + x[15]

    @staticmethod
    def cec2006_g16_g37(x):
        return 2802713 - x[16]

    @staticmethod
    def cec2006_g16_g38(x):
        return x[16] - 12146108


class CEC2006_G17(Problem):
    
    SUPERIOR = np.array([400, 1000, 420, 420, 1000, 0.5236])
    INFERIOR = np.array([0, 0, 340, 340, -1000, 0])

    def __init__(self):
        rest_h = [
            self.cec2006_g17_h1, self.cec2006_g17_h2, self.cec2006_g17_h3,
            self.cec2006_g17_h4
        ]
        super().__init__(
            ProblemType.CONSTRAINED,
            self.SUPERIOR, self.INFERIOR,
            [], rest_h
        )
    
    def fitness(self, individuo: np.array) -> float:
        f_x = self.f1(individuo[0]) + self.f2(individuo[1])
        return f_x

    @staticmethod
    def f1(x1):
        if 0 <= x1 < 300:
            return 30 * x1
        elif 300 <= x1 < 400:
            return 31 * x1

    @staticmethod
    def f2(x2):
        if 0 <= x2 < 100:
            return 28 * x2
        elif 100 <= x2 < 200:
            return 29 * x2
        elif 200 <= x2 < 1000:
            return 30 * x2

    @staticmethod
    def cec2006_g17_h1(x):
        return -x[0] + 300 - x[2] * x[3] / 131.078 * np.cos(1.48477 - x[5]) + 0.90798 * x[2] ** 2 / 131.078 * np.cos(1.47588)

    @staticmethod
    def cec2006_g17_h2(x):
        return -x[1] - x[2] * x[3] / 131.078 * np.cos(1.48477 + x[5]) + 0.90798 * x[3] ** 2 / 131.078 * np.cos(1.47588)

    @staticmethod
    def cec2006_g17_h3(x):
        return -x[4] - x[2] * x[3] / 131.078 * np.sin(1.48477 + x[5]) + 0.90798 * x[3] ** 2 / 131.078 * np.sin(1.47588)

    @staticmethod
    def cec2006_g17_h4(x):
        return 200 - x[2] * x[3] / 131.078 * np.sin(1.48477 - x[5]) + 0.90798 * x[2] ** 2 / 131.078 * np.sin(1.47588)


class CEC2006_G18(Problem):
    
    SUPERIOR = np.array([10, 10, 10, 10, 10, 10, 10, 10, 20])
    INFERIOR = np.array([-10, -10, -10, -10, -10, -10, -10, -10, 0])

    def __init__(self):
        rest_g = [
            self.cec2006_g18_g1, self.cec2006_g18_g2, self.cec2006_g18_g3,
            self.cec2006_g18_g4, self.cec2006_g18_g5, self.cec2006_g18_g6,
            self.cec2006_g18_g7, self.cec2006_g18_g8, self.cec2006_g18_g9,
            self.cec2006_g18_g10, self.cec2006_g18_g11, self.cec2006_g18_g12,
            self.cec2006_g18_g13
        ]
        super().__init__(
            ProblemType.CONSTRAINED,
            self.SUPERIOR, self.INFERIOR,
            rest_g, []
        )
    
    def fitness(self, individuo: np.array) -> float:
        f_x = -0.5 * (individuo[0] * individuo[3] - individuo[1] * individuo[2] + individuo[2] * individuo[8] - individuo[4] * individuo[8] + individuo[4] * individuo[7] - individuo[5] * individuo[6])
        return f_x

    @staticmethod
    def cec2006_g18_g1(x):
        return x[2] ** 2 + x[3] ** 2 - 1

    @staticmethod
    def cec2006_g18_g2(x):
        return x[8] ** 2 - 1

    @staticmethod
    def cec2006_g18_g3(x):
        return x[4] ** 2 + x[5] ** 2 - 1

    @staticmethod
    def cec2006_g18_g4(x):
        return x[0] ** 2 + (x[1] - x[8]) ** 2 - 1

    @staticmethod
    def cec2006_g18_g5(x):
        return (x[0] - x[4]) ** 2 + (x[1] - x[5]) ** 2 - 1

    @staticmethod
    def cec2006_g18_g6(x):
        return (x[0] - x[6]) ** 2 + (x[1] - x[7]) ** 2 - 1

    @staticmethod
    def cec2006_g18_g7(x):
        return (x[2] - x[4]) ** 2 + (x[3] - x[5]) ** 2 - 1

    @staticmethod
    def cec2006_g18_g8(x):
        return (x[2] - x[6]) ** 2 + (x[3] - x[7]) ** 2 - 1

    @staticmethod
    def cec2006_g18_g9(x):
        return x[6] ** 2 + (x[7] - x[8]) ** 2 - 1

    @staticmethod
    def cec2006_g18_g10(x):
        return x[1] * x[2] - x[0] * x[3]

    @staticmethod
    def cec2006_g18_g11(x):
        return -x[2] * x[8]

    @staticmethod
    def cec2006_g18_g12(x):
        return x[4] * x[8]

    @staticmethod
    def cec2006_g18_g13(x):
        return x[5] * x[6] - x[4] * x[7]


class CEC2006_G19(Problem):
    
    SUPERIOR = np.array([10] * 15)
    INFERIOR = np.array([0] * 15)

    def __init__(self):
        rest_g = [self.cec2006_g19_g1, self.cec2006_g19_g2, self.cec2006_g19_g3, self.cec2006_g19_g4, self.cec2006_g19_g5]
        super().__init__(
            ProblemType.CONSTRAINED,
            self.SUPERIOR, self.INFERIOR,
            rest_g, []
        )
    
    def fitness(self, individuo: np.array) -> float:
        c = np.array([
            [30, -20, -10, 32, -10],
            [-20, 39, -6, -31, 32],
            [-10, -6, 10, -6, -10],
            [32, -31, -6, 39, -20],
            [-10, 32, -10, -20, 30]
        ])
        d = np.array([4, 8, 10, 6, 2])
        b = np.array([-40, -2, -0.25, -4, -4, -1, -40, -60, 5, 1])
        sum_c = np.sum([c[i][j] * individuo[10+i] * individuo[10+j] for i in range(5) for j in range(5)])
        sum_d = np.sum([d[j] * (individuo[10+j] ** 3) for j in range(5)])
        f_x = sum_c + 2 * sum_d - np.sum(b * individuo[:10])
        return f_x

    @staticmethod
    def cec2006_g19_g1(x):
        return -2 * np.sum([30 * x[10+i] for i in range(5)]) - 3 * 4 * (x[10+0] ** 2) - 15 + 10 * x[0]

    @staticmethod
    def cec2006_g19_g2(x):
        return -2 * np.sum([39 * x[10+i] for i in range(5)]) - 3 * 8 * (x[10+1] ** 2) - 27 + 2 * x[1]

    @staticmethod
    def cec2006_g19_g3(x):
        return -2 * np.sum([10 * x[10+i] for i in range(5)]) - 3 * 10 * (x[10+2] ** 2) - 36 + 2 * x[2]

    @staticmethod
    def cec2006_g19_g4(x):
        return -2 * np.sum([31 * x[10+i] for i in range(5)]) - 3 * 6 * (x[10+3] ** 2) - 18 + 0.4 * x[3]

    @staticmethod
    def cec2006_g19_g5(x):
        return -2 * np.sum([32 * x[10+i] for i in range(5)]) - 3 * 2 * (x[10+4] ** 2) - 12 + 2 * x[4]


class CEC2006_G20(Problem):
    
    SUPERIOR = np.array([10] * 24)
    INFERIOR = np.array([0] * 24)

    def __init__(self):
        rest_g = [
            self.cec2006_g20_g1, self.cec2006_g20_g2, self.cec2006_g20_g3,
            self.cec2006_g20_g4, self.cec2006_g20_g5, self.cec2006_g20_g6
        ]
        rest_h = [
            self.cec2006_g20_h1, self.cec2006_g20_h2, self.cec2006_g20_h3,
            self.cec2006_g20_h4, self.cec2006_g20_h5, self.cec2006_g20_h6,
            self.cec2006_g20_h7, self.cec2006_g20_h8, self.cec2006_g20_h9,
            self.cec2006_g20_h10, self.cec2006_g20_h11, self.cec2006_g20_h12,
            self.cec2006_g20_h13, self.cec2006_g20_h14
        ]
        super().__init__(
            ProblemType.CONSTRAINED,
            self.SUPERIOR, self.INFERIOR,
            rest_g, rest_h
        )
    
    def fitness(self, individuo: np.array) -> float:
        a = np.array([
            0.0693, 0.0577, 0.05, 0.2, 0.26, 0.55, 0.06, 0.1, 0.12, 0.18, 0.1, 0.09,
            0.0693, 0.0577, 0.05, 0.2, 0.26, 0.55, 0.06, 0.1, 0.12, 0.18, 0.1, 0.09
        ])
        f_x = np.sum(a * individuo)
        return f_x

    @staticmethod
    def cec2006_g20_g1(x):
        return (x[0] + x[12]) / np.sum(x) + 0.1

    @staticmethod
    def cec2006_g20_g2(x):
        return (x[1] + x[13]) / np.sum(x) + 0.3

    @staticmethod
    def cec2006_g20_g3(x):
        return (x[2] + x[14]) / np.sum(x) + 0.4

    @staticmethod
    def cec2006_g20_g4(x):
        return (x[3] + x[15]) / np.sum(x) + 0.3

    @staticmethod
    def cec2006_g20_g5(x):
        return (x[4] + x[16]) / np.sum(x) + 0.6

    @staticmethod
    def cec2006_g20_g6(x):
        return (x[5] + x[17]) / np.sum(x) + 0.3

    @staticmethod
    def cec2006_g20_h1(x):
        return (x[12] / 44.094 + x[13] / 58.12 + x[14] / 58.12 + x[15] / 137.4 + x[16] / 120.9 + x[17] / 170.9
                + x[18] / 62.501 + x[19] / 84.94 + x[20] / 133.425 + x[21] / 82.507 + x[22] / 46.07 + x[23] / 60.097
                - x[0] / 0.0693 - x[1] / 0.0577 - x[2] / 0.05 - x[3] / 0.2 - x[4] / 0.26 - x[5] / 0.55
                - x[6] / 0.06 - x[7] / 0.1 - x[8] / 0.12 - x[9] / 0.18 - x[10] / 0.1 - x[11] / 0.09)

    @staticmethod
    def cec2006_g20_h2(x):
        return np.sum(x) - 1

    @staticmethod
    def cec2006_g20_h3(x):
        return (x[12] / 44.094 + x[13] / 58.12 + x[14] / 58.12 + x[15] / 137.4 + x[16] / 120.9 + x[17] / 170.9
                + x[18] / 62.501 + x[19] / 84.94 + x[20] / 133.425 + x[21] / 82.507 + x[22] / 46.07 + x[23] / 60.097
                - x[0] / 0.0693 - x[1] / 0.0577 - x[2] / 0.05 - x[3] / 0.2 - x[4] / 0.26 - x[5] / 0.55
                - x[6] / 0.06 - x[7] / 0.1 - x[8] / 0.12 - x[9] / 0.18 - x[10] / 0.1 - x[11] / 0.09)

    @staticmethod
    def cec2006_g20_h4(x):
        return np.sum(x) - 1.671

    @staticmethod
    def cec2006_g20_h5(x):
        return (x[12] / 44.094 + x[13] / 58.12 + x[14] / 58.12 + x[15] / 137.4 + x[16] / 120.9 + x[17] / 170.9
                + x[18] / 62.501 + x[19] / 84.94 + x[20] / 133.425 + x[21] / 82.507 + x[22] / 46.07 + x[23] / 60.097
                - x[0] / 0.0693 - x[1] / 0.0577 - x[2] / 0.05 - x[3] / 0.2 - x[4] / 0.26 - x[5] / 0.55
                - x[6] / 0.06 - x[7] / 0.1 - x[8] / 0.12 - x[9] / 0.18 - x[10] / 0.1 - x[11] / 0.09)

    @staticmethod
    def cec2006_g20_h6(x):
        return np.sum(x) - 1

    @staticmethod
    def cec2006_g20_h7(x):
        return (x[12] / 44.094 + x[13] / 58.12 + x[14] / 58.12 + x[15] / 137.4 + x[16] / 120.9 + x[17] / 170.9
                + x[18] / 62.501 + x[19] / 84.94 + x[20] / 133.425 + x[21] / 82.507 + x[22] / 46.07 + x[23] / 60.097
                - x[0] / 0.0693 - x[1] / 0.0577 - x[2] / 0.05 - x[3] / 0.2 - x[4] / 0.26 - x[5] / 0.55
                - x[6] / 0.06 - x[7] / 0.1 - x[8] / 0.12 - x[9] / 0.18 - x[10] / 0.1 - x[11] / 0.09)

    @staticmethod
    def cec2006_g20_h8(x):
        return np.sum(x) - 1

    @staticmethod
    def cec2006_g20_h9(x):
        return (x[12] / 44.094 + x[13] / 58.12 + x[14] / 58.12 + x[15] / 137.4 + x[16] / 120.9 + x[17] / 170.9
                + x[18] / 62.501 + x[19] / 84.94 + x[20] / 133.425 + x[21] / 82.507 + x[22] / 46.07 + x[23] / 60.097
                - x[0] / 0.0693 - x[1] / 0.0577 - x[2] / 0.05 - x[3] / 0.2 - x[4] / 0.26 - x[5] / 0.55
                - x[6] / 0.06 - x[7] / 0.1 - x[8] / 0.12 - x[9] / 0.18 - x[10] / 0.1 - x[11] / 0.09)

    @staticmethod
    def cec2006_g20_h10(x):
        return np.sum(x) - 1

    @staticmethod
    def cec2006_g20_h11(x):
        return (x[12] / 44.094 + x[13] / 58.12 + x[14] / 58.12 + x[15] / 137.4 + x[16] / 120.9 + x[17] / 170.9
                + x[18] / 62.501 + x[19] / 84.94 + x[20] / 133.425 + x[21] / 82.507 + x[22] / 46.07 + x[23] / 60.097
                - x[0] / 0.0693 - x[1] / 0.0577 - x[2] / 0.05 - x[3] / 0.2 - x[4] / 0.26 - x[5] / 0.55
                - x[6] / 0.06 - x[7] / 0.1 - x[8] / 0.12 - x[9] / 0.18 - x[10] / 0.1 - x[11] / 0.09)

    @staticmethod
    def cec2006_g20_h12(x):
        return np.sum(x) - 1

    @staticmethod
    def cec2006_g20_h13(x):
        return (x[12] / 44.094 + x[13] / 58.12 + x[14] / 58.12 + x[15] / 137.4 + x[16] / 120.9 + x[17] / 170.9
                + x[18] / 62.501 + x[19] / 84.94 + x[20] / 133.425 + x[21] / 82.507 + x[22] / 46.07 + x[23] / 60.097
                - x[0] / 0.0693 - x[1] / 0.0577 - x[2] / 0.05 - x[3] / 0.2 - x[4] / 0.26 - x[5] / 0.55
                - x[6] / 0.06 - x[7] / 0.1 - x[8] / 0.12 - x[9] / 0.18 - x[10] / 0.1 - x[11] / 0.09)

    @staticmethod
    def cec2006_g20_h14(x):
        return np.sum(x) - 1.671


class CEC2006_G21(Problem):
    
    SUPERIOR = np.array([1000, 40, 40, 300, 6.7, 6.4, 6.25])
    INFERIOR = np.array([0, 0, 0, 100, 6.3, 5.9, 4.5])

    def __init__(self):
        rest_g = [self.cec2006_g21_g1]
        rest_h = [
            self.cec2006_g21_h1, self.cec2006_g21_h2, self.cec2006_g21_h3,
            self.cec2006_g21_h4, self.cec2006_g21_h5
        ]
        super().__init__(
            ProblemType.CONSTRAINED,
            self.SUPERIOR, self.INFERIOR,
            rest_g, rest_h
        )
    
    def fitness(self, individuo: np.array) -> float:
        f_x = individuo[0]
        return f_x

    @staticmethod
    def cec2006_g21_g1(x):
        return -x[0] + 35 * (x[1] ** 0.6) + 35 * (x[2] ** 0.6)

    @staticmethod
    def cec2006_g21_h1(x):
        return -300 * x[2] + 7500 * x[4] - 7500 * x[5] - 25 * x[3] * x[4] + 25 * x[3] * x[5] + x[2] * x[3]

    @staticmethod
    def cec2006_g21_h2(x):
        return 100 * x[1] + 155.365 * x[3] + 2500 * x[6] - x[1] * x[3] - 25 * x[3] * x[6] - 15536.5

    @staticmethod
    def cec2006_g21_h3(x):
        return -x[4] + np.log(-x[3] + 900)

    @staticmethod
    def cec2006_g21_h4(x):
        return -x[5] + np.log(x[3] + 300)

    @staticmethod
    def cec2006_g21_h5(x):
        return -x[6] + np.log(-2 * x[3] + 700)


class CEC2006_G22(Problem):
    
    SUPERIOR = np.array([20000, 1e6, 1e6, 1e6, 4e7, 4e7, 4e7, 299.99, 399.99, 300, 400, 600, 500, 500, 500, 300, 400, 6.25, 6.25, 6.25, 6.25, 6.25])
    INFERIOR = np.array([0, 0, 0, 0, 0, 0, 0, 100, 100, 100.01, 100, 100, 0, 0, 0, 0.01, 0.01, -4.7, -4.7, -4.7, -4.7, -4.7])

    def __init__(self):
        rest_g = [self.cec2006_g22_g1]
        rest_h = [
            self.cec2006_g22_h1, self.cec2006_g22_h2, self.cec2006_g22_h3,
            self.cec2006_g22_h4, self.cec2006_g22_h5, self.cec2006_g22_h6,
            self.cec2006_g22_h7, self.cec2006_g22_h8, self.cec2006_g22_h9,
            self.cec2006_g22_h10, self.cec2006_g22_h11, self.cec2006_g22_h12,
            self.cec2006_g22_h13, self.cec2006_g22_h14, self.cec2006_g22_h15,
            self.cec2006_g22_h16, self.cec2006_g22_h17, self.cec2006_g22_h18,
            self.cec2006_g22_h19
        ]
        super().__init__(
            ProblemType.CONSTRAINED,
            self.SUPERIOR, self.INFERIOR,
            rest_g, rest_h
        )
    
    def fitness(self, individuo: np.array) -> float:
        f_x = individuo[0]
        return f_x

    @staticmethod
    def cec2006_g22_g1(x):
        return -x[0] + 0.0193 * x[3]

    @staticmethod
    def cec2006_g22_h1(x):
        return x[8] - 300 - 100 * x[11] + x[12]

    @staticmethod
    def cec2006_g22_h2(x):
        return x[9] - 400 - 100 * x[10] + x[13]

    @staticmethod
    def cec2006_g22_h3(x):
        return -x[14] - x[11] + 600

    @staticmethod
    def cec2006_g22_h4(x):
        return -x[15] - x[10] + 300

    @staticmethod
    def cec2006_g22_h5(x):
        return x[16] - 500 - 100 * x[10] + x[14]

    @staticmethod
    def cec2006_g22_h6(x):
        return x[17] - 500 - 100 * x[10] + x[14]

    @staticmethod
    def cec2006_g22_h7(x):
        return x[18] - 500 - 100 * x[10] + x[14]

    @staticmethod
    def cec2006_g22_h8(x):
        return x[19] - 500 - 100 * x[10] + x[14]

    @staticmethod
    def cec2006_g22_h9(x):
        return x[20] - 500 - 100 * x[10] + x[14]

    @staticmethod
    def cec2006_g22_h10(x):
        return x[21] - 500 - 100 * x[10] + x[14]

    @staticmethod
    def cec2006_g22_h11(x):
        return x[0] + x[6] - 600

    @staticmethod
    def cec2006_g22_h12(x):
        return x[1] + x[7] - 600

    @staticmethod
    def cec2006_g22_h13(x):
        return x[2] + x[4] - 600

    @staticmethod
    def cec2006_g22_h14(x):
        return x[3] + x[5] - 600

    @staticmethod
    def cec2006_g22_h15(x):
        return x[4] - 600 - 300 * x[5]

    @staticmethod
    def cec2006_g22_h16(x):
        return x[5] - 600 - 300 * x[4]

    @staticmethod
    def cec2006_g22_h17(x):
        return x[6] - 600 - 300 * x[7]

    @staticmethod
    def cec2006_g22_h18(x):
        return x[7] - 600 - 300 * x[6]

    @staticmethod
    def cec2006_g22_h19(x):
        return -x[4] - 600 * x[6] + x[5]


class CEC2006_G23(Problem):
    
    SUPERIOR = np.array([300, 300, 100, 200, 100, 300, 100, 200, 100, 300])
    INFERIOR = np.array([0] * 10)

    def __init__(self):
        rest_h = [
            self.cec2006_g23_h1, self.cec2006_g23_h2, self.cec2006_g23_h3,
            self.cec2006_g23_h4, self.cec2006_g23_h5, self.cec2006_g23_h6,
            self.cec2006_g23_h7, self.cec2006_g23_h8
        ]
        super().__init__(
            ProblemType.CONSTRAINED,
            self.SUPERIOR, self.INFERIOR,
            [], rest_h
        )
    
    def fitness(self, individuo: np.array) -> float:
        f_x = individuo[0]
        return f_x

    @staticmethod
    def cec2006_g23_h1(x):
        return -x[0] + 300 * x[1]

    @staticmethod
    def cec2006_g23_h2(x):
        return -x[1] + 300 * x[2]

    @staticmethod
    def cec2006_g23_h3(x):
        return -x[2] + 300 * x[3]

    @staticmethod
    def cec2006_g23_h4(x):
        return -x[3] + 300 * x[4]

    @staticmethod
    def cec2006_g23_h5(x):
        return -x[4] + 300 * x[5]

    @staticmethod
    def cec2006_g23_h6(x):
        return -x[5] + 300 * x[6]

    @staticmethod
    def cec2006_g23_h7(x):
        return -x[6] + 300 * x[7]

    @staticmethod
    def cec2006_g23_h8(x):
        return -x[7] + 300 * x[8]


class CEC2006_G24(Problem):
    
    SUPERIOR = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1])
    INFERIOR = np.array([0] * 9)

    def __init__(self):
        rest_h = [
            self.cec2006_g24_h1, self.cec2006_g24_h2, self.cec2006_g24_h3,
            self.cec2006_g24_h4, self.cec2006_g24_h5, self.cec2006_g24_h6,
            self.cec2006_g24_h7, self.cec2006_g24_h8, self.cec2006_g24_h9
        ]
        super().__init__(
            ProblemType.CONSTRAINED,
            self.SUPERIOR, self.INFERIOR,
            [], rest_h
        )
    
    def fitness(self, individuo: np.array) -> float:
        f_x = -np.sum(individuo)
        return f_x

    @staticmethod
    def cec2006_g24_h1(x):
        return -x[0] + x[1] - x[2] + x[3] - x[4] + x[5] - x[6] + x[7] - x[8] + 9

    @staticmethod
    def cec2006_g24_h2(x):
        return x[0] - x[1] - x[2] + x[3] + x[4] - x[5] + x[6] - x[7] + x[8]

    @staticmethod
    def cec2006_g24_h3(x):
        return -x[0] - x[1] + x[2] + x[3] + x[4] + x[5] - x[6] + x[7] + x[8]

    @staticmethod
    def cec2006_g24_h4(x):
        return x[0] + x[1] + x[2] + x[3] - x[4] - x[5] + x[6] - x[7] + x[8]

    @staticmethod
    def cec2006_g24_h5(x):
        return x[0] - x[1] + x[2] - x[3] + x[4] - x[5] - x[6] + x[7] + x[8]

    @staticmethod
    def cec2006_g24_h6(x):
        return x[0] - x[1] + x[2] - x[3] + x[4] - x[5] + x[6] - x[7] - x[8]

    @staticmethod
    def cec2006_g24_h7(x):
        return -x[0] + x[1] + x[2] + x[3] - x[4] + x[5] - x[6] + x[7] - x[8]

    @staticmethod
    def cec2006_g24_h8(x):
        return x[0] + x[1] - x[2] - x[3] + x[4] - x[5] + x[6] + x[7] - x[8]

    @staticmethod
    def cec2006_g24_h9(x):
        return x[0] - x[1] - x[2] + x[3] - x[4] + x[5] - x[6] + x[7] + x[8]
