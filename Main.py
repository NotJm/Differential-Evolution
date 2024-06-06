# Limites y restricciones
from bounds_handler import BoundaryHandler
from contraints_functions import ConstriantsFunctionsHandler

# Estrategias de velocidad
from utils.constants import ITERATIONS

# Funciones objetivas
from functions.cec2006problems import *
from functions.cec2020problems import *
from functions.cec2022problems import *

from differential_evolution import Differential_Evolution

problema = CEC2006_G01()

def main():
    de = Differential_Evolution(
        problema.fitness,
        ConstriantsFunctionsHandler.a_is_better_than_b_deb,
        BoundaryHandler.reflex,
        (problema.SUPERIOR, problema.INFERIOR),
        problema.rest_g,
        problema.rest_h,
        F=0.8,  # Factores de escala, puedes ajustarlo si es necesario
        CR=0.9,  # Tasas de recombinación, puedes ajustarlo si es necesario
        strategy='randtobest1'  # Especifica la estrategia de mutación aquí
    )
    de.evolution()
    
if __name__ == "__main__":
    main()

# randtobest1 con fitness de -14.99999 
# rand1 con fitnees de -14.999
# best2 con fitness menores a -14.55  
# rand2 con fitness entre -13 y -14 
# best1 con fitness entre -12, -13 y -14 
# currenttobest1 con fitness de -12 o de -13 