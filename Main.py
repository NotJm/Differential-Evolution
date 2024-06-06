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
        BoundaryHandler.repeat_unitl_within_bounds,
        (problema.SUPERIOR, problema.INFERIOR),
        problema.rest_g,
        problema.rest_h
    )
    de.evolution()
    

if __name__ == "__main__":
    main()
