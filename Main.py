# Limites y restricciones
from bounds_handler import BoundaryHandler
from contraints_functions import ConstriantsFunctionsHandler

# Estrategias de velocidad
from utils.constants import ITERATIONS

from utils.generate_csv import *

# Funciones objetivas
from functions.cec2006problems import *
from functions.cec2020problems import *
from functions.cec2022problems import *

from differential_evolution import Differential_Evolution

problema = CEC2006_G01()
resultados = []
def main():       
    for i in range(1, ITERATIONS + 1):  # Asegúrate de que ITERATIONS está bien definido
        print(f"Ejecución: {i}")     
        de = Differential_Evolution(
            problema.fitness,
            ConstriantsFunctionsHandler.a_is_better_than_b_deb,
            BoundaryHandler.reflex,
            (problema.SUPERIOR, problema.INFERIOR),
            problema.rest_g,
            problema.rest_h,
            strategy= 'rand1'
        )
        de.evolution()
        resultado = {
                'Ejecucion': i,
                'Fitness': de.best_fitness,
                'Violaciones': de.best_violations
            }

        guardar_resultados_csv(resultado, 'rand1.csv')
    
if __name__ == "__main__":
    main()