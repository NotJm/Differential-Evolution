import gc
from boundary_handler import BoundaryHandler
from contraints_functions import ConstriantsFunctionsHandler

from utils.constants import EXECUTIONS
from utils.convergencia import generate_convergencia_mean_graphic
from utils.calculate_mean import calculate_mean_boxes_constraints

# Funciones objetivas
from functions.cec2020problems import *

from differential_evolution import Differential_Evolution

def run():
    problema = CEC2020_RC01()
    
    results_generate_lampiden = []
    results_generate_centroid = []

    for _ in range(1, EXECUTIONS + 1):
        print(f"Execution {_}:")
        
        # Ejecutar Differential Evolution sin centroid
        diffential_evolution = Differential_Evolution(
            problema.fitness,
            ConstriantsFunctionsHandler.a_is_better_than_b_deb,
            BoundaryHandler.periodic_mode,
            (problema.superior, problema.inferior),
            problema.rest_g,
            problema.rest_h,
            centroide=False
        )
        diffential_evolution.evolution(verbose=True)
        results_generate_lampiden.append(diffential_evolution.solutions_generate)
        
        # Liberar memoria innecesaria
        del diffential_evolution
        gc.collect()
        
        # Ejecutar Differential Evolution con centroid
        differentia_Evolution_ceontroide = Differential_Evolution(
            problema.fitness,
            ConstriantsFunctionsHandler.a_is_better_than_b_deb,
            BoundaryHandler.calculate_centroid,
            (problema.superior, problema.inferior),
            problema.rest_g,
            problema.rest_h,
            centroide=True
        )
        
        differentia_Evolution_ceontroide.evolution(verbose=True)
        results_generate_centroid.append(differentia_Evolution_ceontroide.solutions_generate)
        
        # Liberar memoria innecesaria
        del differentia_Evolution_ceontroide
        gc.collect()

    # Calcular las medias para cada método
    mean, mean_centroide = calculate_mean_boxes_constraints(results_generate_lampiden, results_generate_centroid)
     
    # Generar la gráfica de convergencia
    generate_convergencia_mean_graphic(
        ["Zhang Handling Periodic Mode Constraint", "Juarez Efren Centroid Constraint"],  # Nombres de los métodos
        "Zhang vs Juarez Efren.png",  # Nombre del archivo de la gráfica
        "Zhang Handling Periodic Mode Constraint vs Juarez Efren Centroid Constraint",  # Título de la gráfica
        "Generations",  # Etiqueta del eje x
        "Violations",  # Etiqueta del eje y
        (mean, mean_centroide)  # Medias de los métodos
    )

if __name__ == "__main__":
    run() 
