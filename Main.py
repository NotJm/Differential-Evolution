# Clases para el manejo de límites y restricciones
from boundary_handler import BoundaryHandler
from constraints_functions import ConstriantsFunctionsHandler

# Funciones Generalizadas
from utils.constants import EXECUTIONS
from utils.generate_csv import save_results_csv_file
from utils.calculate_mean import calculate_mean
# Funciones objetivas
from functions.cec2020problems import *

import pandas as pd
import matplotlib.pyplot as plt
import os

# Algoritmo
from differential_evolution import Differential_Evolution

# Garbage collector
import gc
import numpy as np

problems = {
    "R01": CEC2020_RC01,
    "R02": CEC2020_RC02,
    "R03": CEC2020_RC03,
    "R06": CEC2020_RC06, 
}



bounds = {
    "juarez-centroide": BoundaryHandler.juarez_centroide,
    "adham-clamp": BoundaryHandler.adham_clamp_position,
    "adham-shrink": BoundaryHandler.adham_shrink_position,
    "qin-wrapping": BoundaryHandler.qin_wrapping,
    "qin-reflection": BoundaryHandler.qin_reflection,
    "qin-boundary-repair": BoundaryHandler.qin_boundary_repair
}

def run():
    all_results = []

    for problem_name, problem_class in problems.items():
        problema = problem_class()

        results = {
            "juarez-centroide": np.zeros((EXECUTIONS, 1)),
            "adham-clamp": np.zeros((EXECUTIONS, 1)),
            "adham-shrink": np.zeros((EXECUTIONS, 1)),
            "qin-wrapping": np.zeros((EXECUTIONS, 1)),
            "qin-reflection": np.zeros((EXECUTIONS, 1)),
            "qin-boundary-repair": np.zeros((EXECUTIONS, 1))
        }
        
        for _ in range(EXECUTIONS):
            print(f"Execution {_ + 1} for problem {problem_name}:")
            for key, boundary_function in bounds.items():
                print(f"Constraint {key}:")
                algorithm = Differential_Evolution(
                    problema.fitness,
                    ConstriantsFunctionsHandler.a_is_better_than_b_deb,
                    bounds_constraints=BoundaryHandler.mirror if key == "juarez-centroide" else boundary_function,
                    bounds=(problema.superior, problema.inferior),
                    g_functions=problema.rest_g,
                    h_functions=problema.rest_h,
                    centroide=(key == "juarez-centroide"),
                    w_p_function=BoundaryHandler.juarez_W_p if key == "juarez-centroide" else None,
                    centroide_function=BoundaryHandler.juarez_centroide if key == "juarez-centroide" else None,
                )
                
                algorithm.evolution(verbose=True)
                
               
                del algorithm
                gc.collect()
        
        # Guardar los resultados en el formato deseado
        for key, result in results.items():
            for fitness in result:
                all_results.append({
                    "Problema": problem_name,
                    "Restriccion": key,
                    "Fitness": fitness[0]
                })

    # Crear DataFrame y guardar en CSV
    df_results = pd.DataFrame(all_results)
    df_results.to_csv("results_combined.csv", index=False)
    print("Resultados guardados en 'results_combined.csv'")


if __name__ == "__main__":
    pass
    