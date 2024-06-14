import pandas as pd
import gc
import numpy as np
from multiprocessing import Pool
from boundary_handler import BoundaryHandler
from constraints_functions import ConstriantsFunctionsHandler
from utils.constants import EXECUTIONS
from differential_evolution import Differential_Evolution
from functions.cec2020problems import *
from functions.cec2006problems import *

problems = {
    "R01": CEC2020_RC01,
    "R02": CEC2020_RC02,
    "R03": CEC2020_RC03,
    "R06": CEC2020_RC06,
}

fitness_bounds = {
    "R01": [200, 1000],
    "R02": [6000, 9000],
    "R03": [-5000, -4000],
    "R06": [1.5, 2]
}

bounds = {
    "juarez-centroid": BoundaryHandler.centroid_method,
    "adham-velocity-clamp": BoundaryHandler.adham_clamp_position,
    "adham-shink": BoundaryHandler.adham_shrink_position,
    "andreaa-saturation": BoundaryHandler.andreaa_saturation,
    "andreaa-mirror": BoundaryHandler.andreaa_mirror,
    "andreaa-uniform": BoundaryHandler.andreaa_uniform,
    "andreaa-nearest": BoundaryHandler.andreaa_nearest,
    "andreaa-random-within-bounds": BoundaryHandler.andreaa_random_within_bounds,
    "agarwl-reflection": BoundaryHandler.agarwl_reflect,
    "agarwl-nearest": BoundaryHandler.agarwl_nearest,
    "shi_classical": BoundaryHandler.shi_classical_boundary_handling,
}

def execute_experiment(args):
    problem_name, problem_class, execution, key, boundary_function = args
    problema = problem_class()
    lower_bound, upper_bound = fitness_bounds[problem_name]
    
    try:
        algorithm = Differential_Evolution(
            problema.fitness,
            ConstriantsFunctionsHandler.a_is_better_than_b_deb,
            bounds_constraints=boundary_function,
            bounds=(problema.superior, problema.inferior),
            g_functions=problema.rest_g,
            h_functions=problema.rest_h,
            centroid=(key == "juarez-centroid")
        )

        algorithm.evolution(verbose=False)
        
        if lower_bound <= algorithm.gbest_fitness <= upper_bound:
            fitness = algorithm.gbest_fitness
        else:
            fitness = np.nan
            print(f"Fitness fuera de los límites: {algorithm.gbest_fitness}")

    except Exception as e:
        print(f"Error con constraint {key} en ejecución {execution + 1}: {e}")
        fitness = np.nan

    del algorithm
    gc.collect()
    return (problem_name, key, fitness)

def run():
    all_results = []
    precision_results = []
    tasks = []

    for problem_name, problem_class in problems.items():
        for key, boundary_function in bounds.items():
            for execution in range(EXECUTIONS):
                tasks.append((problem_name, problem_class, execution, key, boundary_function))

    with Pool() as pool:
        results = pool.map(execute_experiment, tasks)

    results_dict = {key: [] for key in bounds.keys()}
    
    for problem_name, key, fitness in results:
        if not np.isnan(fitness):
            results_dict[key].append(fitness)
            all_results.append(
                {
                    "Problema": problem_name,
                    "Restriccion": key,
                    "Fitness": fitness,
                }
            )
    
    for key, fitness_list in results_dict.items():
        if fitness_list:
            avg_fitness = np.mean(fitness_list)
            precision = np.std(fitness_list)
            precision_results.append(
                {
                    "Problema": problem_name,
                    "Restriccion": key,
                    "Precision": precision,
                }
            )

    # Crear DataFrame y guardar en CSV
    df_results = pd.DataFrame(all_results)
    df_results.to_csv("results_combined.csv", index=False)
    print("Resultados de fitness guardados en 'results_combined.csv'")

    df_precision = pd.DataFrame(precision_results)
    df_precision.to_csv("precision_results.csv", index=False)
    print("Resultados de precisión guardados en 'precision_results.csv'")

    # Imprimir resultados de precisión en la consola
    for result in precision_results:
        print(f"Problema: {result['Problema']}, Restriccion: {result['Restriccion']}, Precision: {result['Precision']}")

if __name__ == "__main__":
    run()
