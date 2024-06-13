
import pandas as pd
import gc
import numpy as np
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
    # "juarez-centroide": BoundaryHandler.juarez_centroide,
    "adham-clamp": BoundaryHandler.adham_clamp_position,
    "adham-shink": BoundaryHandler.adham_shrink_position,
    "andreaa-saturation": BoundaryHandler.andreaa_saturation,
    "andreaa-mirror": BoundaryHandler.andreaa_mirror,
    "andreaa-uniform": BoundaryHandler.andreaa_uniform,
    "andreaa-nearest": BoundaryHandler.andreaa_nearest,
    "andreaa-random-within-bounds": BoundaryHandler.andreaa_random_within_bounds,
}

def check_for_pause():
    """
    Function to pause the execution when Ctrl+P is pressed.
    """
    if keyboard.is_pressed('ctrl') and keyboard.is_pressed('p'):
        print("Paused. Press Ctrl+R to resume.")
        while True:
            if keyboard.is_pressed('ctrl') and keyboard.is_pressed('r'):
                print("Resuming...")
                break

def run():
    all_results = []

    for problem_name, problem_class in problems.items():
        problema = problem_class()
        lower_bound, upper_bound = fitness_bounds[problem_name]
        results = {
            # "juarez-centroide": np.zeros((EXECUTIONS, 2)),
            "adham-clamp": np.zeros((EXECUTIONS, 2)),
            "adham-shink": np.zeros((EXECUTIONS, 2)),
            "andreaa-saturation": np.zeros((EXECUTIONS, 2)),
            "andreaa-mirror": np.zeros((EXECUTIONS, 2)),
            "andreaa-uniform": np.zeros((EXECUTIONS, 2)),
            "andreaa-nearest": np.zeros((EXECUTIONS, 2)),
            "andreaa-random-within-bounds": np.zeros((EXECUTIONS, 2)),
            "andreaa-shr": np.zeros((EXECUTIONS, 2)),
        }

        for _ in range(EXECUTIONS):
            print(f"Execution {_ + 1} for problem {problem_name}:")
            for key, boundary_function in bounds.items():
                print(f"Constraint {key}:")
                
                
                
                algorithm = Differential_Evolution(
                    problema.fitness,
                    ConstriantsFunctionsHandler.a_is_better_than_b_deb,
                    bounds_constraints=(
                        BoundaryHandler.mirror
                        if key == "juarez-centroide"
                        else boundary_function
                    ),
                    bounds=(problema.superior, problema.inferior),
                    g_functions=problema.rest_g,
                    h_functions=problema.rest_h,
                    centroide=(key == "juarez-centroide"),
                    w_p_function=(
                        BoundaryHandler.juarez_W_p
                        if key == "juarez-centroide"
                        else None
                    ),
                    centroide_function=(
                        BoundaryHandler.juarez_centroide
                        if key == "juarez-centroide"
                        else None
                    ),
                )

                algorithm.evolution(verbose=True)
                
                if lower_bound <= algorithm.gbest_fitness <= upper_bound:
                    results[key][_] = algorithm.gbest_fitness

                del algorithm
                gc.collect()

        # Guardar los resultados en el formato deseado
        for key, result in results.items():
            for fitness in result:
                all_results.append(
                    {
                        "Problema": problem_name,
                        "Restriccion": key,
                        "Fitness": fitness[0],
                    }
                )

    # Crear DataFrame y guardar en CSV
    df_results = pd.DataFrame(all_results)
    df_results.to_csv("results_combined.csv", index=False)
    print("Resultados guardados en 'results_combined.csv'")


if __name__ == "__main__":
    run()
