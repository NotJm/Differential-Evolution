import gc
import pandas as pd
import os
import glob
from core.differential_evolution import Differential_Evolution
from core.constraints_functions import ConstriantsFunctionsHandler
from utils.constants import EXECUTIONS
from utils.utility import generate_random_filename
from utils.save_file import save_results_to_csv_constraint
from utils.plotting import (
    plot_convergence_fitness,
    plot_fitness_boxplot,
)


def execute_algorithm(
    problem_name,
    problem_class,
    constraint_name,
    bounds,
    directory,
    problem_prefix,
):
    problema = problem_class()
    fitness_data = []
    violations_data = []
    convergence_fitness_data = []
    convergence_violations_data = []

    for _ in range(EXECUTIONS):
        print(
            f"Ejecucion {_ + 1}: Problema {problem_name} del {problem_prefix} | BCHM {constraint_name}:"
        )
        algorithm = None
        try:
            algorithm = Differential_Evolution(
                problem_name,
                problema.fitness,
                bounds_constraints_method=constraint_name,
                constraints_functions=ConstriantsFunctionsHandler.a_is_better_than_b_deb,
                bounds=(problema.SUPERIOR, problema.INFERIOR),
                g_functions=problema.rest_g,
                h_functions=problema.rest_h,
            )

            # No mostrar datos
            algorithm.evolution(verbose=True)

            # Obteniendo datos necesarios
            fitness_data.append(algorithm.gbest_fitness)
            violations_data.append(algorithm.gbest_violation)
            convergence_fitness_data.append(algorithm.gbest_fitness_list)
            convergence_violations_data.append(algorithm.gbest_violations_list)

        except Exception as e:
            print(
                f"Excepcion Encontrada: Ejecucion {_ + 1} | Problema {problem_name} del {problem_prefix} | BCHM {constraint_name}"
            )
            raise e
        finally:
            if algorithm is not None:
                del algorithm
            gc.collect()

    # Filtrando solo por datos factibles
    factible_indices = [i for i, v in enumerate(violations_data) if v == 0]
    factible_fitness_data = [fitness_data[i] for i in factible_indices]

    # Guardar datos en archivos por restricción y problema
    save_results_to_csv_constraint(
        problem_name,
        constraint_name,
        fitness_data,
        violations_data,
        directory,
        problem_prefix,
    )

    return (
        fitness_data,
        violations_data,
        convergence_violations_data,
        factible_fitness_data,
    )
