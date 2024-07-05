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
                problema.fitness,
                ConstriantsFunctionsHandler.a_is_better_than_b_deb,
                bounds_constraints=bounds[constraint_name],
                bounds=(problema.SUPERIOR, problema.INFERIOR),
                g_functions=problema.rest_g,
                h_functions=problema.rest_h,
                centroid_method=(constraint_name == "centroid"),
                centroid_repair_method=(constraint_name == "centroid_repair"),
                adaptive_centroid=(constraint_name == "adaptive_centroid"),
                beta_method=(constraint_name == "beta"),
                evolutionary_method=(constraint_name == "evolutionary"),
                resrand_method=(constraint_name == "res&rand"),
                ADS=(constraint_name == "ADS"),
            )

            # No mostrar datos
            algorithm.evolution(verbose=True)

            # Obteniendo datos necesarios
            fitness_data.append(algorithm.gbest_fitness)
            violations_data.append(algorithm.gbest_violation)

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



def calculate_average_violations(root_folder, problem_prefix):
    average_violations = {}

    # Iterar sobre todas las restricciones (subcarpetas en root_folder)
    for restriction_folder in glob.glob(os.path.join(root_folder, "*")):
        if os.path.isdir(restriction_folder):
            restriction_name = os.path.basename(restriction_folder)
            violation_counts = []

            # Iterar sobre todos los archivos CSV dentro de la carpeta de restricción
            for csv_file in glob.glob(
                os.path.join(restriction_folder, f"{problem_prefix}*.csv")
            ):
                df = pd.read_csv(csv_file)
                # Obtener las infracciones y agregarlas a la lista
                violation_counts.extend(df["Violations"])

            if violation_counts:
                # Calcular el promedio de infracciones
                average_violations[restriction_name] = sum(violation_counts) / len(
                    violation_counts
                )
            else:
                average_violations[restriction_name] = 0

    return average_violations


def generate_summary_xlsx(root_folder, problems, output_file):
    summary_data = []

    for problem_prefix in problems:
        avg_violations = calculate_average_violations(root_folder, problem_prefix)
        avg_violations["Problema"] = problem_prefix
        summary_data.append(avg_violations)

    # Crear un DataFrame de los datos resumidos
    summary_df = pd.DataFrame(summary_data)
    # Reordenar columnas para que 'Problema' sea la primera columna
    columns_order = ["Problema"] + [
        col for col in summary_df.columns if col != "Problema"
    ]
    summary_df = summary_df[columns_order]
    # Guardar el DataFrame en un archivo Excel
    summary_df.to_csv(output_file, index=False)
    print(f"Archivo Excel generado: {output_file}")
