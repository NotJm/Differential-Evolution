import pandas as pd
import gc
import numpy as np
import inquirer
import os
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate
from core.boundary_handler import BoundaryHandler
from core.constraints_functions import ConstriantsFunctionsHandler
from utils.constants import EXECUTIONS
from core.differential_evolution import Differential_Evolution
from functions.cec2020problems import *
from functions.cec2006problems import *

problems = {
    "R01": CEC2020_RC01,
    "R02": CEC2020_RC02,
    "R06": CEC2020_RC06,
}

fitness_bounds = {
    "R01": [200, 1000],
    "R02": [6000, 9000],
    "R06": [1, 2]
}

bounds = {
    "adham-velocity-clamp": BoundaryHandler.adham_clamp_position, #
    "adham-shink": BoundaryHandler.adham_shrink_position,# 
    "shi_classical": BoundaryHandler.shi_classical_boundary_handling, #
    "agarwl-reflection": BoundaryHandler.agarwl_reflect, #
    "agarwl-nearest": BoundaryHandler.agarwl_nearest, #
    "juarez-centroid": BoundaryHandler.centroid_method, 
    "andreaa-saturation": BoundaryHandler.andreaa_saturation, #
    "andreaa-mirror": BoundaryHandler.andreaa_mirror, #
    "andreaa-uniform": BoundaryHandler.andreaa_uniform, #
    "andreaa-nearest": BoundaryHandler.andreaa_nearest, #
    "andreaa-random-within-bounds": BoundaryHandler.andreaa_random_within_bounds, #
    "andreaa-beta": BoundaryHandler.andreaa_beta,
    "andreaa-vector-wise-correction": BoundaryHandler.andreaa_vector_wise_correction
}

def run():
    all_results = []
    precision_results = []

    for problem_name, problem_class in problems.items():
        problema = problem_class()
        lower_bound, upper_bound = fitness_bounds[problem_name]
        results = {key: {'fitness': [], 'violations': []} for key in bounds.keys()}

        for _ in range(EXECUTIONS):
            print(f"Execution {_ + 1} for problem {problem_name}:")
            for key, boundary_function in bounds.items():
                print(f"Constraint {key}:")
                try:
                    algorithm = Differential_Evolution(
                        problema.fitness,
                        ConstriantsFunctionsHandler.a_is_better_than_b_deb,
                        bounds_constraints=boundary_function,
                        bounds=(problema.superior, problema.inferior),
                        g_functions=problema.rest_g,
                        h_functions=problema.rest_h,
                        centroid=(key == "juarez-centroid"),
                        beta=(key == "andreaa-beta")
                    )

                    algorithm.evolution(verbose=True)
                    
                    if lower_bound <= algorithm.gbest_fitness <= upper_bound:
                        results[key]['fitness'].append(algorithm.gbest_fitness)
                        results[key]['violations'].append(algorithm.gbest_violation)
                    else:
                        print(f"Fitness fuera de los límites: {algorithm.gbest_fitness}")

                except Exception as e:
                    print(f"Error con constraint {key} en ejecución {_ + 1}: {e}")

                del algorithm
                gc.collect()

        # Guardar los resultados en el formato deseado
        for key, result in results.items():
            valid_fitness = [f for f in result['fitness'] if not np.isnan(f)]
            valid_violations = [v for v in result['violations'] if not np.isnan(v)]
            if valid_fitness:
                avg_fitness = np.mean(valid_fitness)
                avg_violations = np.mean(valid_violations)
                precision = np.std(valid_fitness)  # Calcula la precisión como desviación estándar
                precision_results.append(
                    {
                        "Problema": problem_name,
                        "Restriccion": key,
                        "Precision": precision,
                        "Promedio_Fitness": avg_fitness,
                        "Promedio_Violaciones": avg_violations,
                    }
                )
                for fitness, violation in zip(valid_fitness, valid_violations):
                    all_results.append(
                        {
                            "Problema": problem_name,
                            "Restriccion": key,
                            "Fitness": fitness,
                            "Violaciones": violation,
                        }
                    )

    # Crear DataFrame y guardar en CSV
    df_results = pd.DataFrame(all_results)
    df_results.to_csv("results_combined.csv", index=False)
    print("Resultados de fitness y violaciones guardados en 'results_combined.csv'")

    df_precision = pd.DataFrame(precision_results)
    df_precision.to_csv("precision_results.csv", index=False)
    print("Resultados de precisión guardados en 'precision_results.csv'")

    # Imprimir resultados de precisión en la consola
    for result in precision_results:
        print(f"Problema: {result['Problema']}, Restriccion: {result['Restriccion']}, Precision: {result['Precision']}, Promedio Fitness: {result['Promedio_Fitness']}, Promedio Violaciones: {result['Promedio_Violaciones']}")

def interactive_run():
    questions = [
        inquirer.List('action',
                      message="Seleccione la acción a realizar",
                      choices=['Ejecutar un problema con una restricción', 'Ejecutar todos los problemas con todas las restricciones']),
    ]

    answers = inquirer.prompt(questions)

    if answers['action'] == 'Ejecutar un problema con una restricción':
        questions = [
            inquirer.List('problem',
                          message="Seleccione el problema a ejecutar",
                          choices=list(problems.keys())),
            inquirer.List('constraint',
                          message="Seleccione la restricción de manejo de límites",
                          choices=list(bounds.keys()))
        ]

        answers = inquirer.prompt(questions)
        problem_name = answers['problem']
        constraint_name = answers['constraint']
        boundary_function = bounds[constraint_name]

        run_single(problem_name, constraint_name, boundary_function)
    
    elif answers['action'] == 'Ejecutar todos los problemas con todas las restricciones':
        run()

def run_single(problem_name, constraint_name, boundary_function):
    problema = problems[problem_name]()
    lower_bound, upper_bound = fitness_bounds[problem_name]
    results = {'fitness': [], 'violations': []}

    for _ in range(EXECUTIONS):
        print(f"Execution {_ + 1} for problem {problem_name} with constraint {constraint_name}:")
        try:
            algorithm = Differential_Evolution(
                problema.fitness,
                ConstriantsFunctionsHandler.a_is_better_than_b_deb,
                bounds_constraints=boundary_function,
                bounds=(problema.superior, problema.inferior),
                g_functions=problema.rest_g,
                h_functions=problema.rest_h,
                centroid=(constraint_name == "juarez-centroid"),
                beta=(constraint_name == "andreaa-beta")
            )

            algorithm.evolution(verbose=True)

            if lower_bound <= algorithm.gbest_fitness <= upper_bound:
                results['fitness'].append(algorithm.gbest_fitness)
                results['violations'].append(algorithm.gbest_violation)
            else:
                print(f"Fitness fuera de los límites: {algorithm.gbest_fitness}")

        except Exception as e:
            print(f"Error con constraint {constraint_name} en ejecución {_ + 1}: {e}")

        del algorithm
        gc.collect()

    valid_fitness = [f for f in results['fitness'] if not np.isnan(f)]
    valid_violations = [v for v in results['violations'] if not np.isnan(v)]
    if valid_fitness:
        avg_fitness = np.mean(valid_fitness)
        avg_violations = np.mean(valid_violations)
        precision = np.std(valid_fitness)  # Calcula la precisión como desviación estándar

        # Crear la ruta de directorios si no existe
        directory = f"report/cec2020/{problem_name}/{constraint_name}"
        if not os.path.exists(directory):
            os.makedirs(directory)

        # Guardar resultados en CSV
        df_results = pd.DataFrame({
            "Fitness": valid_fitness,
            "Violaciones": valid_violations
        })
        df_results.to_csv(f"{directory}/results.csv", index=False)

        df_summary = pd.DataFrame({
            "Métrica": ["Promedio de Fitness", "Promedio de Violaciones", "Precisión"],
            "Valor": [avg_fitness, avg_violations, precision]
        })
        df_summary.to_csv(f"{directory}/summary.csv", index=False)

        # Imprimir resultados
        print("\nResultados de Fitness y Precisión:")
        print(tabulate(df_summary, headers='keys', tablefmt='grid'))

if __name__ == "__main__":
    # Definir el directorio base y las restricciones
    base_dir = "report/cec2020/R01"
    restrictions = [
        "adham-velocity-clamp", "adham-shink", "shi_classical",
        "agarwl-reflection", "agarwl-nearest",
        "andreaa-saturation", "andreaa-mirror", "andreaa-uniform",
        "andreaa-nearest", "andreaa-random-within-bounds",
        "andreaa-beta", "andreaa-vector-wise-correction"
    ]

    # Leer los datos desde todas las carpetas de restricciones y combinarlos
    all_data = []
    for restriction in restrictions:
        file_path = os.path.join(base_dir, restriction, "results.csv")
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            df['Restriccion'] = restriction
            all_data.append(df)

    # Combinar todos los datos en un único DataFrame
    combined_df = pd.concat(all_data, ignore_index=True)

    # Crear el box plot
    plt.figure(figsize=(14, 8))
    sns.boxplot(x='Restriccion', y='Fitness', data=combined_df, palette="Set2")
    plt.title('Box Plot de Fitness para R01')
    plt.xticks(rotation=45, ha='right')
    plt.xlabel('Restricción')
    plt.ylabel('Fitness')
    plt.tight_layout()
    
    output_dir = "report/cec2020/R01/plots"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    plt.savefig(os.path.join(output_dir, "boxplot_fitness_R01.png"))
        
    plt.show()
    
    
