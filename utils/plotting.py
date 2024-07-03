import matplotlib.pyplot as plt
import os
import pandas as pd
from utils.utility import ensure_directory_exists

def plot_convergence_fitness(
    problem_name, constraint_name, convergence_fitness_data, directory, problem_prefix
):
    plt.figure(figsize=(10, 6))
    mean_convergence_fitness = [
        sum(gen) / len(gen) for gen in zip(*convergence_fitness_data)
    ]
    plt.plot(mean_convergence_fitness, label=constraint_name)
    plt.title(
        f"Grafica de Convergencia (Fitness) para {problem_name} : BCHM {constraint_name}"
    )
    plt.xlabel("Generaciones")
    plt.ylabel("Fitness")
    plt.legend()
    plt.grid(True)
    directory = f"{directory}/{constraint_name}/{problem_name}/convergence_fitness"
    ensure_directory_exists(directory)
    filename = f"{directory}/{problem_prefix}_convergence_fitness_{problem_name}_{constraint_name}.png"
    plt.savefig(filename)
    plt.close()


def plot_fitness_boxplot(
    problem_name, constraint_name, factible_fitness_data, directory, problem_prefix
):
    plt.figure(figsize=(12, 8))
    plt.boxplot(factible_fitness_data, labels=[constraint_name])
    plt.title(f"Box Plot de Fitness para {problem_name} : BCHM {constraint_name}")
    plt.xlabel("Restricción")
    plt.ylabel("Fitness")
    plt.xticks(rotation=45)
    plt.grid(True)
    directory = f"{directory}/{constraint_name}/{problem_name}/boxplot_fitness"
    ensure_directory_exists(directory)
    filename = f"{directory}/{problem_prefix}_fitness_boxplot_{problem_name}_{constraint_name}.png"
    plt.savefig(filename)
    plt.close()

def plot_convergence_violations_all(
    problem_name, 
    all_convergence_data, 
    bounds, 
    directory, 
    problem_prefix
):
    plt.figure(figsize=(10, 6))
    for constraint_name in bounds:
        if constraint_name in all_convergence_data:
            mean_convergence_violations = [
                sum(gen) / len(gen)
                for gen in zip(*all_convergence_data[constraint_name])
            ]
            plt.plot(mean_convergence_violations, label=constraint_name)

    plt.title(f"Convergence Plot (Violations) for {problem_name}")
    plt.xlabel("Generations")
    plt.ylabel("Violations")
    plt.legend()
    plt.grid(True)

    directory = f"{directory}/convergence_violations_all"
    ensure_directory_exists(directory)
    filename = f"{directory}/{problem_prefix}_convergence_violations_{problem_name}.png"
    plt.savefig(filename)
    plt.close()

def plot_violations_boxplot_all(
    problem_name,
    all_violations_data,
    directory,
    problem_prefix
):
    plt.figure(figsize=(12, 8))
    data = [violations for violations in all_violations_data.values()]
    labels = [constraint for constraint in all_violations_data.keys()]
    plt.boxplot(data, labels=labels)
    plt.title(f"Box Plot de Violations para {problem_name} del {problem_prefix}")
    plt.xlabel("Restricción")
    plt.ylabel("Violations")
    plt.xticks(rotation=45)
    plt.grid(True)
    directory = f"{directory}/graphics/violations_all"
    ensure_directory_exists(directory)
    filename = f"{directory}/{problem_prefix}_boxplot_violations_{problem_name}.png"
    plt.savefig(filename)
    plt.close()


def plot_fitness_boxplot_all(
    problem_name,
    all_fitness_data,
    directory,
    problem_prefix
):
    plt.figure(figsize=(12, 8))
    data = [fitness for fitness in all_fitness_data.values()]
    labels = [constraint for constraint in all_fitness_data.keys()]
    plt.boxplot(data, labels=labels)
    plt.title(f"Box Plot de Fitness para {problem_name} del {problem_prefix}")
    plt.xlabel("Restricción")
    plt.ylabel("Fitness")
    plt.xticks(rotation=45)
    plt.grid(True)
    directory = f"{directory}/graphics/fitness_all"
    ensure_directory_exists(directory)
    filename = f"{directory}/{problem_prefix}_fitness_boxplot_{problem_name}.png"
    plt.savefig(filename)
    plt.close()
    
def plot_fitness_boxplot_from_csvs(directory, problem_prefix, problem_name, exclude=[]):
    fitness_data = {}
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".csv") and problem_name in file:
                constraint_name = root.split(os.sep)[
                    -1
                ]  # Obtener el nombre de la restricción
                if constraint_name not in exclude:
                    file_path = os.path.join(root, file)
                    df = pd.read_csv(file_path)
                    if "Fitness" in df.columns:
                        fitness_data[constraint_name] = df["Fitness"].dropna().values

    if not fitness_data:
        print(
            f"No se encontraron datos de fitness para el problema {problem_name} con las restricciones especificadas."
        )
        return

    # Ordenar los datos por el valor medio del fitness
    sorted_fitness_data = sorted(fitness_data.items(), key=lambda x: x[1].mean())

    # Desempaquetar los datos ordenados
    sorted_labels, sorted_data = zip(*sorted_fitness_data)

    plt.figure(figsize=(12, 8))
    plt.boxplot(sorted_data, labels=sorted_labels)
    plt.title(f"Box Plot de Fitness para {problem_name}")
    plt.xlabel("Restricción")
    plt.ylabel("Fitness")
    plt.xticks(rotation=45)
    plt.grid(True)
    exclude_str = ",".join(exclude)
    # plt.savefig(
    #     f"{directory}/{problem_prefix}_fitness_boxplot_{problem_name}_({exclude_str}).png"
    # )
    plt.show()
    plt.close()


def plot_violations_boxplot_from_csvs(directory, problem_prefix, problem_name, exclude=[]):
    violations_data = {}
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".csv") and problem_name in file:
                constraint_name = root.split(os.sep)[
                    -1
                ]  # Obtener el nombre de la restricción
                if constraint_name not in exclude:
                    file_path = os.path.join(root, file)
                    df = pd.read_csv(file_path)
                    if "Violations" in df.columns:
                        violations_data[constraint_name] = (
                            df["Violations"].dropna().values
                        )

    if not violations_data:
        print(
            f"No se encontraron datos de violaciones para el problema {problem_name} con las restricciones especificadas."
        )
        return

    # Ordenar los datos por el valor medio de las violaciones
    sorted_violations_data = sorted(violations_data.items(), key=lambda x: x[1].mean())

    # Desempaquetar los datos ordenados
    sorted_labels, sorted_data = zip(*sorted_violations_data)

    plt.figure(figsize=(12, 8))
    plt.boxplot(sorted_data, labels=sorted_labels)
    plt.title(f"Box Plot de Violations para {problem_name}")
    plt.xlabel("Restricción")
    plt.ylabel("Violations")
    plt.xticks(rotation=45)
    plt.grid(True)
    exclude_str = ",".join(exclude)
    plt.savefig(
        f"{directory}/{problem_prefix}_violations_boxplot_{problem_name}_({exclude_str}).png"
    )
    plt.show()
    plt.close()

