import matplotlib.pyplot as plt
import os
import pandas as pd
from utils.utility import ensure_directory_exists, generate_random_filename
from utils.save_file import load_results_from_json

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

def plot_convergence_from_json(problem_name, directory, problem_prefix):
    data = load_results_from_json(problem_name, directory, problem_prefix)
    
    if data is None:
        return
    
    plt.figure(figsize=(12, 8))
    
    line_styles = ['-', '--', '-.', ':', '-', '--']
    markers = ['o', 's', 'D', '^', 'v', 'x']
    colors = ['b', 'g', 'r', 'c', 'm', 'y']
    
    for i, (constraint_name, values) in enumerate(data.items()):
        plt.plot(values, label=constraint_name, linestyle=line_styles[i % len(line_styles)], 
                 marker=markers[i % len(markers)], color=colors[i % len(colors)], linewidth=2, markersize=6)

    plt.title(f"Convergence Plot (Violations) for {problem_name}", fontsize=16)
    plt.xlabel("Generations", fontsize=14)
    plt.ylabel("Violations", fontsize=14)
    plt.yscale('log')  # Añade escala logarítmica si las violaciones varían mucho
    plt.legend(fontsize=12)
    plt.grid(True, which="both", ls="--", linewidth=0.5)
    
    graphics_directory = f"{directory}/graphics/convergence_violations_all_methods"
    ensure_directory_exists(graphics_directory)
    filename = f"{graphics_directory}/{problem_prefix}_convergence_violations_{problem_name}_from_json.png"
    plt.savefig(filename, bbox_inches='tight', dpi=300)
    plt.show()
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
    directory = f"{directory}/graphics/custom/fitness"
    ensure_directory_exists(directory)
    plt.savefig(
        f"{directory}/{problem_prefix}_fitness_boxplot_{problem_name}_{generate_random_filename()}.png"
    )
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
    directory = f"{directory}/graphics/custom/violations"
    ensure_directory_exists(directory)
    plt.savefig(
        f"{directory}/{problem_prefix}_violations_boxplot_{problem_name}_{generate_random_filename()}.png"
    )
    plt.show()
    plt.close()

