# Limites y restricciones
from core.boundary_handler import BoundaryHandler
from core.constraints_functions import ConstriantsFunctionsHandler

# Estrategias de velocidad
from utils.constants import EXECUTIONS, SIZE_POPULATION

# Funciones objetivas
from functions.cec2006problems import *
from functions.cec2020problems import *
from functions.cec2022problems import *
from functions.cec2017problems import *
import os 

from core.differential_evolution import Differential_Evolution
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

problema = CEC2017_C03()
strategies = ['combined_100%', 'combined_95%', 'combined_90%', 'combined_85%', 'combined_80%', 'combined_75%', 'combined_70%', 'combined_65%', 'combined_60%', 'combined_55%', 'combined_50%',
              'combined_45%', 'combined_40%', 'combined_35%', 'combined_30%', 'combined_25%', 'combined_20%', 'combined_15%', 'combined_10%', 'combined_05%', 'combined_0%']

def main():
    problem = "CEC2017_C03"
    limite = "Reflex"
    
    for strategy in strategies:
        fitness_data = []
        violations_data = []
        for i in range(1, EXECUTIONS + 1):
            print(f"Ejecución: {i} con estrategia: {strategy}")
            de = Differential_Evolution(
                problema.fitness,
                ConstriantsFunctionsHandler.a_is_better_than_b_deb,
                BoundaryHandler.agarwl_reflect,
                (SUPERIOR_1, INFERIOR_1),
                problema.rest_g,
                problema.rest_h,
                strategy=strategy,
                centroid=False
            )
            de.evolution()
            fitness_data.append(de.best_fitness)
            violations_data.append(de.best_violations)
        
        # Guardar resultados de la estrategia actual en un archivo Excel
        save_results_to_excel(problem, limite, strategy, fitness_data, violations_data)

def save_results_to_excel(problema, restriccion, estrategia, fitness_data, violations_data):
    # Preparar los datos para el DataFrame
    rows = []
    for fitness, violations in zip(fitness_data, violations_data):
        rows.append([ fitness, violations])
    
    # Crear el DataFrame
    df = pd.DataFrame(rows, columns=["Fitness", "Violaciones"])
    
    # Nombre del archivo basado en problema, restricción y estrategia
    filename = f"{problema}_{restriccion}_{estrategia}.xlsx"
    
    # Guardar el DataFrame en un archivo Excel
    df.to_excel(filename, index=False)
    print(f"Resultados guardados en '{filename}'")

    plt.show()

def plot_fitness_boxplot_from_excels(directory, problem_name, exclude=[]):
    """
    Genera un boxplot de fitness excluyendo ciertas restricciones basándose en los archivos Excel generados.

    Parámetros:
    directory (str): Directorio donde se encuentran los archivos Excel.
    problem_name (str): Nombre del problema a analizar.
    exclude (list): Lista de restricciones a excluir del boxplot.
    """
    fitness_data = {}
    Violaciones = {}
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".xlsx") and problem_name in file:
                # Obtener el nombre de la restricción y el porcentaje combinado
                parts = file.split('_')
                constraint_name = f"{parts[3]}_{parts[4].split('.')[0]}"  # Obtener el nombre de la restricción y el porcentaje combinado
                if constraint_name not in exclude:
                    file_path = os.path.join(root, file)
                    df = pd.read_excel(file_path)
                    if 'Fitness' in df.columns:
                        if constraint_name in fitness_data:
                            fitness_data[constraint_name].extend(df['Fitness'].dropna().values)
                        else:
                            fitness_data[constraint_name] = df['Fitness'].dropna().values.tolist()

    if not fitness_data:
        print(f"No se encontraron datos de fitness para el problema {problem_name} con las restricciones especificadas.")
        return

    # Ordenar los datos por el valor medio del fitness
    sorted_fitness_data = sorted(fitness_data.items(), key=lambda x: pd.Series(x[1]).mean())

    # Desempaquetar los datos ordenados
    sorted_labels, sorted_data = zip(*sorted_fitness_data)

    plt.figure(figsize=(12, 8))
    plt.boxplot(sorted_data, labels=sorted_labels)
    plt.title(f'Box Plot de Fitness para {problem_name}')
    plt.xlabel('Restricción')
    plt.ylabel('Fitness')
    plt.xticks(rotation=45)
    plt.grid(True)
    #plt.yscale('log')
    plt.savefig(f'{directory}/CEC2017_fitness_boxplot_{problem_name}_C03_1.png')
    plt.close()

###########################################
def plot_violations_boxplot_from_excels(directory, problem_name, exclude=[]):
    """
    Genera un boxplot de violaciones excluyendo ciertas restricciones basándose en los archivos Excel generados.

    Parámetros:
    directory (str): Directorio donde se encuentran los archivos Excel.
    problem_name (str): Nombre del problema a analizar.
    exclude (list): Lista de restricciones a excluir del boxplot.
    """
    violations_data = {}
    
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".xlsx") and problem_name in file:
                # Obtener el nombre de la restricción y el porcentaje combinado
                parts = file.split('_')
                constraint_name = f"{parts[3]}_{parts[4].split('.')[0]}"  # Obtener el nombre de la restricción y el porcentaje combinado
                if constraint_name not in exclude:
                    file_path = os.path.join(root, file)
                    df = pd.read_excel(file_path)
                    if 'Violaciones' in df.columns:
                        if constraint_name in violations_data:
                            violations_data[constraint_name].extend(df['Violaciones'].dropna().values)
                        else:
                            violations_data[constraint_name] = df['Violaciones'].dropna().values.tolist()

    if not violations_data:
        print(f"No se encontraron datos de violaciones para el problema {problem_name} con las restricciones especificadas.")
        return

    # Ordenar los datos por el valor medio de las violaciones
    sorted_violations_data = sorted(violations_data.items(), key=lambda x: pd.Series(x[1]).mean())

    # Desempaquetar los datos ordenados
    sorted_labels, sorted_data = zip(*sorted_violations_data)

    plt.figure(figsize=(12, 8))
    plt.boxplot(sorted_data, labels=sorted_labels)
    plt.title(f'Box Plot de Violaciones para {problem_name}')
    plt.xlabel('Restricción')
    plt.ylabel('Violaciones')
    plt.xticks(rotation=45)
    plt.grid(True)
    #plt.yscale('log')
    plt.savefig(f'{directory}/CEC2017_violations_boxplot_{problem_name}_C03_1.png')
    plt.close()
###########################################

if __name__ == "__main__":
    #main()
    # plot_fitness_boxplot_from_excels('C02/', 'CEC2017_C02', exclude=[
    # ])
    plot_violations_boxplot_from_excels('C03/', 'CEC2017_C03', exclude=[
     ])

#'combined_0%', 'combined_05%', 'combined_10%', 'combined_15%', 'combined_20%', 'combined_25%', 'combined_30%', 'combined_35%', 'combined_40%', 'combined_45%','combined_55%', 'combined_100%'
