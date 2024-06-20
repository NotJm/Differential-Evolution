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

from core.differential_evolution import Differential_Evolution
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

problema = CEC2017_C01()
strategies = ['combined_100%', 'combined_95%', 'combined_90%', 'combined_85%', 'combined_80%', 'combined_75%', 'combined_70%', 'combined_65%', 'combined_60%', 'combined_55%', 'combined_50%',
              'combined_45%', 'combined_40%', 'combined_35%', 'combined_30%', 'combined_25%', 'combined_20%', 'combined_15%', 'combined_10%', 'combined_05%', 'combined_0%']

def main():
    resultados = {strategy: [] for strategy in strategies}
    
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
        
        # Filtrar ejecuciones factibles
        factible_indices = [i for i, v in enumerate(violations_data) if v == 0]
        factible_fitness_data = [fitness_data[i] for i in factible_indices]
        
        # Calcular promedio de fitness para ejecuciones factibles
        if factible_fitness_data:
            average_fitness = sum(factible_fitness_data) / len(factible_fitness_data)
        else:
            average_fitness = "N/A"

        resultados[strategy] = factible_fitness_data
    
    problem = "CEC2017_C01"
    limite = "Centroid"
    plot_box_plot(resultados, problem, limite)
    save_results_to_excel(resultados, problem, limite)
    # plot_radar_chart(resultados)
    
def plot_box_plot(data, problema, restriccion):
    # Crear un box plot con los datos recolectados
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.boxplot(data.values())
    
    # Extraer solo los porcentajes de las estrategias
    percentages = [strategy.split('_')[1] for strategy in data.keys()]
    
    ax.set_xticklabels(percentages, fontsize=10)
    plt.suptitle('Comparación de porcentajes de las Estrategias de Mutación (Fitness)', size=20, color='blue')
    ax.set_title(f'Problema: {problema}, Restricción de límite: {restriccion}', fontsize=12, color='gray')
    ax.set_ylabel('Fitness')
    ax.set_xlabel('Estrategias de Mutación combined')

    plt.show()

def save_results_to_excel(resultados, problema, restriccion):
    # Preparar los datos para el DataFrame
    rows = []
    for strategy, fitnesses in resultados.items():
        for fitness in fitnesses:
            rows.append([problema, restriccion, strategy, fitness])
    
    # Crear el DataFrame
    df = pd.DataFrame(rows, columns=["Problema", "Manejo de limites", "Estrategia", "Fitness"])
    
    # Guardar el DataFrame en un archivo Excel
    df.to_excel("resultados_experimentacion.xlsx", index=False)
    print("Resultados guardados en 'resultados_experimentacion.xlsx'")

if __name__ == "__main__":
    main()
