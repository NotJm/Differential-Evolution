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


problema = CEC2017_C02()
strategies = ['combined_100%','combined_95%', 'combined_90%', 'combined_85%', 'combined_80%', 'combined_75%', 'combined_70%', 'combined_65%', 'combined_60%', 'combined_55%', 'combined_50%',
               'combined_45%','combined_40%', 'combined_35%','combined_30%', 'combined_25%', 'combined_20%', 'combined_15%', 'combined_10%', 'combined_05%', 'combined_0%',]

def main():
    
    resultados = {strategy: [] for strategy in strategies}
    
    for strategy in strategies:
        for i in range(1, EXECUTIONS + 1):
            print(f"Ejecución: {i} con estrategia: {strategy}")
            de = Differential_Evolution(
                problema.fitness,
                ConstriantsFunctionsHandler.a_is_better_than_b_deb,
                BoundaryHandler.mirror,
                (SUPERIOR_1, INFERIOR_1),
                problema.rest_g,
                problema.rest_h,
                strategy=strategy,
                centroid = False
            )
            de.evolution()
            resultados[strategy].append(de.best_fitness)
    problem = "CEC2017_C02"
    limite = "Mirror"
    plot_box_plot(resultados, problem, limite)
    #plot_radar_chart(resultados)
    
def plot_box_plot(data, problema, restriccion):
    # Crear un box plot con los datos recolectados
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.boxplot(data.values())
    
    # Extraer solo los porcentajes de las estrategias
    percentages = [strategy.split('_')[1] for strategy in data.keys()]
    
    ax.set_xticklabels(percentages, fontsize=10)
    plt.suptitle('Comparación de Estrategias de Mutación (Fitness)', size=20, color='blue')
    ax.set_title(f'Problema: {problema}, Restricción de límite: {restriccion}', fontsize=12, color='gray')
    ax.set_ylabel('Fitness')
    ax.set_xlabel('Estrategias de Mutación combined')

    plt.show()


    
    
if __name__ == "__main__":
    main()