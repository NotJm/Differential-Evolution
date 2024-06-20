# Limites y restricciones
from bounds_handler import BoundaryHandler
from contraints_functions import ConstriantsFunctionsHandler

# Estrategias de velocidad
from utils.constants import ITERATIONS, SIZE_POPULATION

from utils.generate_csv import *
from utils.radar import *
from utils.boxplot import *

# Funciones objetivas
from functions.cec2006problems import *
from functions.cec2020problems import *
from functions.cec2022problems import *

from differential_evolution import Differential_Evolution
import matplotlib.pyplot as plt
import numpy as np

problema = CEC2020_RC05()
strategies = ['combined_95%', 'combined_90%', 'combined_85%', 'combined_80%', 'combined_75%', 'combined_70%'  ]

def main():
    
    resultados = {strategy: [] for strategy in strategies}
    
    for strategy in strategies:
        for i in range(1, ITERATIONS + 1):
            print(f"Ejecución: {i} con estrategia: {strategy}")
            de = Differential_Evolution(
                problema.fitness,
                ConstriantsFunctionsHandler.a_is_better_than_b_deb,
                BoundaryHandler.reflex,
                (problema.SUPERIOR, problema.INFERIOR),
                problema.rest_g,
                problema.rest_h,
                strategy=strategy
            )
            de.evolution()
            resultados[strategy].append(de.best_fitness)
    problem = "CEC2020_RC05"
    limite = "reflex"
    plot_box_plot(resultados, problem, limite)
    #plot_radar_chart(resultados)
    


    
    
if _name_ == "_main_":
    main()