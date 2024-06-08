from boundary_handler import BoundaryHandler
from contraints_functions import ConstriantsFunctionsHandler

from utils.constants import EXECUTIONS
from utils.convergencia import generate_convergencia_mean_graphic
from utils.calculate_mean import calculate_mean_boxes_constraints

# Funciones objetivas
from functions.cec2020problems import *

from differential_evolution import Differential_Evolution

def main():
    problema = CEC2020_RC01()

    # Listas para obtener los resultados de cada método
    results_periodic_mode = []
    results_reflex = []
    result_projection = []
    result_replace_out_of_bounds = []

    for _ in range(1, EXECUTIONS + 1):
        print(f"Execution {_}:")
        de_periodic_mode = Differential_Evolution(
            problema.fitness,
            ConstriantsFunctionsHandler.a_is_better_than_b_deb,
            BoundaryHandler.periodic_mode,
            (problema.superior, problema.inferior),
            problema.rest_g,
            problema.rest_h,
        )
        de_periodic_mode.evolution(verbose=False)
        results_periodic_mode.append(de_periodic_mode.solutions_generate)

        de_reflex = Differential_Evolution(
            problema.fitness,
            ConstriantsFunctionsHandler.a_is_better_than_b_deb,
            BoundaryHandler.reflex,
            (problema.superior, problema.inferior),
            problema.rest_g,
            problema.rest_h,
        )
        de_reflex.evolution(verbose=False)
        results_reflex.append(de_reflex.solutions_generate)

        # Agregar los métodos restantes
        de_projection = Differential_Evolution(
            problema.fitness,
            ConstriantsFunctionsHandler.a_is_better_than_b_deb,
            BoundaryHandler.reflex,
            (problema.superior, problema.inferior),
            problema.rest_g,
            problema.rest_h,
        )
        de_projection.evolution(verbose=False)
        result_projection.append(de_projection.solutions_generate)

        de_replace_out_of_bounds = Differential_Evolution(
            problema.fitness,
            ConstriantsFunctionsHandler.a_is_better_than_b_deb,
            BoundaryHandler.reflex,
            (problema.superior, problema.inferior),
            problema.rest_g,
            problema.rest_h,
        )
        de_replace_out_of_bounds.evolution(verbose=False)
        result_replace_out_of_bounds.append(de_replace_out_of_bounds.solutions_generate)

    # Calcular las medias para cada método
    mean_periodic_mode, mean_reflex, mean_projection, mean_replace_out_of_bounds = calculate_mean_boxes_constraints(
        results_periodic_mode, results_reflex, result_projection, result_replace_out_of_bounds
    )

    # Generar la gráfica de convergencia
    generate_convergencia_mean_graphic(
        ["Periodic Mode", "Reflex", "Projection", "Replace Out of Bounds"],  # Nombres de los métodos
        "Convergence_All_Methods.png",  # Nombre del archivo de la gráfica
        "Convergence of All Methods",  # Título de la gráfica
        "Generations",  # Etiqueta del eje x
        "Violations",  # Etiqueta del eje y
        (mean_periodic_mode, mean_reflex, mean_projection, mean_replace_out_of_bounds)  # Medias de los métodos
    )

if __name__ == "__main__":
    main()
