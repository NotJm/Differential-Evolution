import numpy as np
from matplotlib import pyplot as plt

def update_plot(
    population,
    fitness,
    violations,
    gbest,
    generation,
    scatter_plot_factible,
    scatter_plot_infactible,
    scatter_plot_gbest,
    ax,
):
    """Actualiza la gráfica con la población actual."""
    factibles = population[violations == 0]
    infactibles = population[violations != 0]

    scatter_plot_factible.set_offsets(factibles[:, :2])
    scatter_plot_infactible.set_offsets(infactibles[:, :2])
    scatter_plot_gbest.set_offsets(gbest[:, :2])

    ax.set_title(f"Generación: {generation}")
    plt.draw()
    plt.pause(0.1)


def setup_plot(bounds):
    """Configura la gráfica inicial."""
    plt.ion()
    fig, ax = plt.subplots()
    lower_bounds, upper_bounds = bounds
    ax.set_xlim(lower_bounds[0], upper_bounds[0])
    ax.set_ylim(lower_bounds[1], upper_bounds[1])

    # # Dibujar las islas como puntos
    # if len(islas) > 0:
    #     ax.scatter(islas[:, 0], islas[:, 1], s=1, c='blue', alpha=0.5, label='Zonas Factibles')
    
    scatter_plot_factible = ax.scatter([], [], c="green", label="Factible")
    scatter_plot_infactible = ax.scatter([], [], c="red", label="Infactible")
    scatter_plot_gbest = ax.scatter([], [], c="yellow", label="Mejor")

    ax.legend()
    return scatter_plot_factible, scatter_plot_infactible, scatter_plot_gbest, ax
