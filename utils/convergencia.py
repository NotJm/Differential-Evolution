import matplotlib.pyplot as plt

def graficar_convergencia(mejores_fitness, nombre_archivo, titulo):
    plt.plot(range(1, len(mejores_fitness) + 1), mejores_fitness, marker='o', color='b')
    plt.xlabel('Iteraciones')
    plt.ylabel('Infracciones')
    plt.title(titulo)
    plt.grid(True)
    plt.savefig(nombre_archivo)  # Guardar la gráfica como imagen
    plt.show()