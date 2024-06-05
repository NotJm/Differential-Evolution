import matplotlib.pyplot as plt

def generar_histograma(mejores_violaciones, nombre_archivo, titulo):
    plt.hist(mejores_violaciones, bins=20, color='blue', alpha=0.7)
    plt.xlabel('Valor de la mejor violación')
    plt.ylabel('Frecuencia')
    plt.title(titulo)
    plt.grid(True)
    plt.savefig(nombre_archivo)  # Guardar el histograma como imagen
    plt.show()