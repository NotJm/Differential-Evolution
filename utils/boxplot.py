import matplotlib.pyplot as plt

def generar_boxplot(soluciones_reflex, soluciones_ABC, nombre_archivo):
    data = [soluciones_reflex, soluciones_ABC]
    labels = ['DE Reflex', 'DE ABC']

    plt.boxplot(data, patch_artist=True, labels=labels)
    plt.xlabel('Mejor Fitness')
    plt.title('Comparación de DE Reflex y DE ABC')
    plt.grid(True)
    plt.savefig(nombre_archivo)  # Guardar el boxplot como imagen
    plt.show()