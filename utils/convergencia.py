from typing import Callable, Tuple, List
import matplotlib.pyplot as plt


def generate_convergencia_graphic(data, nombre_archivo, titulo, xlabel, ylabel):
    plt.plot(range(1, len(data) + 1), data, marker="o", color="b")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(titulo)
    plt.grid(True)
    plt.savefig(nombre_archivo)
    plt.show()


# Genera un grafico de convergencia mediante la media
def generate_convergencia_mean_graphic(
    method_names: List[str],
    filename: str,
    title: str,
    xlabel: str,
    ylabel: str,
    means: List[Tuple[float, ...]]
):
    plt.figure(figsize=(10, 6))
    for method_name, method_mean in zip(method_names, means):
        plt.plot(method_mean, label=method_name)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.savefig(filename)
    plt.show()