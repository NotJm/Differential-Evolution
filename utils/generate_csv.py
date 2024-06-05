import csv
import os

# Función para guardar los resultados óptimos en un CSV, agregando nuevas filas
def guardar_resultados_csv(datos: dict, filename: str):
    # Verificar si el archivo ya existe
    file_exists = os.path.isfile(filename)

    with open(filename, mode='a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=datos.keys())
        # Escribir los encabezados solo si el archivo no existe
        if not file_exists:
            writer.writeheader()
        # Escribir los resultados
        writer.writerow(datos)