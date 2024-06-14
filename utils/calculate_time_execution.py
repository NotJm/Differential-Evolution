import time

def measurement_timer_execution(funcion, *args, **kwargs):
    inicio = time.time()
    resultado = funcion(*args, **kwargs)
    fin = time.time()
    tiempo_ejecucion = fin - inicio
    print(f'Tiempo de ejecución: {tiempo_ejecucion:.6f} segundos')
    return resultado, tiempo_ejecucion

