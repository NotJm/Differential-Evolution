import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# Definir universos de discurso para entradas y salidas
delta_f = ctrl.Antecedent(np.arange(-1, 1, 0.01), 'delta_f')
delta_x = ctrl.Antecedent(np.arange(-1, 1, 0.01), 'delta_x')
F_new = ctrl.Consequent(np.arange(0, 2, 0.01), 'F_new')
CR_new = ctrl.Consequent(np.arange(0, 1, 0.01), 'CR_new')

# Definir funciones de membresía para las entradas
delta_f.automf(3)
delta_x.automf(3)

# Definir funciones de membresía para las salidas
F_new['low'] = fuzz.trimf(F_new.universe, [0, 0, 1])
F_new['medium'] = fuzz.trimf(F_new.universe, [0, 1, 2])
F_new['high'] = fuzz.trimf(F_new.universe, [1, 2, 2])

CR_new['low'] = fuzz.trimf(CR_new.universe, [0, 0, 0.5])
CR_new['medium'] = fuzz.trimf(CR_new.universe, [0, 0.5, 1])
CR_new['high'] = fuzz.trimf(CR_new.universe, [0.5, 1, 1])

# Definir reglas difusas
rule1 = ctrl.Rule(delta_f['poor'] | delta_x['poor'], F_new['low'])
rule2 = ctrl.Rule(delta_f['average'], F_new['medium'])
rule3 = ctrl.Rule(delta_f['good'] | delta_x['good'], F_new['high'])

rule4 = ctrl.Rule(delta_f['poor'] | delta_x['poor'], CR_new['low'])
rule5 = ctrl.Rule(delta_f['average'], CR_new['medium'])
rule6 = ctrl.Rule(delta_f['good'] | delta_x['good'], CR_new['high'])

# Crear el sistema de control difuso
F_control = ctrl.ControlSystem([rule1, rule2, rule3])
CR_control = ctrl.ControlSystem([rule4, rule5, rule6])

F_simulation = ctrl.ControlSystemSimulation(F_control)
CR_simulation = ctrl.ControlSystemSimulation(CR_control)

# Ajustar los valores de entrada y calcular las salidas
def adapt_parameters(delta_f_val, delta_x_val):
    F_simulation.input['delta_f'] = delta_f_val
    F_simulation.input['delta_x'] = delta_x_val
    CR_simulation.input['delta_f'] = delta_f_val
    CR_simulation.input['delta_x'] = delta_x_val

    F_simulation.compute()
    CR_simulation.compute()

    return F_simulation.output['F_new'], CR_simulation.output['CR_new']

# Ejemplo de uso
delta_f_val = 0.1  # Ejemplo de cambio en la función objetivo
delta_x_val = 0.2  # Ejemplo de cambio en el vector de parámetros

F_adapted, CR_adapted = adapt_parameters(delta_f_val, delta_x_val)
print(f"F adaptado: {F_adapted}")
print(f"CR adaptado: {CR_adapted}")
