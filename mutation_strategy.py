import numpy as np

class MutationStrategies:
    def __init__(self, population, scale):
        self.population = population
        self.scale = scale

    def _best1(self, samples):
        r0, r1 = samples[:2]
        return (self.population[0] + self.scale *
                (self.population[r0] - self.population[r1]))

    def _rand1(self, samples):
        r0, r1, r2 = samples[:3]
        return (self.population[r0] + self.scale *
                (self.population[r1] - self.population[r2]))

    def _randtobest1(self, samples):
        r0, r1, r2 = samples[:3]
        bprime = np.copy(self.population[r0])
        bprime += self.scale * (self.population[0] - bprime)
        bprime += self.scale * (self.population[r1] - self.population[r2])
        return bprime

    def _currenttobest1(self, candidate, samples):
        r0, r1 = samples[:2]
        bprime = (self.population[candidate] + self.scale *
                  (self.population[0] - self.population[candidate] +
                   self.population[r0] - self.population[r1]))
        return bprime

    def _best2(self, samples):
        r0, r1, r2, r3 = samples[:4]
        bprime = (self.population[0] + self.scale *
                  (self.population[r0] + self.population[r1] -
                   self.population[r2] - self.population[r3]))
        return bprime

    def _rand2(self, samples):
        r0, r1, r2, r3, r4 = samples
        bprime = (self.population[r0] + self.scale *
                  (self.population[r1] + self.population[r2] -
                   self.population[r3] - self.population[r4]))
        return bprime
    
    def res_rand(self, i, lower, upper):
        NP, D = self.population.shape
        NoRes = 0
        valid = False

        while NoRes <= 3 * D and not valid:
            # Selección de tres índices aleatorios diferentes
            indices = np.random.choice(np.delete(np.arange(NP), i), 3, replace=False)
            r1, r2, r3 = indices

            # Generación del vector mutante
            mutant = self.population[r1] + self.scale * (self.population[r2] - self.population[r3])
            
            # Verificación de validez
            if np.all([lower[j] <= mutant[j] <= upper[j] for j in range(D)]):
                valid = True
            
            NoRes += 1

        if not valid:
            # Reparación del vector fuera de límites
            for j in range(D):
                if mutant[j] < lower[j] or mutant[j] > upper[j]:
                    mutant[j] = lower[j] + np.random.rand() * (upper[j] - lower[j])

        return mutant


