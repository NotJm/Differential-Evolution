import numpy as np


class VelocityHandler:
    
    """
    2004 – Robinson Particle
    """

    @staticmethod
    # Muros Absorbentes
    def absorbing_walls(position, velocity, bounds):
        for i in range(len(position)):
            if position[i] < bounds[i][0] or position[i] > bounds[i][1]:
                velocity[i] = 0
        return velocity

    @staticmethod
    # Muros Reflectantes
    def reflecting_walls(position, velocity, bounds):
        for i in range(len(position)):
            if position[i] < bounds[i][0] or position[i] > bounds[i][1]:
                velocity[i] = -velocity[i]
        return velocity
    
    """
    2004 Mendes Population
    """

    @staticmethod
    def mendes_apply_bound_restriction(positions, velocities, lower_bounds, upper_bounds):
        for i in range(positions.shape[0]):
            for d in range(positions.shape[1]):
                if positions[i, d] < lower_bounds[d]:
                    positions[i, d] = lower_bounds[d]
                    velocities[i, d] = 0  # Opcional: Resetear velocidad a 0
                elif positions[i, d] > upper_bounds[d]:
                    positions[i, d] = upper_bounds[d]
                    velocities[i, d] = 0  # Opcional: Resetear velocidad a 0
        return positions
    
    """
    2005 Huang Hybrid
    """

    @staticmethod
    def huang_apply_damping_boundaries(position, velocity):
        for i in range(len(position)):
            if position[i] < 0 or position[i] > 1:
                alpha = np.random.uniform(0, 1)
                velocity[i] = -alpha * velocity[i]
                position[i] = np.clip(position[i], 0, 1)
                
    """
    2006 Confinements
    """

    @staticmethod
    def no_Confinement(position, velocity):
        return position, velocity

    @staticmethod
    def no_Confinement_ArtificialLandscape(
        position, velocity, lower_bound, upper_bound
    ):
        if np.any(position < lower_bound) or np.any(position > upper_bound):
            position = np.clip(position, lower_bound, upper_bound)
        return position, velocity

    @staticmethod
    def standard(position, velocity, lower_bound, upper_bound):
        position = np.clip(position, lower_bound, upper_bound)
        velocity = np.where(position == lower_bound, -velocity, velocity)
        velocity = np.where(position == upper_bound, -velocity, velocity)
        return position, velocity

    @staticmethod
    def deterministic_back(position, velocity, lower_bound, upper_bound, gamma=0.5):
        if np.any(position < lower_bound) or np.any(position > upper_bound):
            position = np.clip(position, lower_bound, upper_bound)
            velocity *= -gamma
        return position, velocity

    @staticmethod
    def random_back(position, velocity, lower_bound, upper_bound):
        if np.any(position < lower_bound) or np.any(position > upper_bound):
            random_factor = np.random.uniform(0, 1, size=position.shape)
            position = np.clip(position, lower_bound, upper_bound)
            velocity *= -random_factor
        return position, velocity

    @staticmethod
    def consistent(position, velocity, lower_bound, upper_bound):
        if np.any(position < lower_bound) or np.any(position > upper_bound):
            for i in range(len(position)):
                if position[i] < lower_bound[i] or position[i] > upper_bound[i]:
                    velocity[i] = -velocity[i]
        return position, velocity

    @staticmethod
    def hyperbolic(position, velocity, lower_bound, upper_bound):
        if np.any(position < lower_bound) or np.any(position > upper_bound):
            position = np.clip(position, lower_bound, upper_bound)
            velocity = -velocity / (1 + np.abs(velocity))
        return position, velocity

    @staticmethod
    def relativity_like(position, velocity, lower_bound, upper_bound, c=1):
        if np.any(position < lower_bound) or np.any(position > upper_bound):
            position = np.clip(position, lower_bound, upper_bound)
            velocity = -velocity / (1 + (velocity**2 / c**2))
        return position, velocity

    @staticmethod
    def random_forth(position, velocity, lower_bound, upper_bound):
        if np.any(position < lower_bound) or np.any(position > upper_bound):
            random_factor = np.random.uniform(0, 1, size=position.shape)
            position = np.clip(position, lower_bound, upper_bound)
            velocity *= random_factor
        return position, velocity

    @staticmethod
    def hybrid(position, velocity, lower_bound, upper_bound):
        position = np.clip(position, lower_bound, upper_bound)
        velocity = np.where(position == lower_bound, -velocity, velocity)
        velocity = np.where(position == upper_bound, -velocity, velocity)

        if np.any(position < lower_bound) or np.any(position > upper_bound):
            if np.any(position < lower_bound) or np.any(position > upper_bound):
                position = np.clip(position, lower_bound, upper_bound)
                velocity = -velocity / (1 + np.abs(velocity))
        return position, velocity

