
import numpy as np

# A force for making particles move towards the origin
class ConvergenceForce:
    def __init__(self, speed=2):
        self.speed = speed

    # This function is required by all force type classes
    def __call__(self, simulator):
        parts = simulator.particles
        return - np.sign(parts.x) * self.speed * simulator.dt