
import numpy as np
from HelperFunctions import *
from scipy.optimize import nnls

class GaussParticles:

    # Initiate gaussian particles with a flat terrain
    def initFlat(self, height=1, numParticles=100, bounds=[0, 1], smoothness=2):
        self.x = np.linspace(bounds[0], bounds[1], numParticles)
        self.w =  height * (bounds[1] - bounds[0]) / numParticles
        self.initSpacing = self.x[1] - self.x[0]
        self.std = smoothness * self.initSpacing * np.ones(numParticles)
        self.xEval = np.copy(self.x)
        self.numParticles = numParticles
        self.smoothness = smoothness
        return self
    
    # Use RBF to initiate particles by fitting them from data
    def initFromData(self, x, y, smoothness=2):
        self.x = x
        self.y = y
        self.numParticles = len(x)
        self.std = smoothness * (x[1] - x[0])
        self.xEval = np.copy(self.x)

        # Compute RBF to get weights of particles
        dist = getDistance(self.x, self.x)
        RBF = gaussRBF(dist, self.std)
        self.w, _ = nnls(RBF, self.y)
        return self

    # Evaluate the resulting heightmap at points xEval by summing all particles
    def evaluate(self, xEval=None, x=None):
        if type(xEval) == type(None):
            xEval = self.xEval
        if type(x) == type(None):
            x = self.x
        dist = getDistance(xEval, x)
        RBF = gaussRBF(dist, self.std)
        self.particles = self.w * RBF
        self.y = np.sum(self.particles, axis=1)
        return self.y
    
    # Creates a plot of all particles and their combined evaluation
    def plotAllParticles(self, ax, color='blue', scalars=None):
        eval = self.evaluate()
        ax.plot(self.xEval, eval, color)

        # If a scalar is given, then color code particles based on scalars
        # Begin by bringing it to a range of [0, 1]
        displayScalars = type(scalars) != type(None)
        if displayScalars:
            scalars -= np.min(scalars)
            scalars /= np.max(scalars)

        # Plot each individual particle
        for i, p in enumerate(self.particles.T):
            if displayScalars:
                c = scalars[i]# / np.max(scalars)
                color = [c, 0, 1-c]
            ax.plot(self.xEval, p, color=color, linewidth=0.5)
        ax.set_title('Evaluations and Individual Particles')
        return ax