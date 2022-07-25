
import numpy as np
from HelperFunctions import *

class ParticleSimulator:
    def __init__(self, particles, dt=0.1, finalTime=6, forces=[], initiators=[]):

        # Time related parameters
        self.dt = dt
        self.finalTime = finalTime
        self.iterations = int(finalTime//dt) + 1
        self.time = np.linspace(0, finalTime, self.iterations)

         # Particles related parameters
        self.particles = particles
        self.forces = forces

        self.xHist = np.ones((self.iterations, particles.numParticles))
        self.xHist *= self.particles.x
        self.stdHist = np.ones((self.iterations, particles.numParticles))
        self.stdHist *= self.particles.std

        # Run optional initiators
        for init in initiators:
            init(self)
    
    # This gets called every simulation iteration
    def simulationIteration(self):
        self.dx = np.zeros(self.particles.x.shape)
        for force in self.forces:
            force(self)
    
    # Run the simulation
    def runSimulation(self):
        self.xHist[0] = self.particles.x
        self.stdHist[0] = self.particles.std
        for i, t in enumerate(self.time[1:]):
            self.simulationIteration()
            self.xHist[i+1] = self.particles.x
            self.stdHist[i+1] = self.particles.std
        
        
    
    # Get plot of historical landscape
    def getHistoricLandscapePlot(self, ax):
        for i, x in enumerate(self.xHist):
            c = i / len(self.xHist)
            self.particles.std = self.stdHist[i]
            heightMap = self.particles.evaluate(x=x)
            ax.plot(self.particles.xEval, heightMap, color=[c, 0.0, (1 - c)])
        ax.set_title('Historical Landscape')
        ax.set_xlabel('Position x')
        ax.set_ylabel('Height')
        return ax
    
    # Plot for visualizing the historical positions of particles
    def getHistoricalParticlePositionPlot(self, ax):
        for x in self.xHist.T:
            ax.plot(x, self.time, 'b', linewidth=0.8)
        bounds = [np.min(self.particles.xEval), np.max(self.particles.xEval)]
        ax.set_xlim(bounds)
        ax.set_ylim([self.finalTime + 0.2, -0.2])
        ax.set_title('Historical Location of Particles')
        ax.set_xlabel('Position x')
        ax.set_ylabel('Time')
        return ax

