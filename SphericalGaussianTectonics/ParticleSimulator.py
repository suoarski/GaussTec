
from GaussianParticles import *

class ParticleSimulator:
    def __init__(self, particles, startTime=10, endTime=0, dt=1, forces=[], lookAtLonLat=[0, 0], cameraZoom=1.4):

        # Time related parameters
        self.dt = dt
        self.endTime = endTime
        self.iterations = int((startTime - endTime)//dt) + 1
        self.time = np.linspace(startTime, endTime, self.iterations)

         # Other parameters
        self.particles = particles
        self.forces = forces

        # Animation related parameters
        self.lookAtLonLat = lookAtLonLat
        self.cameraZoom = cameraZoom
        
    
    # This gets called every simulation iteration
    def simulationIteration(self):
        self.particles.iterationPrecompuations()
        for force in self.forces:
            force(self)
    
    # Run the simulation
    def runSimulation(self, printCurrentIterations=False):
        for i, t in enumerate(self.time):
            if printCurrentIterations:
                print('Current Simulations Time: {} MYA'.format(t))
            self.currentTime = t
            self.simulationIteration()
    
    # Run the simulation and animate results as MP4 file
    def animate(self, animationDir='Animation.mp4', includeEvaluations=True, framesPerIteration=4, clim=[-6000, 10000],
                    includeGlyphs=False, glyphSize=0.5):

        # Set up plotter for animation
        plotter = pv.Plotter(notebook=False, off_screen=True)
        if includeEvaluations:
            plotter.add_mesh(self.particles.mesh, scalars=self.particles.evaluate(), clim=clim)
        if includeGlyphs:
            plotter.add_mesh(self.particles.getGlyphs(size=glyphSize))

        # Set up camera positions
        plotter.camera_position = 'yz'
        plotter.camera.zoom(self.cameraZoom)
        plotter.camera.elevation = self.lookAtLonLat[0]
        plotter.camera.azimuth = 180 + self.lookAtLonLat[1]

        #Write initial frames of movie
        plotter.open_movie(animationDir)
        for i in range(framesPerIteration):
            plotter.write_frame()

        # Run simulation and animate
        for i, t in enumerate(self.time):
            self.currentTime = t
            self.simulationIteration()

            # Update mesh data for this iteration
            if includeEvaluations:
                plotter.update_scalars(self.particles.evaluate())
            if includeGlyphs:
                plotter.update_coordinates(self.particles.getGlyphs(size=glyphSize).points)

            # Write frames
            for i in range(framesPerIteration):
                plotter.write_frame()
        plotter.close()

