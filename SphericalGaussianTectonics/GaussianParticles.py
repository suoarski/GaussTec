
import pyvista as pv
from HelperFunctions import *
from scipy.spatial import cKDTree

class GaussParticlesSphere:
    def __init__(self, subdivisions=5, smoothness=2, plotSubdivisions=5, numNeighbs=100,
                    maxOceanDepth = -6000, maxMountainHeight=8000, averageTerrainHeights=-1800):
        
        # Set up particles from icosphere
        self.subdivisions = subdivisions
        self.XYZ, _ = getIcosphere(subdivisions=subdivisions)
        self.std = smoothness

        # Set radius of sphere
        self.numParticles = self.XYZ.shape[0]
        self.radius = (self.numParticles / (4 * np.pi))**0.5
        self.XYZ *= self.radius

        # Set number of neighbours used in distance calculations
        if numNeighbs > self.numParticles:
            numNeighbs = self.numParticles
        self.numNeighbs = numNeighbs

        # Set up sphere for plotting
        self.plotSubdivisions = plotSubdivisions
        self.plotXYZ, self.plotCells = getIcosphere(subdivisions=plotSubdivisions)
        self.plotFaces = cellsToFaces(self.plotCells)
        self.plotXYZ *= self.radius
        self.mesh = pv.PolyData(self.plotXYZ, self.plotFaces)
        self.plotKDTree = cKDTree(self.plotXYZ)

        # Height parameters
        self.maxOceanDepth = maxOceanDepth
        self.averageTerrainHeights = averageTerrainHeights
        self.maxMountainHeight = maxMountainHeight # Not used yet

        # Set mass of particles to a constant such that we get our desired average terrain height
        self.w = 1
        self.w = (averageTerrainHeights - maxOceanDepth) / np.mean(self.evaluate(shiftZero=False))
    
    # Evaluate the resulting landscape of particles using RBF
    # This function is optimized but only works for points on our icosphere for plotting
    def evaluate(self, shiftZero=True):
        dist, idx = self.plotKDTree.query(self.XYZ, k=self.numNeighbs)
        RBF = self.w * gaussRBF(dist, self.std)
        heights = np.zeros(self.plotXYZ.shape[0])
        np.add.at(heights, idx, RBF)
        if shiftZero:
            heights += self.maxOceanDepth
        return heights

    # Evaluate the resulting landscape of particles using RBF
    # This function is slower, but can evaluate any point in space
    def evaluateFromAnyPoints(self, XYZEval, numNeighbs=100):
        dist, _ = cKDTree(self.XYZ).query(XYZEval, k=numNeighbs)
        RBF = self.w * gaussRBF(dist, self.std)
        heights = np.sum(RBF, axis=1)
        return heights + self.maxOceanDepth
    
    # Glyphs are for visualizing particle locations
    def getGlyphs(self, size=0.2, scalars=None):
        glyphGeom = pv.Sphere(theta_resolution=1, phi_resolution=1)
        pointCloud = pv.PolyData(self.XYZ)
        if isinstance(scalars, np.ndarray):
            pointCloud['color'] = scalars
        glyphs = pointCloud.glyph(geom=glyphGeom, factor=size, scale=False, indices='color')
        return glyphs
    
    # These calculations may be used by various algorithms,
    # So we calculate them once for each iteration rather than each time they are used in code
    def iterationPrecompuations(self):
        self.kDTree = cKDTree(self.XYZ)