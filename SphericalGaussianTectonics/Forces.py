
import json
from pathlib import Path
from HelperFunctions import *
from scipy.spatial import cKDTree
from sklearn.cluster import DBSCAN
from scipy.spatial.transform import Rotation
mainDir = Path(".").parent.absolute().parent.absolute()

# =================================================== Move Tectonic Plates ===================================================
# Class for identifying and moving tectonic plates based on historical data
class PlateMover:
    def __init__(self, subdivision=4, plateIdsDirFormat=None, rotationsDir=None):

        # Set parameters
        self.subdivision = subdivision

        # Directory for Plate Ids
        if type(plateIdsDirFormat) == type(None):
            self.plateIdsDirFormat = str(mainDir) + '/Data/PlateIDs/Sub{}/Time{}.npz'
        else:
            self.plateIdsDirFormat = plateIdsDirFormat
        
        # Directory for Plate Rotations
        if type(rotationsDir) == type(None):
            self.rotationsDir = str(mainDir)+'/Data/PlateRotations.json'
        else:
            self.rotationsDir = rotationsDir

        with open(self.rotationsDir, 'r') as f:
            self.rotationsData = json.load(f)
        self.rotationsData = self.convertDict(self.rotationsData)
        
        # Prepare KDTree for identifying nearest neighbours
        verts, _ = getIcosphere(subdivisions=subdivision)
        self.kdTree = cKDTree(verts)
    
    # Identify plate IDs using nearest negihbours
    def identifyPoints(self, time, XYZ):
        plateIds = np.load(self.plateIdsDirFormat.format(self.subdivision, int(time)))['ids']
        _, idx = self.kdTree.query(XYZ)
        return plateIds[idx]
    
    # This function will be called by the simulator to move tectonic plates by one iteration
    def __call__(self, sim):

        # Get simulation parameters
        dt = sim.dt
        time = sim.currentTime
        XYZ = np.copy(sim.particles.XYZ)

        # Set up dictionaries of rotations
        plateIds = self.identifyPoints(time, XYZ)
        axis = self.rotationsData[time]['axii']
        angle = self.rotationsData[time]['angles']

        # Apply rotations to each plate
        for idx in np.unique(plateIds):
            quat = quaternion(axis[idx], dt * angle[idx])
            rot = Rotation.from_quat(quat) 
            XYZ[plateIds==idx] = rot.apply(XYZ[plateIds==idx])

        # Apply changes to coordinates
        sim.particles.XYZ = XYZ
    
    # Convert JSON data to prefered data types
    def convertDict(self, d):
        dic = {}
        for k, v in d.items():
            if str.isnumeric(k):
                dic[int(k)] = self.convertValue(v)
            else:
                dic[k] = self.convertValue(v)
        return dic

    # Used for processing lists during recursion
    def convertList(self, lst):
        return [self.convertValue(item) for item in lst]

    # Use recursion to traverse through nested dictionary
    def convertValue(self, v):
        if isinstance(v, dict):
            return self.convertDict(v)
        elif isinstance(v, list):
            return self.convertList(v)
        else:
            return v

# ==================================================== Other =======================================================
# Prevents particles from moving too close to each other
class ParticleCollider:
    def __init__(self, particleRadius=0.2):
        self.radius = particleRadius
    
    def __call__(self, sim):
        XYZ = np.copy(sim.particles.XYZ)

        # Get indices of all pairs of colliding particles
        pairs = sim.particles.kDTree.query_pairs(2*self.radius, output_type='ndarray')
        if pairs.shape[0] == 0:
            return XYZ
        
        # Get coordinates of pairs
        XYZ1 = XYZ[pairs[:, 0]]
        XYZ2 = XYZ[pairs[:, 1]]

        # Calculate direction and distance that particle need to move
        # to avoid being too close to each other
        displacement = XYZ2 - XYZ1
        distance = np.linalg.norm(displacement, axis=1)
        direction = displacement / distance[:, None]
        distToMove = self.radius - (distance/2)

        # Move particles accordingly
        XYZ[pairs[:, 0]] = XYZ1 - direction * distToMove[:, None]
        XYZ[pairs[:, 1]] = XYZ2 + direction * distToMove[:, None]
        sim.particles.XYZ = XYZ

# Particles may be moved off from the surface of the sphere
# This class projects them back onto the surface of the sphere
class ProjectOntoSphere:
    def __call__(self, sim):
        r = sim.particles.radius
        XYZ = sim.particles.XYZ
        length = np.linalg.norm(XYZ, axis=1)
        sim.particles.XYZ = r * XYZ / length[:, None]

class Despawner:
    def __init__(self, clusterSizeCoeff=1, maxClusterDistance=0.7, probDespawn=0.02):
        self.clusterSizeCoeff = clusterSizeCoeff
        self.maxClusterDistance = maxClusterDistance
        self.probDespawn = probDespawn # To Do: Adjust this based on dt

    def __call__(self, sim):
        XYZ = sim.particles.XYZ

        # Get clusters of particles
        numParticles = XYZ.shape[0]
        clusterSize = self.clusterSizeCoeff * np.log10(numParticles)
        clustering = DBSCAN(eps=self.maxClusterDistance, min_samples=int(clusterSize)).fit(XYZ)

        # Choose random particles withing cluster to despawn
        isInCluster = (clustering.labels_ != -1)
        inClusterIdx = np.argwhere(isInCluster)
        numInCluster = np.sum(isInCluster)
        randNum = np.random.random(numInCluster)
        probDespawn = self.probDespawn * sim.dt
        despawnThisParticle = (probDespawn > randNum)
        despawnIdx = inClusterIdx[despawnThisParticle]

        # Despawn the particles
        sim.particles.XYZ = np.delete(sim.particles.XYZ, despawnIdx[:, 0], 0)

class Spawner:
    def __call__(self, sim):
        XYZ = sim.particles.XYZ

        # Get number of missing particles
        currentNumParts = XYZ.shape[0]
        requiredNumParts = sim.particles.numParticles
        missingNumParts = requiredNumParts - currentNumParts

        # Spawn particles at random coordinates (for now)
        randCoords = 2 * np.random.random((missingNumParts, 3)) - 1
        sim.particles.XYZ = np.concatenate((XYZ, randCoords), axis=0)
