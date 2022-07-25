
# Set path of main directory
import numpy as np
from pathlib import Path
mainDir = Path(".").parent.absolute().parent.absolute()

# =========================================== Reading Icosphere Files =================================================
# Read NPZ files for Icosphere data
def getIcosphere(subdivisions=4):
    fileDir = str(mainDir) + '/Data/SpheresNPZ/IcosphereSubs{}.npz'.format(subdivisions)
    data = np.load(fileDir)
    return data['vertices'], data['cells']

# Converts cells format to pyvista faces for plotting
def cellsToFaces(cells):
    faces = []
    for c in cells:
        faces.append(3)
        for x in c:
            faces.append(x)
    return np.array(faces)

# ================================================ Spherical Coordinate Transformations =============================
# Coordinate transformation function between polar and cartesian
def polarToCartesian(radius, theta, phi, useLonLat=True):
    if useLonLat == True:
        theta, phi = np.radians(theta+180.), np.radians(90. - phi)
    X = radius * np.cos(theta) * np.sin(phi)
    Y = radius * np.sin(theta) * np.sin(phi)
    Z = radius * np.cos(phi)
    
    #Return data either as a list of XYZ coordinates or as a single XYZ coordinate
    if (type(X) == np.ndarray):
        return np.stack((X, Y, Z), axis=1)
    else:
        return np.array([X, Y, Z])

# Coordinate transformation function between polar and cartesian
def cartesianToPolarCoords(XYZ, useLonLat=True):
    X, Y, Z = XYZ[:, 0], XYZ[:, 1], XYZ[:, 2]
    R = (X**2 + Y**2 + Z**2)**0.5
    theta = np.arctan2(Y, X)
    phi = np.arccos(Z / R)
    
    #Return results either in spherical polar or leave it in radians
    if useLonLat == True:
        theta, phi = np.degrees(theta), np.degrees(phi)
        lon, lat = theta - 180, 90 - phi
        lon[lon < -180] = lon[lon < -180] + 360
        return R, lon, lat
    else:
        return R, theta, phi

# Given an axis of rotation and angle, create a quaternion
def quaternion(axis, angle):
    return [np.sin(angle/2) * axis[0], 
            np.sin(angle/2) * axis[1], 
            np.sin(angle/2) * axis[2], 
            np.cos(angle/2)]

# ===================================================== Other ============================================================
# Gaussian radial basis function
def gaussRBF(radius, std):
    coeff = 1/(std*(2*np.pi)**0.5)
    return coeff * np.exp(-0.5*(radius / std)**2)