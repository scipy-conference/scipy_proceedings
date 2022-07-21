from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.pyplot as plt
import numpy as np

def calcRho(alt):  # alt is altitude above sea level [m]
    # Constants
    p0 = 101325  # sea-level standard pressure [Pa]
    T0 = 288.15  # sea-level standard temperature [K]
    g = 9.81     # gravitational acceleration [m/s^2]
    L = 0.0065   # [K/m]
    R = 8.3145   # [J/(mol*K)]
    M = 0.02897  # [kg/mol]

    # Calculations
    # See: https://en.wikipedia.org/wiki/Density_of_air#Variation_with_altitude
    T = T0 - L*alt  # Temperature at alititude [Pa]
    p = p0*(1-(L*alt/T0))**(g*M/(R*L) - 1)  # Air pressure at alititude [Pa]
    rho = p*M/(R*T)
    return rho


# Generate a baseball field in a 3d plot
def plot_baseball_field(ax):
    d2r = np.pi/180  # degrees to radians conversion [rad/deg]

    ax.set_ylim([-80, 80])
    ax.set_xlim([-10, 150])
    ax.set_zlim([0, 45])

    # Pitchers mount
    circle_angs = np.arange(-180, 180+1, 1)*d2r
    pitchers_mound = np.array([5.47/2*np.cos(circle_angs) + 18.39, 5.47/2*np.sin(circle_angs)]).T
    # ax.plot(pitchers_mound[:, 0], pitchers_mound[:, 1], 0*pitchers_mound[:, 0], c='k', zorder=5)
    verts = [list(zip(pitchers_mound[:, 0], pitchers_mound[:, 1], 0*pitchers_mound[:, 0]))]
    coll = Poly3DCollection(verts, color='wheat')
    coll.set_sort_zpos(-1)
    ax.add_collection3d(coll)

    # Base Diamond
    ang = np.cos(d2r*45)
    diamond = np.array([[0, 0],
                        [ang, -ang],
                        [2*ang, 0],
                        [ang, ang],
                        [0, 0]]) * 27.43
    # ax.plot(diamond[:, 0], diamond[:, 1], 0*diamond[:, 0], c='k', zorder=5)
    verts = [list(zip(diamond[:, 0], diamond[:, 1], 0*diamond[:, 0]))]
    coll = Poly3DCollection(verts, color='forestgreen')
    coll.set_sort_zpos(-2)
    ax.add_collection3d(coll)

    # Infield Boundary
    angs = np.arange(-45, 45+0.1, 0.1)*d2r
    infield_ys = np.append(np.arange(-27.524, 27.524+0.1, 0.1), 27.524)
    infield_xs = np.sqrt(29**2-infield_ys**2) + 18.39
    infield = np.array([[0, 0]])
    infield = np.append(infield, np.array([infield_xs, infield_ys]).T, axis=0)
    infield = np.append(infield, np.array([[0, 0]]), axis=0)
    # ax.plot(infield[:, 0], infield[:, 1], 0*infield[:, 0], c='k', zorder=5)
    verts = [list(zip(infield[:, 0], infield[:, 1], 0*infield[:, 0]))]
    coll = Poly3DCollection(verts, color='wheat')
    coll.set_sort_zpos(-3)
    ax.add_collection3d(coll)

    # Outfield Boundary
    outfield_ys = np.sin(angs)*100
    outfield_xs = outfield_x(outfield_ys)
    outfield = np.array([[0, 0]])
    outfield = np.append(outfield, np.array([outfield_xs, outfield_ys]).T, axis=0)
    outfield = np.append(outfield, np.array([[0, 0]]), axis=0)
    ax.plot(outfield[:, 0], outfield[:, 1], 0*outfield[:, 0], c='k', zorder=5)
    verts = [list(zip(outfield[:, 0], outfield[:, 1], 0*outfield[:, 0]))]
    coll = Poly3DCollection(verts, color='forestgreen')
    coll.set_sort_zpos(-4)
    ax.add_collection3d(coll)

    return


# Equation for the outfield boundary (350 ft x 400 ft)
def outfield_x(outfield_y):
    return 125 - 0.01085786 * outfield_y**2
