import numpy as np
from firedrake import *
# TODO: read from data

def inside_dark(x, y, z, data):
    l = data["l"]
    R = data["R"]
    t = data["t"]

    a = 2*np.sqrt(l*l/4 + R*R*2)
    l = l/a
    R = R/a
    t = t/a
    a = a/a

    xc = a/2
    yc = a/2
    theta = np.arctan(l/(2*R)) + pi/4


    rr = sqrt((x-xc)**2 + (y-yc)**2)
    inside_ring = And(rr >= R-t/2,
                rr <= R+t/2)
    
    def inside_beam(theta_beam):

        # Tangency point on the ring
        x0 = xc + R*np.cos(theta_beam)
        y0 = yc + R*np.sin(theta_beam)

        # Tangent direction
        tx = -np.sin(theta_beam)
        ty =  np.cos(theta_beam)

        # Endpoint of the ligament centerline
        x1 = x0 + (l/2)*tx
        y1 = y0 + (l/2)*ty

        dx = x1 - x0
        dy = y1 - y0
        L2 = dx*dx + dy*dy

        # Projection parameter
        u = ((x - x0)*dx + (y - y0)*dy) / L2
        u = min_value(max_value(u, 0.0), 1.0)

        # Closest point on the segment
        xp = x0 + u*dx
        yp = y0 + u*dy

        # Distance from point to centerline
        dist = sqrt((x - xp)**2 + (y - yp)**2)

        return dist <= t/2

    inside_dark = inside_ring
    for k in range(4):
        inside_dark = Or(
            inside_dark,
            inside_beam(theta + k*np.pi/2)
        )

    return inside_dark

# mesh = CubeMesh(n, n, 1, a)

# P0 = FunctionSpace(mesh, "DG", 0)

# x, y, z = SpatialCoordinate(mesh)
# mu_fun = lambda x,y,z: conditional(inside_dark(x,y,z), Constant(0.2), Constant(0.8))
# mu_expr = mu_fun(x,y,z)
# mu = Function(P0).interpolate(mu_expr)

# VTKFile("mu.pvd").write(mu)
