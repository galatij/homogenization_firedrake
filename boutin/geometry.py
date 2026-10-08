import numpy as np
from firedrake import *
import gmsh
import meshio
import math

def make_load_function(data):

    coeff = data["coefficients"]
    flags = data["flags"]
    dim = data["dim"]
    load = coeff["loadx"]
    Tx = data["geometry"]["fs_geom"]["Lx_macro"]
    Ty = data["geometry"]["fs_geom"]["Ly_macro"]
    Tz = data["geometry"]["fs_geom"]["Lz_macro"]

    if flags["is_per"]:
        if dim == 2:
            return lambda x, y, z: as_vector((5e3*sin(2*pi/Tx*x), 0.0))
        elif dim == 3:
            f_vec[2] = coeff.get("loadz", 0.0)
            return lambda x, y, z: as_vector((5e3*sin(2*pi/Tx*x), 0.0, 0.0))

    f_vec = np.zeros(dim)
    f_vec[0] = coeff["loadx"]
    f_vec[1] = coeff["loady"]

    if dim == 3:
        f_vec[2] = coeff.get("loadz", 0.0)

    return lambda x, y, z: as_vector(f_vec)

def make_material_functions(data, fs=False):

    coeff = data["coefficients"]

    mu_dark = Constant(coeff["mu_dark"])
    mu_light = Constant(coeff["mu_light"])

    lmbda_dark = Constant(coeff["lmbda_dark"])
    lmbda_light = Constant(coeff["lmbda_light"])

    def mu_fun(x, y, z):

        dark = inside_geometry(
            x, y, z,
            data,
            fine_scale=fs,
        )

        return conditional(
            dark,
            mu_dark,
            mu_light,
        )

    def lmbda_fun(x, y, z):

        dark = inside_geometry(
            x, y, z,
            data,
            fine_scale=fs,
        )

        return conditional(
            dark,
            lmbda_dark,
            lmbda_light,
        )

    return mu_fun, lmbda_fun

def inside_geometry(x, y, z, data, fine_scale=False):

    name = data["flags"]["microstructure"]

    try:
        geometry = GEOMETRIES[name]
        # if name == "layered":
        #     raise NotImplementedError(f"layered_geometry requires implementation.")
    except KeyError:
        raise ValueError(f"Unknown microstructure: {name}")

    return geometry(
        x, y, z,
        data["geometry"],
        fine_scale=fine_scale,
    )

def inside_66(x, y, z, data_geometry, fine_scale=False):
    
    if fine_scale:
        nx = data_geometry["fs_geom"]["nx_macro"]
        ny = data_geometry["fs_geom"]["ny_macro"]
        nz = data_geometry["fs_geom"]["nz_macro"]
        n_micro = data_geometry["fs_geom"]["n_micro_fs"]
        l_micro = data_geometry["fs_geom"]["l_micro"]
        x, y, z = periodic_map(x, y, z, (nx, ny, nz), l_micro)

    mirror = data_geometry["chiral"]["mirror"]
    l = data_geometry["chiral"]["l"]
    R = data_geometry["chiral"]["R"]
    t = data_geometry["chiral"]["t"]

    a = 2*np.sqrt(l*l/4 + R*R)
    l = l/a
    R = R/a
    t = t/a
    a = a/a

    xc = a/2
    yc = a/2

    if data_geometry["chiral"]["mirror"]:
        # Reflect x across the local cell center axis (xc = 0.5)
        x = 2 * xc - x  

    theta = pi/2 - np.arctan(l/(2*R)) #if mirror == False else pi/2 + np.arctan(l/(2*R)) + err


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

    inside_66 = inside_ring
    for k in range(4):
        inside_66 = Or(
            inside_66,
            inside_beam(theta + k*np.pi/2)
        )

    return inside_66

def periodic_map(x, y, z, ncell, Lcell=1):
    nx, ny, nz = ncell

    return (
        periodic_coordinate(x, nx, Lcell),
        periodic_coordinate(y, ny, Lcell),
        periodic_coordinate(z, nz, Lcell),
    )

def periodic_coordinate(x, n, Lcell):
    """
    Map x from [0, n*Lcell] to [0, 1],
    where Lcell is the physical period.
    """

    xloc = x

    for k in range(n):
        xloc = conditional(
            And(
                x >= k*Lcell,
                x < (k+1)*Lcell
            ),
            (x-k*Lcell)/Lcell,
            xloc
        )

    # Map the right boundary back to 0
    xloc = conditional(
        x >= n*Lcell,
        0.0,
        xloc
    )

    return xloc

def inside_X(x, y, z, data_geometry, fine_scale=False):

    if fine_scale:
        nx = data_geometry["fs_geom"]["nx_macro"]
        ny = data_geometry["fs_geom"]["ny_macro"]
        nz = data_geometry["fs_geom"]["nz_macro"]
        l_micro = data_geometry["fs_geom"]["l_micro"]
        x, y, z = periodic_map(x, y, z, (nx, ny, nz), l_micro)

    t = data_geometry["crossed"]["t"]
    l_micro = data_geometry["crossed"]["l_micro"]

    # Geometry is normalized to [0,1] x [0,1]
    t = t / l_micro

    # Diagonal y = x
    d1 = abs(y - x) / np.sqrt(2)

    # Diagonal y = 1 - x
    d2 = abs(y + x - 1) / np.sqrt(2)

    return Or(
        d1 <= t/2,
        d2 <= t/2
    )

def inside_layered(x, y, z, data_geometry, fine_scale=False):
    if fine_scale:
        nx = data_geometry["fs_geom"]["nx_macro"]
        ny = data_geometry["fs_geom"]["ny_macro"]
        nz = data_geometry["fs_geom"]["nz_macro"]
        l_micro = data_geometry["fs_geom"]["l_micro"]
        x, y, z = periodic_map(x, y, z, (nx, ny, nz), l_micro)

    return y < 0.5 

GEOMETRIES = {
    "X": inside_X,
    "66": inside_66,
    "layered": inside_layered,
}

def check_gmsh_periodicity(filename):

    with open(filename, "r") as f:
        text = f.read()

    print("\nGmsh file periodicity check")
    print("---------------------------")

    if "$Periodic" not in text:
        print("[FAIL] No $Periodic section found.")
        return False

    print("[OK] $Periodic section found.")

    start = text.find("$Periodic")
    end = text.find("$EndPeriodic")

    periodic_section = text[start:end]

    # print("\n$Periodic section:")
    # print(periodic_section)

    return True

def check_vertical_pair(left_tag, right_tag, L, tol=1e-10):

    _, left = curve_nodes(left_tag)
    _, right = curve_nodes(right_tag)

    left_y = np.sort(left[:, 1])
    right_y = np.sort(right[:, 1])

    print(
        f"\nVertical pair {left_tag} <-> {right_tag}: "
        f"{len(left_y)} vs {len(right_y)} nodes"
    )

    if len(left_y) != len(right_y):
        print("  [FAIL] Different number of nodes.")
        return False

    error = np.max(np.abs((right[:, 0].mean() - L) - left[:, 0].mean()))

    # Compare y coordinates directly.
    error_y = np.max(np.abs(right_y - left_y))

    print(f"  max y mismatch = {error_y:.3e}")

    return error_y < tol

def check_horizontal_pair(bottom_tag, top_tag, L, tol=1e-10):

    _, bottom = curve_nodes(bottom_tag)
    _, top = curve_nodes(top_tag)

    bottom_x = np.sort(bottom[:, 0])
    top_x = np.sort(top[:, 0])

    print(
        f"\nHorizontal pair {bottom_tag} <-> {top_tag}: "
        f"{len(bottom_x)} vs {len(top_x)} nodes"
    )

    if len(bottom_x) != len(top_x):
        print("  [FAIL] Different number of nodes.")
        return False

    error_x = np.max(np.abs(top_x - bottom_x))

    print(f"  max x mismatch = {error_x:.3e}")

    return error_x < tol

def generate_X_mesh(
    t,
    mesh_size_matrix,
    mesh_size_fiber,
    filename="output/X_mesh_cell.msh",
    num_refinements = 0,
    diagnostics=False
):
    """
    Generate a periodic X-shaped fiber geometry in [0,1] x [0,1].

    Physical groups:
        1 -> Fiber
        2 -> Matrix

    The left/right and bottom/top boundaries are identified periodically
    with translations of one unit cell.
    """
    gmsh.initialize()
    gmsh.model.add("X_mesh")

    L = 1.0
    h = t / 2.0
    sqrt2 = math.sqrt(2.0)
    R = 0.3 * t

    # ------------------------------------------------------------
    # Helper: coordinates from diagonal coordinates (s,n)
    # ------------------------------------------------------------

    def point_from_st(s, n, sign):
        if sign == +1:
            # y = x
            x = (s - n) / sqrt2
            y = (s + n) / sqrt2
        else:
            # y = 1-x
            x = (s + n) / sqrt2
            y = (-s + n) / sqrt2 + 1.0
        return x, y

    # ------------------------------------------------------------
    # Fiber 1: strip around y = x
    # ------------------------------------------------------------

    fiber1_pts = []

    corners = [(0.0, -h), (sqrt2, -h), (sqrt2, +h), (0.0, +h)]

    for s, n in corners:
        x, y = point_from_st(s, n, +1)
        p = gmsh.model.occ.addPoint(x, y, 0)
        fiber1_pts.append(p)

    lines1 = []
    for i in range(4):
        lines1.append(
            gmsh.model.occ.addLine(
                fiber1_pts[i],
                fiber1_pts[(i + 1) % 4],
            )
        )

    loop1 = gmsh.model.occ.addCurveLoop(lines1)
    surf1 = gmsh.model.occ.addPlaneSurface([loop1])

    # ------------------------------------------------------------
    # Fiber 2: strip around y = 1-x
    # ------------------------------------------------------------

    fiber2_pts = []

    for s, n in corners:
        x, y = point_from_st(s, n, -1)
        p = gmsh.model.occ.addPoint(x, y, 0)
        fiber2_pts.append(p)

    lines2 = []

    for i in range(4):
        lines2.append(
            gmsh.model.occ.addLine(
                fiber2_pts[i],
                fiber2_pts[(i + 1) % 4],
            )
        )

    loop2 = gmsh.model.occ.addCurveLoop(lines2)
    surf2 = gmsh.model.occ.addPlaneSurface([loop2])

    # ------------------------------------------------------------
    # Unit square
    # ------------------------------------------------------------

    square = gmsh.model.occ.addRectangle(0., 0., 0., L, L)

    gmsh.model.occ.synchronize()

    # ------------------------------------------------------------
    # Clip the two fibers to the unit square
    # ------------------------------------------------------------

    fiber1_cut, _ = gmsh.model.occ.intersect(
        [(2, surf1)],
        [(2, square)],
        removeObject=True,
        removeTool=False,
    )

    fiber2_cut, _ = gmsh.model.occ.intersect(
        [(2, surf2)],
        [(2, square)],
        removeObject=True,
        removeTool=False,
    )

    gmsh.model.occ.synchronize()
    
    # ------------------------------------------------------------
    # Fuse the two fibers
    # ------------------------------------------------------------

    fiber_union, _ = gmsh.model.occ.fuse(
        fiber1_cut,
        fiber2_cut,
        removeObject=True,
        removeTool=True,
    )

    gmsh.model.occ.synchronize()

    # SMOOTH CORNERS BEFORE THE DEFINITION OF Matrix (To be done later, now ignore this comment!!)

    # ------------------------------------------------------------
    # Matrix = square minus fibers
    # ------------------------------------------------------------

    matrix, _ = gmsh.model.occ.cut(
        [(2, square)],
        fiber_union,
        removeObject=True,
        removeTool=False,
    )

    gmsh.model.occ.synchronize()

    # ------------------------------------------------------------
    # Physical groups
    # ------------------------------------------------------------

    fiber_surfaces = [tag for dim, tag in fiber_union if dim == 2]
    matrix_surfaces = [tag for dim, tag in matrix if dim == 2]

    if fiber_surfaces:
        gmsh.model.addPhysicalGroup(2, fiber_surfaces, 1)
        gmsh.model.setPhysicalName(2, 1, "Fiber")

    if matrix_surfaces:
        gmsh.model.addPhysicalGroup(2, matrix_surfaces, 2)
        gmsh.model.setPhysicalName(2, 2, "Matrix")

    # ------------------------------------------------------------
    # Mesh-size control
    # ------------------------------------------------------------

    fiber_points = []

    for dim, tag in fiber_union:
        boundary = gmsh.model.getBoundary([(dim, tag)], recursive=True)
        for bdim, btag in boundary:
            if bdim == 0:
                fiber_points.append(btag)

    fiber_points = list(set(fiber_points))

    if fiber_points:
        gmsh.model.mesh.setSize([(0, p) for p in fiber_points], mesh_size_fiber)

    gmsh.option.setNumber("Mesh.MeshSizeMax", mesh_size_matrix)

    # ============================================================
    # Identify periodic boundary curves
    # ============================================================

    tol = 1e-6

    # Get all curves bounding the final matrix surfaces.
    boundary_curves = set()

    for surface_tag in matrix_surfaces:

        boundary = gmsh.model.getBoundary([(2, surface_tag)], combined=False, oriented=False, recursive=False)

        for dim, tag in boundary:
            if dim == 1:
                boundary_curves.add(tag)

    boundary_curves = sorted(boundary_curves)


    left_curves = []
    right_curves = []
    bottom_curves = []
    top_curves = []


    # Classify curves from their endpoint coordinates.
    for curve_tag in boundary_curves:

        endpoints = gmsh.model.getBoundary([(1, curve_tag)], combined=False, oriented=False, recursive=False)
        point_tags = [tag for dim, tag in endpoints if dim == 0]

        if len(point_tags) != 2:
            continue

        coordinates = []

        for point_tag in point_tags:
            xyz = gmsh.model.getValue(0, point_tag, [])
            coordinates.append((xyz[0], xyz[1]))

        (x1, y1), (x2, y2) = coordinates


        # --------------------------------------------------------
        # Vertical curve
        # --------------------------------------------------------

        if abs(x1 - x2) < tol:
            x = 0.5 * (x1 + x2)
            if abs(x) < tol:
                left_curves.append(curve_tag)
            elif abs(x - L) < tol:
                right_curves.append(curve_tag)


        # --------------------------------------------------------
        # Horizontal curve
        # --------------------------------------------------------

        elif abs(y1 - y2) < tol:
            y = 0.5 * (y1 + y2)
            if abs(y) < tol:
                bottom_curves.append(curve_tag)
            elif abs(y - L) < tol:
                top_curves.append(curve_tag)


    # ------------------------------------------------------------
    # Sort corresponding curves consistently
    # ------------------------------------------------------------

    def curve_midpoint(curve_tag):

        endpoints = gmsh.model.getBoundary([(1, curve_tag)], combined=False, oriented=False, recursive=False)

        point_tags = [tag for dim, tag in endpoints if dim == 0]

        coords = [gmsh.model.getValue(0, p, []) for p in point_tags]

        x = 0.5 * (coords[0][0] + coords[1][0])
        y = 0.5 * (coords[0][1] + coords[1][1])

        return x, y

    left_curves.sort(key=lambda c: curve_midpoint(c)[1])
    right_curves.sort(key=lambda c: curve_midpoint(c)[1])
    bottom_curves.sort(key=lambda c: curve_midpoint(c)[0])
    top_curves.sort(key=lambda c: curve_midpoint(c)[0])

    if diagnostics==True:
        print("\nPeriodic curves:")
        print("  left  :", left_curves)
        print("  right :", right_curves)
        print("  bottom:", bottom_curves)
        print("  top   :", top_curves)


    # ------------------------------------------------------------
    # Sanity checks
    # ------------------------------------------------------------

    if len(left_curves) == 0:
        raise RuntimeError("No left boundary curves found.")

    if len(right_curves) == 0:
        raise RuntimeError("No right boundary curves found.")

    if len(bottom_curves) == 0:
        raise RuntimeError("No bottom boundary curves found.")

    if len(top_curves) == 0:
        raise RuntimeError("No top boundary curves found.")

    if len(left_curves) != len(right_curves):

        raise RuntimeError(
            f"Left/right periodic boundaries have "
            f"different numbers of curves: "
            f"{len(left_curves)} vs {len(right_curves)}."
        )

    if len(bottom_curves) != len(top_curves):

        raise RuntimeError(
            f"Bottom/top periodic boundaries have "
            f"different numbers of curves: "
            f"{len(bottom_curves)} vs {len(top_curves)}."
        )

    # ============================================================
    # Impose periodicity
    # ============================================================

    translation_x = [
        1, 0, 0, L,
        0, 1, 0,  0,
        0, 0, 1,  0,
        0, 0, 0,  1,
    ]

    translation_y = [
        1, 0, 0,  0,
        0, 1, 0, L,
        0, 0, 1,  0,
        0, 0, 0,  1,
    ]

    # Right -> Left
    gmsh.model.mesh.setPeriodic(1, right_curves, left_curves, translation_x)
    # Top -> Bottom
    gmsh.model.mesh.setPeriodic(1, top_curves, bottom_curves, translation_y)

    print("\n[OK] Periodic boundary conditions defined.")

    # ============================================================
    # Generate mesh
    # ============================================================
    # gmsh.option.setNumber("Mesh.MshFileVersion", 4.1)
    gmsh.model.mesh.generate(2)

    # ------------------------------------------------------------
    # Save
    # ------------------------------------------------------------

    gmsh.write(filename)

    gmsh.finalize()

def generate_periodic_X_mesh(
    filename,
    nx_macro,
    ny_macro,
    l_micro,
    t,
    mesh_size_matrix,
    mesh_size_fiber,
    num_refinements=0,
    diagnostics=False
):
    """
    Generate an explicitly periodic X-microstructure.

    The domain is

        [0, nx_macro*l_micro] x [0, ny_macro*l_micro]

    and every micro-cell contains the same X-shaped fiber.

    Parameters
    ----------
    filename : str
        Output .msh file.

    nx_macro, ny_macro : int
        Number of micro-cells in x and y.

    l_micro : float
        Physical side length of one micro-cell.

    t : float
        Physical fiber thickness.

    mesh_size_matrix : float
        Target mesh size in the matrix.

    mesh_size_fiber : float
        Target mesh size near the fiber boundary.
    """

    gmsh.initialize()
    gmsh.model.add("Periodic_X_mesh")

    Lx = nx_macro * l_micro
    Ly = ny_macro * l_micro

    t = t*l_micro
    
    mesh_size_matrix *= l_micro
    mesh_size_fiber *= l_micro

    h = t/2
    sqrt2 = math.sqrt(2.0)
    R = 0.3 * t

    # ========================================================
    # Helper: construct one X inside one cell
    # ========================================================

    def make_local_X(x0, y0):
        """
        Construct one X-shaped fiber in

            [x0, x0+l_micro] x [y0, y0+l_micro].
        """

        # ----------------------------------------------------
        # Diagonal y = x
        # ----------------------------------------------------

        def point_from_st(s, n, sign):
            if sign == +1:
                x = (s - n) / sqrt2
                y = (s + n) / sqrt2
            else:
                x = (s + n) / sqrt2
                y = (-s + n) / sqrt2 + l_micro
            return x0 + x, y0 + y

        # In local coordinates the diagonal length is
        # sqrt(2)*l_micro.
        smax = sqrt2 * l_micro
        corners = [(0.0, -h),(smax, -h),(smax, +h),(0.0, +h)]

        # ----------------------------------------------------
        # First diagonal
        # ----------------------------------------------------

        fiber1_pts = []
        for s, n in corners:
            x, y = point_from_st(s, n, +1)
            p = gmsh.model.occ.addPoint(x, y, 0)
            fiber1_pts.append(p)

        lines1 = []
        for i in range(4):
            lines1.append(
                gmsh.model.occ.addLine(fiber1_pts[i], fiber1_pts[(i + 1) % 4]))

        loop1 = gmsh.model.occ.addCurveLoop(lines1)
        surf1 = gmsh.model.occ.addPlaneSurface([loop1])

        # ----------------------------------------------------
        # Second diagonal
        # ----------------------------------------------------

        fiber2_pts = []
        for s, n in corners:
            x, y = point_from_st(s, n, -1)
            p = gmsh.model.occ.addPoint(x, y, 0)
            fiber2_pts.append(p)

        lines2 = []
        for i in range(4):
            lines2.append(
                gmsh.model.occ.addLine(fiber2_pts[i], fiber2_pts[(i + 1) % 4]))
                
        loop2 = gmsh.model.occ.addCurveLoop(lines2)
        surf2 = gmsh.model.occ.addPlaneSurface([loop2])

        # ----------------------------------------------------
        # Local cell
        # ----------------------------------------------------

        square = gmsh.model.occ.addRectangle(x0, y0, 0, l_micro, l_micro)
        gmsh.model.occ.synchronize()

        # ----------------------------------------------------
        # Clip first diagonal to cell
        # ----------------------------------------------------

        fiber1_cut, _ = gmsh.model.occ.intersect([(2, surf1)], [(2, square)], removeObject=True, removeTool=False)
        gmsh.model.occ.synchronize()

        # ----------------------------------------------------
        # Clip second diagonal to cell
        # ----------------------------------------------------

        fiber2_cut, _ = gmsh.model.occ.intersect([(2, surf2)], [(2, square)], removeObject=True, removeTool=False)
        gmsh.model.occ.synchronize()

        # ----------------------------------------------------
        # Fuse the two diagonals
        # ----------------------------------------------------

        fiber_union, _ = gmsh.model.occ.fuse(
            fiber1_cut,
            fiber2_cut,
            removeObject=True,
            removeTool=True,
        )

        gmsh.model.occ.synchronize()

        return fiber_union
        
        # # ------------------------------------------------------------
        # # Smoothen the corners (TO BE DONE LATER, IGNORE FOR NOW)
        # # ------------------------------------------------------------

    # ========================================================
    # Construct all X cells
    # ========================================================

    all_fibers = []
    for ix in range(nx_macro):
        for iy in range(ny_macro):
            x0 = ix * l_micro
            y0 = iy * l_micro
            fiber = make_local_X(x0, y0)
            all_fibers.extend(fiber)

    gmsh.model.occ.synchronize()

    # ========================================================
    # Fuse all fibers
    # ========================================================

    fiber_surfaces_input = [
        entity
        for entity in all_fibers
        if entity[0] == 2
    ]

    fiber_union, _ = gmsh.model.occ.fuse(
        fiber_surfaces_input[:1],
        fiber_surfaces_input[1:],
        removeObject=True,
        removeTool=True,
    )

    gmsh.model.occ.synchronize()

    # ========================================================
    # Macro-domain rectangle
    # ========================================================

    domain = gmsh.model.occ.addRectangle(0, 0, 0, Lx, Ly)
    gmsh.model.occ.synchronize()

    # ========================================================
    # Matrix = domain - fibers
    # ========================================================

    matrix, _ = gmsh.model.occ.cut(
        [(2, domain)],
        fiber_union,
        removeObject=True,
        removeTool=False,
    )

    gmsh.model.occ.synchronize()

    # ========================================================
    # Physical regions
    # ========================================================

    fiber_surfaces = [
        tag
        for dim, tag in fiber_union
        if dim == 2
    ]

    matrix_surfaces = [
        tag
        for dim, tag in matrix
        if dim == 2
    ]

    if not fiber_surfaces:
        raise RuntimeError("No fiber surfaces found.")

    if not matrix_surfaces:
        raise RuntimeError("No matrix surfaces found.")

    gmsh.model.addPhysicalGroup(2, fiber_surfaces, 1)
    gmsh.model.setPhysicalName(2, 1,  "Fiber")
    gmsh.model.addPhysicalGroup(2, matrix_surfaces, 2)
    gmsh.model.setPhysicalName(2, 2, "Matrix")

    # ========================================================
    # Mesh-size control
    # ========================================================

    fiber_points = []

    for dim, tag in fiber_union:
        boundary = gmsh.model.getBoundary([(dim, tag)], recursive=True)
        for bdim, btag in boundary:
            if bdim == 0:
                fiber_points.append(btag)

    fiber_points = list(set(fiber_points))
    if fiber_points:
        gmsh.model.mesh.setSize(
            [(0, p) for p in fiber_points],
            mesh_size_fiber,
        )

    gmsh.option.setNumber("Mesh.MeshSizeMax", mesh_size_matrix)
    
    # ============================================================
    # Identify periodic boundary curves of the fine-scale domain
    # ============================================================

    tol = 1e-6

    boundary_curves = set()

    for surface_tag in matrix_surfaces:
        boundary = gmsh.model.getBoundary([(2, surface_tag)], combined=False, oriented=False, recursive=False)
        for dim, tag in boundary:
            if dim == 1:
                boundary_curves.add(tag)

    boundary_curves = sorted(boundary_curves)

    left_curves = []
    right_curves = []
    bottom_curves = []
    top_curves = []

    for curve_tag in boundary_curves:

        endpoints = gmsh.model.getBoundary(
            [(1, curve_tag)],
            combined=False,
            oriented=False,
            recursive=False,
        )

        point_tags = [
            tag for dim, tag in endpoints
            if dim == 0
        ]

        if len(point_tags) != 2:
            continue

        coordinates = []
        for point_tag in point_tags:
            xyz = gmsh.model.getValue(0, point_tag, [])
            coordinates.append((xyz[0], xyz[1]))
        (x1, y1), (x2, y2) = coordinates

        # --------------------------------------------------------
        # Vertical curve
        # --------------------------------------------------------

        if abs(x1 - x2) < tol:
            x = 0.5 * (x1 + x2)
            if abs(x) < tol:
                left_curves.append(curve_tag)
            elif abs(x - Lx) < tol:
                right_curves.append(curve_tag)

        # --------------------------------------------------------
        # Horizontal curve
        # --------------------------------------------------------

        elif abs(y1 - y2) < tol:
            y = 0.5 * (y1 + y2)
            if abs(y) < tol:
                bottom_curves.append(curve_tag)
            elif abs(y - Ly) < tol:
                top_curves.append(curve_tag)


    def curve_midpoint(curve_tag):

        endpoints = gmsh.model.getBoundary([(1, curve_tag)], combined=False, oriented=False, recursive=False)
        point_tags = [
            tag for dim, tag in endpoints
            if dim == 0
        ]

        coords = [
            gmsh.model.getValue(0, p, [])
            for p in point_tags
        ]

        x = 0.5 * (coords[0][0] + coords[1][0])
        y = 0.5 * (coords[0][1] + coords[1][1])

        return x, y


    left_curves.sort(key=lambda c: curve_midpoint(c)[1])
    right_curves.sort(key=lambda c: curve_midpoint(c)[1])
    bottom_curves.sort(key=lambda c: curve_midpoint(c)[0])
    top_curves.sort(key=lambda c: curve_midpoint(c)[0])

    if diagnostics==True:
        print("\nFine-scale periodic curves:")
        print("  left  :", left_curves)
        print("  right :", right_curves)
        print("  bottom:", bottom_curves)
        print("  top   :", top_curves)


    # ------------------------------------------------------------
    # Sanity checks
    # ------------------------------------------------------------

    if len(left_curves) == 0:
        raise RuntimeError("No left boundary curves found.")

    if len(right_curves) == 0:
        raise RuntimeError("No right boundary curves found.")

    if len(bottom_curves) == 0:
        raise RuntimeError("No bottom boundary curves found.")

    if len(top_curves) == 0:
        raise RuntimeError("No top boundary curves found.")

    if len(left_curves) != len(right_curves):
        raise RuntimeError(
            f"Left/right periodic boundaries have "
            f"different numbers of curves: "
            f"{len(left_curves)} vs {len(right_curves)}."
        )

    if len(bottom_curves) != len(top_curves):
        raise RuntimeError(
            f"Bottom/top periodic boundaries have "
            f"different numbers of curves: "
            f"{len(bottom_curves)} vs {len(top_curves)}."
        )
    
    translation_x = [
        1, 0, 0, Lx,
        0, 1, 0,  0,
        0, 0, 1,  0,
        0, 0, 0,  1,
    ]

    translation_y = [
        1, 0, 0,  0,
        0, 1, 0, Ly,
        0, 0, 1,  0,
        0, 0, 0,  1,
    ]

    gmsh.model.mesh.setPeriodic(
        1,
        right_curves,
        left_curves,
        translation_x,
    )

    gmsh.model.mesh.setPeriodic(
        1,
        top_curves,
        bottom_curves,
        translation_y,
    )

    # ========================================================
    # Generate mesh with nested refinements
    # ========================================================

    gmsh.model.mesh.generate(2)

    for level in range(num_refinements + 1):

        level_filename = filename.replace(
            ".msh",
            f"_level{level}.msh",
        )

        print(
            f"Writing mesh refinement level {level}: "
            f"{level_filename}"
        )

        gmsh.write(level_filename)
        
        level_filename_vtk = level_filename.replace(
            ".msh",
            f"_level{level}.vtk",
        )
        
        gmsh.write(level_filename_vtk)

        if level < num_refinements:
            gmsh.model.mesh.refine()

    # ========================================================
    # Write
    # ========================================================

    gmsh.write(filename)

    gmsh.finalize()

def generate_periodic_66_mesh(
    filename,
    nx_macro,
    ny_macro,
    l_micro,
    l,
    R,
    t,
    mesh_size_matrix,
    mesh_size_fiber,
    mesh_size_interface=None,   # matrix size right next to the fibers (relative to l_micro)
    transition_width=None,      # distance over which it grows to mesh_size_matrix (relative to l_micro)
    mirror=False,
    num_refinements=0,
    diagnostics=False,
):
    """
    Generate an explicit fine-scale periodic mesh of a chiral "66" geometry.

    Geometry:
        - Each microcell contains a circular ring and four tangent ligaments.
        - The geometry is normalized exactly as in inside_66().
        - The ligament centerlines hit the midpoint of the corresponding
          microcell edges.
        - Ligaments are extended slightly beyond their nominal endpoints so
          neighboring microcells overlap before the global Boolean union.
        - Only the outer boundary of the complete fine-scale domain is made
          periodic.

    Physical groups:
        1 = Fiber
        2 = Matrix

    Parameters
    ----------
    filename : str
        Output .msh filename.

    nx_macro, ny_macro : int
        Number of microcells in x/y.

    l_micro : float
        Physical microcell size.

    l, R, t : float
        Dimensionless parameters of the 66 geometry before normalization,
        matching inside_66().

    mesh_size_matrix : float
        Mesh size in the matrix, relative to l_micro.

    mesh_size_fiber : float
        Mesh size near fiber boundary points, relative to l_micro.

    mirror : bool
        If True, mirror the local geometry with respect to x = 1/2.

    num_refinements : int
        Number of nested Gmsh mesh refinements.

    diagnostics : bool
        Print information about the generated geometry and periodic boundaries.
    """

    import gmsh
    import math

    gmsh.initialize()
    gmsh.model.add("Periodic_66_mesh")

    # ==============================================================
    # Global fine-scale domain
    # ==============================================================

    Lx = nx_macro * l_micro
    Ly = ny_macro * l_micro

    # ==============================================================
    # Normalize geometry exactly as required for midpoint crossing
    #
    # We want:
    #
    #     sqrt(R^2 + l^2/4) = 1/2
    #
    # after normalization.
    # ==============================================================

    a = 2.0 * math.sqrt(l*l / 4.0 + R*R)

    l = l / a
    R = R / a
    t = t / a

    if R - t / 2.0 <= 0.0:
        gmsh.finalize()
        raise ValueError(
            "Invalid 66 geometry: R - t/2 <= 0. "
            "The inner radius of the ring must be positive."
        )

    # Physical mesh sizes
    mesh_size_matrix = mesh_size_matrix * l_micro
    mesh_size_fiber = mesh_size_fiber * l_micro

    # Physical geometry parameters
    l_phys = l * l_micro
    R_phys = R * l_micro
    t_phys = t * l_micro

    # ==============================================================
    # Small extension of each ligament.
    #
    # The nominal centerline has length l/2 starting from the ring
    # tangency point. We extend it beyond the cell boundary so that
    # neighboring copies have a finite-area overlap.
    #
    # This extension is only a CAD construction device. The final
    # geometry is clipped to the global domain.
    # ==============================================================

    beam_extension = 0.5 * t_phys

    # ==============================================================
    # Local 66 geometry
    # ==============================================================

    def make_local_66(x0, y0):

        xc = x0 + 0.5 * l_micro
        yc = y0 + 0.5 * l_micro

        # ----------------------------------------------------------
        # Ring
        # ----------------------------------------------------------

        outer_radius = R_phys + 0.5 * t_phys
        inner_radius = R_phys - 0.5 * t_phys

        outer_ring = gmsh.model.occ.addDisk(
            xc,
            yc,
            0.0,
            outer_radius,
            outer_radius,
        )

        inner_ring = gmsh.model.occ.addDisk(
            xc,
            yc,
            0.0,
            inner_radius,
            inner_radius,
        )

        ring, _ = gmsh.model.occ.cut(
            [(2, outer_ring)],
            [(2, inner_ring)],
            removeObject=True,
            removeTool=True,
        )

        gmsh.model.occ.synchronize()

        # ----------------------------------------------------------
        # Four ligaments
        #
        # theta is chosen so that the unextended centerline reaches
        # exactly the midpoint of a cell edge.
        # ----------------------------------------------------------

        theta = (
            math.pi / 2.0
            - math.atan(l / (2.0 * R))
        )

        beams = []

        for k in range(4):

            angle = theta + k * math.pi / 2.0

            # Mirror with respect to x = xc.
            #
            # For the capsule geometry, reflecting the beam angle
            # in this way is equivalent to reflecting the complete
            # centerline segment because the capsule is invariant
            # under reversal of its tangent direction.
            if mirror:
                angle = math.pi - angle

            # ------------------------------------------------------
            # Tangency point on the centerline of the ring
            # ------------------------------------------------------

            x_center = (
                xc
                + R_phys * math.cos(angle)
            )

            y_center = (
                yc
                + R_phys * math.sin(angle)
            )

            # Tangential direction
            tx = -math.sin(angle)
            ty =  math.cos(angle)

            # ------------------------------------------------------
            # Extended centerline
            #
            # Nominal:
            #
            #     s in [0, l/2]
            #
            # Actual CAD construction:
            #
            #     s in [-extension, l/2 + extension]
            # ------------------------------------------------------

            s_start = -beam_extension
            s_end = l_phys / 2.0 + beam_extension

            x_start = x_center + s_start * tx
            y_start = y_center + s_start * ty

            x_end = x_center + s_end * tx
            y_end = y_center + s_end * ty

            # ------------------------------------------------------
            # Normal to ligament centerline
            # ------------------------------------------------------

            nx = -ty
            ny = tx

            half_t = 0.5 * t_phys

            # ------------------------------------------------------
            # Rectangle around centerline
            # ------------------------------------------------------

            p0m = gmsh.model.occ.addPoint(
                x_start - half_t * nx,
                y_start - half_t * ny,
                0.0,
            )

            p0p = gmsh.model.occ.addPoint(
                x_start + half_t * nx,
                y_start + half_t * ny,
                0.0,
            )

            p1p = gmsh.model.occ.addPoint(
                x_end + half_t * nx,
                y_end + half_t * ny,
                0.0,
            )

            p1m = gmsh.model.occ.addPoint(
                x_end - half_t * nx,
                y_end - half_t * ny,
                0.0,
            )

            line0 = gmsh.model.occ.addLine(p0m, p0p)
            line1 = gmsh.model.occ.addLine(p0p, p1p)
            line2 = gmsh.model.occ.addLine(p1p, p1m)
            line3 = gmsh.model.occ.addLine(p1m, p0m)

            loop = gmsh.model.occ.addCurveLoop([
                line0,
                line1,
                line2,
                line3,
            ])

            rectangle = gmsh.model.occ.addPlaneSurface([loop])

            # ------------------------------------------------------
            # Circular caps.
            #
            # Together with the rectangle these reproduce exactly
            # the distance-to-segment <= t/2 capsule used in
            # inside_66().
            # ------------------------------------------------------

            disk_start = gmsh.model.occ.addDisk(
                x_start,
                y_start,
                0.0,
                half_t,
                half_t,
            )

            disk_end = gmsh.model.occ.addDisk(
                x_end,
                y_end,
                0.0,
                half_t,
                half_t,
            )

            beam, _ = gmsh.model.occ.fuse(
                [(2, rectangle)],
                [
                    (2, disk_start),
                    (2, disk_end),
                ],
                removeObject=True,
                removeTool=True,
            )

            beams.extend(beam)

            gmsh.model.occ.synchronize()

        # ----------------------------------------------------------
        # Combine ring + four beams for this microcell.
        #
        # IMPORTANT:
        # No intersection with the local square is performed here.
        # Ligaments are intentionally allowed to cross the artificial
        # microcell boundary.
        # ----------------------------------------------------------

        fiber_entities = [
            entity
            for entity in ring
            if entity[0] == 2
        ]

        fiber_entities.extend(beams)

        if len(fiber_entities) == 1:
            local_fiber = fiber_entities
        else:
            local_fiber, _ = gmsh.model.occ.fuse(
                fiber_entities[:1],
                fiber_entities[1:],
                removeObject=True,
                removeTool=True,
            )

        gmsh.model.occ.synchronize()

        return local_fiber

    # ==============================================================
    # Construct all local fibers
    # ==============================================================

    all_fiber_entities = []

    for ix in range(nx_macro):
        for iy in range(ny_macro):

            x0 = ix * l_micro
            y0 = iy * l_micro

            local_fiber = make_local_66(x0, y0)

            all_fiber_entities.extend(local_fiber)

    gmsh.model.occ.synchronize()

    if diagnostics:
        print("\nConstructed local 66 geometries:")
        print("  number of microcells =", nx_macro * ny_macro)
        print("  number of fiber entities before global fuse =",
              len(all_fiber_entities))

    # ==============================================================
    # Global fiber union
    #
    # This is the important part:
    #
    # neighboring ligaments overlap before being clipped, so the
    # internal microcell interfaces do not become artificial
    # geometric boundaries.
    # ==============================================================

    if len(all_fiber_entities) == 0:
        gmsh.finalize()
        raise RuntimeError("No fiber entities were generated.")

    if len(all_fiber_entities) == 1:

        fiber_union = all_fiber_entities

    else:

        fiber_union, _ = gmsh.model.occ.fuse(
            all_fiber_entities[:1],
            all_fiber_entities[1:],
            removeObject=True,
            removeTool=True,
        )

    gmsh.model.occ.synchronize()

    # --------------------------------------------------------------
    # Remove coincident/duplicate CAD entities.
    # --------------------------------------------------------------

    gmsh.model.occ.removeAllDuplicates()
    gmsh.model.occ.synchronize()

    if diagnostics:
        print(
            "  fiber entities after global fuse =",
            len(fiber_union)
        )

    # ==============================================================
    # Global square
    # ==============================================================

    domain = gmsh.model.occ.addRectangle(
        0.0,
        0.0,
        0.0,
        Lx,
        Ly,
    )

    gmsh.model.occ.synchronize()

    # ==============================================================
    # Clip fiber ONLY at the external domain boundary
    # ==============================================================

    fiber_clipped, _ = gmsh.model.occ.intersect(
        fiber_union,
        [(2, domain)],
        removeObject=True,
        removeTool=False,
    )

    gmsh.model.occ.synchronize()

    if not fiber_clipped:
        gmsh.finalize()
        raise RuntimeError(
            "Fiber clipping produced no surfaces."
        )

    fiber_surfaces = [
        tag
        for dim, tag in fiber_clipped
        if dim == 2
    ]

    if not fiber_surfaces:
        gmsh.finalize()
        raise RuntimeError(
            "No fiber surfaces found after clipping."
        )

    # ==============================================================
    # Matrix = domain - fiber
    # ==============================================================

    matrix, _ = gmsh.model.occ.cut(
        [(2, domain)],
        fiber_clipped,
        removeObject=True,
        removeTool=False,
    )

    gmsh.model.occ.synchronize()

    matrix_surfaces = [
        tag
        for dim, tag in matrix
        if dim == 2
    ]

    if not matrix_surfaces:
        gmsh.finalize()
        raise RuntimeError(
            "No matrix surfaces found."
        )

    # ==============================================================
    # Physical groups
    # ==============================================================

    gmsh.model.addPhysicalGroup(
        2,
        fiber_surfaces,
        1,
    )
    gmsh.model.setPhysicalName(
        2,
        1,
        "Fiber",
    )

    gmsh.model.addPhysicalGroup(
        2,
        matrix_surfaces,
        2,
    )
    gmsh.model.setPhysicalName(
        2,
        2,
        "Matrix",
    )

    # # ==============================================================
    # # Mesh-size control
    # # ==============================================================

    # # Use fiber boundary points to control the fiber mesh.
    # fiber_points = []

    # for surface_tag in fiber_surfaces:

    #     boundary = gmsh.model.getBoundary(
    #         [(2, surface_tag)],
    #         combined=True,
    #         oriented=False,
    #         recursive=True,
    #     )

    #     for dim, tag in boundary:
    #         if dim == 0:
    #             fiber_points.append(tag)

    # fiber_points = sorted(set(fiber_points))

    # if fiber_points:
    #     gmsh.model.mesh.setSize(
    #         [
    #             (0, point_tag)
    #             for point_tag in fiber_points
    #         ],
    #         mesh_size_fiber,
    #     )

    # # Matrix mesh size
    # gmsh.option.setNumber(
    #     "Mesh.MeshSizeMax",
    #     mesh_size_matrix,
    # )

    # ==============================================================
    # Mesh-size control through background fields
    #
    #   - inside the fiber : size = mesh_size_fiber
    #   - in the matrix    : size = mesh_size_interface at the fiber
    #                        boundary, growing linearly to
    #                        mesh_size_matrix over transition_width
    #
    # mesh_size_matrix and mesh_size_fiber were already multiplied
    # by l_micro above, so the other two are scaled here.
    # ==============================================================

    h_fiber = mesh_size_fiber
    h_matrix = mesh_size_matrix
    h_int = (
        mesh_size_interface * l_micro
        if mesh_size_interface is not None
        else h_matrix
    )
    width = (
        transition_width * l_micro
        if transition_width is not None
        else 3.0 * h_matrix
    )

    # Curves bounding the fiber (interface + pieces on the outer boundary)
    fiber_curves = set()

    for surface_tag in fiber_surfaces:
        for dim, tag in gmsh.model.getBoundary(
            [(2, surface_tag)],
            combined=False,
            oriented=False,
            recursive=False,
        ):
            if dim == 1:
                fiber_curves.add(tag)

    fld = gmsh.model.mesh.field

    # Distance to the fiber boundary
    f_dist = fld.add("Distance")
    fld.setNumbers(f_dist, "CurvesList", sorted(fiber_curves))
    fld.setNumber(f_dist, "Sampling", 200)

    # Matrix size: h_int at the interface -> h_matrix far away
    f_thr = fld.add("Threshold")
    fld.setNumber(f_thr, "InField", f_dist)
    fld.setNumber(f_thr, "SizeMin", h_int)
    fld.setNumber(f_thr, "SizeMax", h_matrix)
    fld.setNumber(f_thr, "DistMin", 0.0)
    fld.setNumber(f_thr, "DistMax", width)

    # Uniform size inside the fiber only
    f_const = fld.add("MathEval")
    fld.setString(f_const, "F", str(h_fiber))

    f_fiber = fld.add("Restrict")
    fld.setNumber(f_fiber, "InField", f_const)
    fld.setNumbers(f_fiber, "SurfacesList", fiber_surfaces)

    # Final size = min of the two
    f_min = fld.add("Min")
    fld.setNumbers(f_min, "FieldsList", [f_thr, f_fiber])
    fld.setAsBackgroundMesh(f_min)

    # Switch off every other size source, otherwise it competes
    # with the background field
    gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
    gmsh.option.setNumber("Mesh.MeshSizeMin", 0.0)
    gmsh.option.setNumber("Mesh.MeshSizeMax", max(h_fiber, h_matrix))

    # Better triangle quality
    gmsh.option.setNumber("Mesh.Algorithm", 6)   # Frontal-Delaunay
    gmsh.option.setNumber("Mesh.Optimize", 1)
    gmsh.option.setNumber("Mesh.Smoothing", 5)

    # ==============================================================
    # Outer boundary curves
    #
    # Only the external boundary of the complete fine-scale domain
    # is periodic. Internal microcell interfaces are ordinary
    # internal interfaces.
    # ==============================================================

    tol = 1e-6

    boundary_curves = set()

    for surface_tag in matrix_surfaces:

        boundary = gmsh.model.getBoundary(
            [(2, surface_tag)],
            combined=False,
            oriented=False,
            recursive=False,
        )

        for dim, tag in boundary:

            if dim == 1:
                boundary_curves.add(tag)

    boundary_curves = sorted(boundary_curves)

    left_curves = []
    right_curves = []
    bottom_curves = []
    top_curves = []

    for curve_tag in boundary_curves:

        endpoints = gmsh.model.getBoundary(
            [(1, curve_tag)],
            combined=False,
            oriented=False,
            recursive=False,
        )

        point_tags = [
            tag
            for dim, tag in endpoints
            if dim == 0
        ]

        if len(point_tags) != 2:
            continue

        coordinates = []

        for point_tag in point_tags:

            xyz = gmsh.model.getValue(
                0,
                point_tag,
                [],
            )

            coordinates.append(
                (xyz[0], xyz[1])
            )

        (x1, y1), (x2, y2) = coordinates

        # Vertical curve
        if abs(x1 - x2) < tol:

            x = 0.5 * (x1 + x2)

            if abs(x) < tol:
                left_curves.append(curve_tag)

            elif abs(x - Lx) < tol:
                right_curves.append(curve_tag)

        # Horizontal curve
        elif abs(y1 - y2) < tol:

            y = 0.5 * (y1 + y2)

            if abs(y) < tol:
                bottom_curves.append(curve_tag)

            elif abs(y - Ly) < tol:
                top_curves.append(curve_tag)

    # ==============================================================
    # Sort corresponding periodic curves
    # ==============================================================

    def curve_midpoint(curve_tag):

        endpoints = gmsh.model.getBoundary(
            [(1, curve_tag)],
            combined=False,
            oriented=False,
            recursive=False,
        )

        point_tags = [
            tag
            for dim, tag in endpoints
            if dim == 0
        ]

        coords = [
            gmsh.model.getValue(
                0,
                point_tag,
                [],
            )
            for point_tag in point_tags
        ]

        x = 0.5 * (
            coords[0][0] +
            coords[1][0]
        )

        y = 0.5 * (
            coords[0][1] +
            coords[1][1]
        )

        return x, y

    left_curves.sort(
        key=lambda c: curve_midpoint(c)[1]
    )

    right_curves.sort(
        key=lambda c: curve_midpoint(c)[1]
    )

    bottom_curves.sort(
        key=lambda c: curve_midpoint(c)[0]
    )

    top_curves.sort(
        key=lambda c: curve_midpoint(c)[0]
    )

    # ==============================================================
    # Diagnostics
    # ==============================================================

    if diagnostics:

        print("\nFine-scale periodic curves:")
        print("  left  :", left_curves)
        print("  right :", right_curves)
        print("  bottom:", bottom_curves)
        print("  top   :", top_curves)

    if len(left_curves) == 0:
        gmsh.finalize()
        raise RuntimeError(
            "No left boundary curves found."
        )

    if len(right_curves) == 0:
        gmsh.finalize()
        raise RuntimeError(
            "No right boundary curves found."
        )

    if len(bottom_curves) == 0:
        gmsh.finalize()
        raise RuntimeError(
            "No bottom boundary curves found."
        )

    if len(top_curves) == 0:
        gmsh.finalize()
        raise RuntimeError(
            "No top boundary curves found."
        )

    if len(left_curves) != len(right_curves):

        gmsh.finalize()

        raise RuntimeError(
            "Left/right periodic boundaries have different "
            "numbers of curves: "
            f"{len(left_curves)} vs {len(right_curves)}."
        )

    if len(bottom_curves) != len(top_curves):

        gmsh.finalize()

        raise RuntimeError(
            "Bottom/top periodic boundaries have different "
            "numbers of curves: "
            f"{len(bottom_curves)} vs {len(top_curves)}."
        )

    # ==============================================================
    # Periodic transformations
    # ==============================================================

    translation_x = [
        1, 0, 0, Lx,
        0, 1, 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1,
    ]

    translation_y = [
        1, 0, 0, 0,
        0, 1, 0, Ly,
        0, 0, 1, 0,
        0, 0, 0, 1,
    ]

    # Right = slave
    # Left  = master
    gmsh.model.mesh.setPeriodic(
        1,
        right_curves,
        left_curves,
        translation_x,
    )

    # Top = slave
    # Bottom = master
    gmsh.model.mesh.setPeriodic(
        1,
        top_curves,
        bottom_curves,
        translation_y,
    )

    # ==============================================================
    # Generate initial mesh
    # ==============================================================

    gmsh.model.mesh.generate(2)

    # ==============================================================
    # Nested mesh refinements
    # ==============================================================

    for level in range(num_refinements + 1):

        level_filename = filename.replace(
            ".msh",
            f"_level{level}.msh",
        )

        print(
            f"Writing mesh refinement level {level}: "
            f"{level_filename}"
        )

        gmsh.write(level_filename)

        if level < num_refinements:
            gmsh.model.mesh.refine()

    # ==============================================================
    # Final output
    # ==============================================================

    gmsh.write(filename)

    if diagnostics:

        print("\n66 mesh generation completed.")
        print("  Domain:")
        print(f"    Lx = {Lx}")
        print(f"    Ly = {Ly}")

        print("  Normalized geometry:")
        print(f"    l = {l}")
        print(f"    R = {R}")
        print(f"    t = {t}")

        print("  Physical geometry:")
        print(f"    l_phys = {l_phys}")
        print(f"    R_phys = {R_phys}")
        print(f"    t_phys = {t_phys}")

        print("  Beam extension:")
        print(f"    {beam_extension}")

    gmsh.finalize()

def check_periodic_X_mesh(mesh):

    print("\n" + "=" * 60)
    print("CHECKING GMSH MESH IMPORT")
    print("=" * 60)

    print(f"Number of cells    : {mesh.num_cells()}")
    print(f"Number of vertices : {mesh.num_vertices()}")
    # print(f"Geometric dimension: {mesh.geometric_dimension()}")

    dm = mesh.topology_dm

    print("\nDMPlex:")
    print(f"  dimension      : {dm.getDimension()}")
    print(f"  coordinate dim : {dm.getCoordinateDim()}")

    # ---------------------------------------------------------
    # Labels
    # ---------------------------------------------------------

    print("\nDMPlex labels:")

    # for i in range(dm.getNumLabels()):
    #     print(f"  {i}: {dm.getLabelName(i)}")

    # ---------------------------------------------------------
    # Firedrake cell subsets
    # ---------------------------------------------------------

    print("\nTesting Firedrake cell subsets:")

    try:
        fiber = mesh.cell_subset(1)
        print(
            f"  Fiber (tag 1): "
            f"{len(fiber.indices)} cells"
        )
    except Exception as e:
        print("  Fiber (tag 1): FAILED")
        print(f"    {e}")

    try:
        matrix = mesh.cell_subset(2)
        print(
            f"  Matrix (tag 2): "
            f"{len(matrix.indices)} cells"
        )
    except Exception as e:
        print("  Matrix (tag 2): FAILED")
        print(f"    {e}")

    # ---------------------------------------------------------
    # Check that all cells belong to one of the two regions
    # ---------------------------------------------------------

    try:
        fiber = mesh.cell_subset(1)
        matrix = mesh.cell_subset(2)

        n_fiber = len(fiber.indices)
        n_matrix = len(matrix.indices)
        n_total = mesh.num_cells()

        print("\nCell partition:")
        print(f"  Fiber  : {n_fiber}")
        print(f"  Matrix : {n_matrix}")
        print(f"  Total  : {n_total}")
        print(f"  Sum    : {n_fiber + n_matrix}")

        if n_fiber + n_matrix == n_total:
            print("  [OK] Fiber + Matrix cover all cells.")
        else:
            print(
                "  [WARNING] Fiber + Matrix do not "
                "cover all cells."
            )

    except Exception as e:
        print(f"\nCell partition test failed: {e}")

    # ---------------------------------------------------------
    # Exterior facets
    # ---------------------------------------------------------

    print("\nExterior facets:")
    print(
        f"  {mesh.exterior_facets.facets.shape[0]}"
    )

    # ---------------------------------------------------------
    # Coordinates
    # ---------------------------------------------------------

    coords = mesh.coordinates.dat.data_ro

    print("\nCoordinate bounds:")

    print(
        f"  x: [{coords[:, 0].min():.12e}, "
        f"{coords[:, 0].max():.12e}]"
    )

    print(
        f"  y: [{coords[:, 1].min():.12e}, "
        f"{coords[:, 1].max():.12e}]"
    )

    print("=" * 60)

def generate_666_mesh(
    l, R, t,
    mesh_size_matrix,
    mesh_size_fiber,
    filename="output/66_mesh_cell.msh",
    num_refinements=0,
    diagnostics=False,
    beta=math.pi / 4,
    mirror=False,
):
    """
    Generate the local 66-chiral geometry in [0,1] x [0,1].

    The geometry consists of:
        - a central circular ring,
        - four tangent ligaments,
        - the corresponding chiral arrangement controlled by beta.

    Physical groups:
        1 -> Fiber
        2 -> Matrix

    Unlike generate_periodic_66_mesh(), this function does NOT impose
    periodicity on the outer boundary.

    The geometry follows the latest periodic construction:
        a = 2*sqrt(l^2/4 + R^2)

    and uses overlapping ligament extensions before the final clipping.
    This is important to avoid the very thin/skewed triangles that appeared
    when the ligaments were clipped independently at the cell boundary.
    """

    gmsh.initialize()
    gmsh.model.add("66_mesh")

    # ============================================================
    # Basic parameters
    # ============================================================

    L = 1.0

    # Physical thickness of the ligaments.
    t_phys = t

    # Half thickness.
    h = 0.5 * t_phys

    # ------------------------------------------------------------
    # Latest corrected normalization.
    #
    # This is deliberately WITHOUT the old "err" correction.
    # ------------------------------------------------------------

    a = 2.0 * math.sqrt((l / 2.0)**2 + R**2)

    # Beam extension used in the latest periodic construction.
    #
    # The extension makes neighbouring ligaments overlap before
    # the global Boolean operation.
    beam_extension = 0.5 * t_phys

    # ============================================================
    # Coordinate transformation
    #
    # Work first in the physical local cell
    #
    #     [-a/2, a/2] x [-a/2, a/2]
    #
    # and finally map it to [0,1] x [0,1].
    # ============================================================

    def physical_to_unit(x, y):

        return (
            (x + 0.5 * a) / a,
            (y + 0.5 * a) / a,
        )

    # ============================================================
    # Unit-cell square in physical coordinates
    # ============================================================

    square = gmsh.model.occ.addRectangle(
        -0.5 * a,
        -0.5 * a,
        0.0,
        a,
        a,
    )

    # ============================================================
    # Central ring
    # ============================================================

    # Outer and inner radii.
    #
    # The ring thickness is t.
    R_outer = R + h
    R_inner = R - h

    outer_disk = gmsh.model.occ.addDisk(
        0.0,
        0.0,
        0.0,
        R_outer,
        R_outer,
    )

    inner_disk = gmsh.model.occ.addDisk(
        0.0,
        0.0,
        0.0,
        R_inner,
        R_inner,
    )

    ring, _ = gmsh.model.occ.cut(
        [(2, outer_disk)],
        [(2, inner_disk)],
        removeObject=True,
        removeTool=True,
    )

    gmsh.model.occ.synchronize()

    # ============================================================
    # Helper: construct a capsule around a line segment
    # ============================================================

    def make_capsule(x1, y1, x2, y2, radius):

        dx = x2 - x1
        dy = y2 - y1

        length = math.sqrt(dx * dx + dy * dy)

        if length < 1e-14:
            raise RuntimeError(
                "Attempted to create a zero-length ligament."
            )

        ex = dx / length
        ey = dy / length

        # Normal vector.
        nx = -ey
        ny = ex

        # Rectangle corners.
        p1 = gmsh.model.occ.addPoint(
            x1 + radius * nx,
            y1 + radius * ny,
            0.0,
        )

        p2 = gmsh.model.occ.addPoint(
            x2 + radius * nx,
            y2 + radius * ny,
            0.0,
        )

        p3 = gmsh.model.occ.addPoint(
            x2 - radius * nx,
            y2 - radius * ny,
            0.0,
        )

        p4 = gmsh.model.occ.addPoint(
            x1 - radius * nx,
            y1 - radius * ny,
            0.0,
        )

        l1 = gmsh.model.occ.addLine(p1, p2)
        l2 = gmsh.model.occ.addLine(p2, p3)
        l3 = gmsh.model.occ.addLine(p3, p4)
        l4 = gmsh.model.occ.addLine(p4, p1)

        loop = gmsh.model.occ.addCurveLoop(
            [l1, l2, l3, l4]
        )

        rectangle = gmsh.model.occ.addPlaneSurface(
            [loop]
        )

        # Circular end caps.
        disk1 = gmsh.model.occ.addDisk(
            x1,
            y1,
            0.0,
            radius,
            radius,
        )

        disk2 = gmsh.model.occ.addDisk(
            x2,
            y2,
            0.0,
            radius,
            radius,
        )

        capsule, _ = gmsh.model.occ.fuse(
            [(2, rectangle)],
            [(2, disk1), (2, disk2)],
            removeObject=True,
            removeTool=True,
        )

        return capsule

    # ============================================================
    # Construct the four tangent ligaments
    # ============================================================

    if mirror:
        beta_eff = -beta
    else:
        beta_eff = beta

    ligament_objects = []

    # ------------------------------------------------------------
    # The four radial directions.
    # ------------------------------------------------------------

    angles = [
        0.0,
        0.5 * math.pi,
        math.pi,
        1.5 * math.pi,
    ]

    for theta in angles:

        # --------------------------------------------------------
        # Tangency point on the central ring.
        # --------------------------------------------------------

        xt = R * math.cos(theta)
        yt = R * math.sin(theta)

        # --------------------------------------------------------
        # Tangent direction.
        #
        # beta introduces the chiral rotation.
        # --------------------------------------------------------

        tangent_angle = theta + 0.5 * math.pi + beta_eff

        ex = math.cos(tangent_angle)
        ey = math.sin(tangent_angle)

        # --------------------------------------------------------
        # Nominal ligament length.
        #
        # The latest construction uses l/2 on each side, together
        # with the beam extension.
        # --------------------------------------------------------

        half_length = 0.5 * l + beam_extension

        # Start/end points of the ligament.
        #
        # We construct the whole capsule before clipping.
        # --------------------------------------------------------

        x1 = xt - half_length * ex
        y1 = yt - half_length * ey

        x2 = xt + half_length * ex
        y2 = yt + half_length * ey

        ligament = make_capsule(
            x1,
            y1,
            x2,
            y2,
            h,
        )

        ligament_objects.extend(ligament)

    gmsh.model.occ.synchronize()

    # ============================================================
    # Global fuse of ring + all ligaments
    # ============================================================

    fiber_objects = [(2, tag) for dim, tag in ring if dim == 2]

    fiber_objects.extend(
        [(dim, tag) for dim, tag in ligament_objects if dim == 2]
    )

    if not fiber_objects:
        raise RuntimeError(
            "No fiber surfaces were generated."
        )

    fiber_union, _ = gmsh.model.occ.fuse(
        fiber_objects[:1],
        fiber_objects[1:],
        removeObject=True,
        removeTool=True,
    )

    gmsh.model.occ.synchronize()

    # ------------------------------------------------------------
    # Remove coincident CAD entities.
    #
    # This is important after the overlapping ligament construction.
    # ------------------------------------------------------------

    gmsh.model.occ.removeAllDuplicates()

    gmsh.model.occ.synchronize()

    # ============================================================
    # Clip the COMPLETE fiber to the physical unit cell
    # ============================================================

    fiber_cut, _ = gmsh.model.occ.intersect(
        fiber_union,
        [(2, square)],
        removeObject=True,
        removeTool=False,
    )

    gmsh.model.occ.synchronize()

    if not fiber_cut:
        raise RuntimeError(
            "The 66 fiber disappeared during clipping."
        )

    # ============================================================
    # Matrix = square minus fiber
    # ============================================================

    matrix, _ = gmsh.model.occ.cut(
        [(2, square)],
        fiber_cut,
        removeObject=True,
        removeTool=False,
    )

    gmsh.model.occ.synchronize()

    # ============================================================
    # Physical groups
    # ============================================================

    fiber_surfaces = [
        tag for dim, tag in fiber_cut
        if dim == 2
    ]

    matrix_surfaces = [
        tag for dim, tag in matrix
        if dim == 2
    ]

    if not fiber_surfaces:
        raise RuntimeError(
            "No fiber surfaces found after clipping."
        )

    if not matrix_surfaces:
        raise RuntimeError(
            "No matrix surfaces found."
        )

    gmsh.model.addPhysicalGroup(
        2,
        fiber_surfaces,
        1,
    )

    gmsh.model.setPhysicalName(
        2,
        1,
        "Fiber",
    )

    gmsh.model.addPhysicalGroup(
        2,
        matrix_surfaces,
        2,
    )

    gmsh.model.setPhysicalName(
        2,
        2,
        "Matrix",
    )

    # ============================================================
    # Mesh-size control
    # ============================================================

    # ------------------------------------------------------------
    # Collect fiber boundary points.
    # ------------------------------------------------------------

    fiber_points = []

    for surface_tag in fiber_surfaces:

        boundary = gmsh.model.getBoundary(
            [(2, surface_tag)],
            recursive=True,
        )

        for bdim, btag in boundary:

            if bdim == 0:
                fiber_points.append(btag)

    fiber_points = list(set(fiber_points))

    if fiber_points:

        gmsh.model.mesh.setSize(
            [(0, p) for p in fiber_points],
            mesh_size_fiber,
        )

    # Matrix resolution away from the interface.
    gmsh.option.setNumber(
        "Mesh.MeshSizeMax",
        mesh_size_matrix,
    )

    # ============================================================
    # Optional global refinement
    # ============================================================

    if num_refinements > 0:

        gmsh.option.setNumber(
            "Mesh.MeshSizeFactor",
            0.5 ** num_refinements,
        )

    # ============================================================
    # Diagnostics
    # ============================================================

    if diagnostics:

        print("\n66 geometry:")
        print(f"  R             = {R}")
        print(f"  l             = {l}")
        print(f"  t             = {t}")
        print(f"  beta          = {beta}")
        print(f"  mirror        = {mirror}")
        print(f"  a             = {a}")
        print(f"  beam extension= {beam_extension}")

        print("\nPhysical groups:")

        print(
            "  Fiber surfaces :",
            len(fiber_surfaces),
        )

        print(
            "  Matrix surfaces:",
            len(matrix_surfaces),
        )

        print("\nOuter boundary is NOT periodic.")

    # ============================================================
    # Generate mesh
    # ============================================================

    gmsh.model.mesh.generate(2)

    # ============================================================
    # Save
    # ============================================================

    gmsh.write(filename)

    gmsh.finalize()


def generate_66_mesh(
    l,
    R,
    t,
    mesh_size_matrix,
    mesh_size_fiber,
    l_micro=1.0,
    filename="output/66_mesh_cell.msh",
    mirror=False,
    num_refinements=0,
    diagnostics=False,
    mesh_size_interface=None,   # size in the matrix right next to the fibers
    transition_width=None,      # distance over which it grows to mesh_size_matrix
):
    """
    Generate a SINGLE periodic cell [0, l_micro]^2 of the chiral "66" geometry.

    Geometry (same construction as the periodic multi-cell generator):
        - a central ring of centerline radius R and thickness t,
        - four tangent ligaments whose unextended centerlines hit the
          midpoints of the cell edges,
        - ligaments are extended by t/2 beyond their nominal endpoints
          before clipping, so no artificial thin slivers appear at the
          cell boundary.

    Periodicity:
        The cell boundary is periodic (left <-> right, bottom <-> top).
        Fiber/matrix transitions on opposite edges are matched curve by
        curve, so the boundary mesh is exactly periodic.

    Physical groups:
        1 = Fiber
        2 = Matrix

    Parameters
    ----------
    filename : str
        Output .msh filename.
    l, R, t : float
        Dimensionless 66 parameters before normalization.
    mesh_size_matrix, mesh_size_fiber : float
        Mesh sizes relative to l_micro.
    l_micro : float
        Physical size of the cell (default 1.0).
    mirror : bool
        Mirror the geometry with respect to x = l_micro/2.
    num_refinements : int
        Number of uniform Gmsh refinements. Every level is written to
        <filename>_level<k>.msh; the finest mesh is also written to `filename`.
    diagnostics : bool
        Print information about the geometry and periodic curves.
    """

    # out_dir = os.path.dirname(filename)
    # if out_dir:
    #     os.makedirs(out_dir, exist_ok=True)

    gmsh.initialize()

    try:
        gmsh.model.add("Periodic_66_cell")
        occ = gmsh.model.occ

        # ==========================================================
        # Normalization: sqrt(R^2 + l^2/4) = 1/2
        # ==========================================================

        a = 2.0 * math.sqrt(l * l / 4.0 + R * R)

        l = l / a
        R = R / a
        t = t / a

        if R - t / 2.0 <= 0.0:
            raise ValueError(
                "Invalid 66 geometry: R - t/2 <= 0. "
                "The inner radius of the ring must be positive."
            )

        # Physical quantities
        L = l_micro
        l_phys = l * L
        R_phys = R * L
        t_phys = t * L
        half_t = 0.5 * t_phys

        mesh_size_matrix = mesh_size_matrix * L
        mesh_size_fiber = mesh_size_fiber * L

        beam_extension = 0.5 * t_phys

        xc = 0.5 * L
        yc = 0.5 * L

        # ==========================================================
        # Cell square
        # ==========================================================

        square = occ.addRectangle(0.0, 0.0, 0.0, L, L)

        # ==========================================================
        # Ring
        # ==========================================================

        outer_disk = occ.addDisk(
            xc, yc, 0.0, R_phys + half_t, R_phys + half_t
        )
        inner_disk = occ.addDisk(
            xc, yc, 0.0, R_phys - half_t, R_phys - half_t
        )

        ring, _ = occ.cut(
            [(2, outer_disk)],
            [(2, inner_disk)],
            removeObject=True,
            removeTool=True,
        )

        # ==========================================================
        # Four ligaments
        # ==========================================================

        theta = math.pi / 2.0 - math.atan(l / (2.0 * R))

        beams = []

        for k in range(4):

            angle = theta + k * math.pi / 2.0

            if mirror:
                angle = math.pi - angle

            # Tangency point on the ring centerline
            x_center = xc + R_phys * math.cos(angle)
            y_center = yc + R_phys * math.sin(angle)

            # Tangent direction
            tx = -math.sin(angle)
            ty = math.cos(angle)

            # Extended centerline: s in [-ext, l/2 + ext]
            s_start = -beam_extension
            s_end = 0.5 * l_phys + beam_extension

            x_start = x_center + s_start * tx
            y_start = y_center + s_start * ty
            x_end = x_center + s_end * tx
            y_end = y_center + s_end * ty

            # Normal
            nx = -ty
            ny = tx

            p0m = occ.addPoint(x_start - half_t * nx, y_start - half_t * ny, 0.0)
            p0p = occ.addPoint(x_start + half_t * nx, y_start + half_t * ny, 0.0)
            p1p = occ.addPoint(x_end + half_t * nx, y_end + half_t * ny, 0.0)
            p1m = occ.addPoint(x_end - half_t * nx, y_end - half_t * ny, 0.0)

            c0 = occ.addLine(p0m, p0p)
            c1 = occ.addLine(p0p, p1p)
            c2 = occ.addLine(p1p, p1m)
            c3 = occ.addLine(p1m, p0m)

            loop = occ.addCurveLoop([c0, c1, c2, c3])
            rectangle = occ.addPlaneSurface([loop])

            disk_start = occ.addDisk(x_start, y_start, 0.0, half_t, half_t)
            disk_end = occ.addDisk(x_end, y_end, 0.0, half_t, half_t)

            beam, _ = occ.fuse(
                [(2, rectangle)],
                [(2, disk_start), (2, disk_end)],
                removeObject=True,
                removeTool=True,
            )

            beams.extend(beam)

        # ==========================================================
        # Fuse ring + beams
        # ==========================================================

        fiber_entities = [e for e in ring if e[0] == 2]
        fiber_entities.extend([e for e in beams if e[0] == 2])

        if len(fiber_entities) == 1:
            fiber_union = fiber_entities
        else:
            fiber_union, _ = occ.fuse(
                fiber_entities[:1],
                fiber_entities[1:],
                removeObject=True,
                removeTool=True,
            )

        # NOTE: removeAllDuplicates() is intentionally not used: it can
        # delete/renumber entities and invalidate the stored tags.

        # ==========================================================
        # Clip the complete fiber to the cell
        # ==========================================================

        fiber_clipped, _ = occ.intersect(
            fiber_union,
            [(2, square)],
            removeObject=True,
            removeTool=False,
        )

        if not fiber_clipped:
            raise RuntimeError("Fiber clipping produced no surfaces.")

        # ==========================================================
        # Fragment square + fiber -> conforming fiber/matrix interface
        #   out_map[0]  : images of the square
        #   out_map[1:] : images of the clipped fiber surfaces
        # ==========================================================

        _, out_map = occ.fragment(
            [(2, square)],
            fiber_clipped,
            removeObject=True,
            removeTool=True,
        )

        occ.synchronize()

        fiber_set = set()
        for images in out_map[1:]:
            for d, tg in images:
                if d == 2:
                    fiber_set.add(tg)

        matrix_set = set()
        for d, tg in out_map[0]:
            if d == 2 and tg not in fiber_set:
                matrix_set.add(tg)

        fiber_surfaces = sorted(fiber_set)
        matrix_surfaces = sorted(matrix_set)

        if not fiber_surfaces:
            raise RuntimeError("No fiber surfaces found.")
        if not matrix_surfaces:
            raise RuntimeError("No matrix surfaces found.")

        # ==========================================================
        # Physical groups
        # ==========================================================

        gmsh.model.addPhysicalGroup(2, fiber_surfaces, 1)
        gmsh.model.setPhysicalName(2, 1, "Fiber")

        gmsh.model.addPhysicalGroup(2, matrix_surfaces, 2)
        gmsh.model.setPhysicalName(2, 2, "Matrix")

        # # ==========================================================
        # # Mesh-size control
        # # ==========================================================

        # fiber_points = set()
        # for s in fiber_surfaces:
        #     for d, tg in gmsh.model.getBoundary(
        #         [(2, s)], combined=False, oriented=False, recursive=True
        #     ):
        #         if d == 0:
        #             fiber_points.add(tg)

        # if fiber_points:
        #     gmsh.model.mesh.setSize(
        #         [(0, p) for p in sorted(fiber_points)],
        #         mesh_size_fiber,
        #     )

        # gmsh.option.setNumber("Mesh.MeshSizeMax", mesh_size_matrix)
        
        # ==========================================================
        # Mesh-size control through background fields
        #
        #   - inside the fiber  : size = mesh_size_fiber
        #   - in the matrix     : size = mesh_size_interface at the
        #                         fiber boundary, growing linearly to
        #                         mesh_size_matrix over transition_width
        #
        # All sizes are relative to L (already multiplied above).
        # ==========================================================

        h_fiber = mesh_size_fiber
        h_matrix = mesh_size_matrix
        h_int = mesh_size_interface * L if mesh_size_interface is not None else h_matrix
        width = transition_width * L if transition_width is not None else 3.0 * h_matrix

        # Curves bounding the fiber (the fiber/matrix interface plus
        # the pieces lying on the cell boundary)
        fiber_curves = set()
        for s in fiber_surfaces:
            for d, tg in gmsh.model.getBoundary(
                [(2, s)], combined=False, oriented=False, recursive=False
            ):
                if d == 1:
                    fiber_curves.add(tg)

        fld = gmsh.model.mesh.field

        # Distance to the fiber boundary
        f_dist = fld.add("Distance")
        fld.setNumbers(f_dist, "CurvesList", sorted(fiber_curves))
        fld.setNumber(f_dist, "Sampling", 200)

        # Matrix size: h_int at the interface -> h_matrix far away
        f_thr = fld.add("Threshold")
        fld.setNumber(f_thr, "InField", f_dist)
        fld.setNumber(f_thr, "SizeMin", h_int)
        fld.setNumber(f_thr, "SizeMax", h_matrix)
        fld.setNumber(f_thr, "DistMin", 0.0)
        fld.setNumber(f_thr, "DistMax", width)

        # Uniform size inside the fiber only
        f_const = fld.add("MathEval")
        fld.setString(f_const, "F", str(h_fiber))

        f_fiber = fld.add("Restrict")
        fld.setNumber(f_fiber, "InField", f_const)
        fld.setNumbers(f_fiber, "SurfacesList", fiber_surfaces)

        # Final size = min of the two
        f_min = fld.add("Min")
        fld.setNumbers(f_min, "FieldsList", [f_thr, f_fiber])
        fld.setAsBackgroundMesh(f_min)

        # Switch off every other size source, otherwise it competes
        # with the background field
        gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
        gmsh.option.setNumber("Mesh.MeshSizeMin", 0.0)
        gmsh.option.setNumber("Mesh.MeshSizeMax", max(h_fiber, h_matrix))

        # Better triangle quality: Frontal-Delaunay + optimization
        gmsh.option.setNumber("Mesh.Algorithm", 6)
        gmsh.option.setNumber("Mesh.Optimize", 1)
        gmsh.option.setNumber("Mesh.Smoothing", 5)


        # ==========================================================
        # Classify boundary curves of the cell
        # ==========================================================

        tol = 1e-6 * max(1.0, L)

        left, right, bottom, top = [], [], [], []

        for d, tg in gmsh.model.getEntities(1):

            xmin, ymin, _, xmax, ymax, _ = gmsh.model.getBoundingBox(d, tg)

            if abs(xmin) < tol and abs(xmax) < tol:
                left.append(tg)
            elif abs(xmin - L) < tol and abs(xmax - L) < tol:
                right.append(tg)
            elif abs(ymin) < tol and abs(ymax) < tol:
                bottom.append(tg)
            elif abs(ymin - L) < tol and abs(ymax - L) < tol:
                top.append(tg)

        def curve_center(tg):
            xmin, ymin, _, xmax, ymax, _ = gmsh.model.getBoundingBox(1, tg)
            return 0.5 * (xmin + xmax), 0.5 * (ymin + ymax)

        def pair_curves(slaves, masters, shift):
            """
            Pair every slave curve with the master curve whose center
            is the slave center minus `shift`. Returns two ordered lists.
            """
            sx, sy = shift
            remaining = list(masters)
            slave_out, master_out = [], []

            for s in slaves:
                cx, cy = curve_center(s)
                found = None
                for m in remaining:
                    mx, my = curve_center(m)
                    if abs(cx - sx - mx) < 1e-6 * L and abs(cy - sy - my) < 1e-6 * L:
                        found = m
                        break
                if found is None:
                    raise RuntimeError(
                        f"No periodic partner found for curve {s}. "
                        "Boundary curves on opposite sides do not match."
                    )
                remaining.remove(found)
                slave_out.append(s)
                master_out.append(found)

            if remaining:
                raise RuntimeError(
                    f"Unmatched master curves: {remaining}"
                )

            return slave_out, master_out

        if not (left and right and bottom and top):
            raise RuntimeError("Could not find all four boundary sides.")

        if len(left) != len(right) or len(bottom) != len(top):
            raise RuntimeError(
                "Opposite sides have different numbers of curves: "
                f"L/R = {len(left)}/{len(right)}, "
                f"B/T = {len(bottom)}/{len(top)}."
            )

        right_s, left_m = pair_curves(right, left, (L, 0.0))
        top_s, bottom_m = pair_curves(top, bottom, (0.0, L))

        if diagnostics:
            print("\nPeriodic curves (slave <- master):")
            print("  x:", list(zip(right_s, left_m)))
            print("  y:", list(zip(top_s, bottom_m)))

        # ==========================================================
        # Periodicity (the transform maps master -> slave)
        # ==========================================================

        translation_x = [
            1, 0, 0, L,
            0, 1, 0, 0,
            0, 0, 1, 0,
            0, 0, 0, 1,
        ]

        translation_y = [
            1, 0, 0, 0,
            0, 1, 0, L,
            0, 0, 1, 0,
            0, 0, 0, 1,
        ]

        gmsh.model.mesh.setPeriodic(1, right_s, left_m, translation_x)
        gmsh.model.mesh.setPeriodic(1, top_s, bottom_m, translation_y)

        # ==========================================================
        # Mesh generation and refinements
        # ==========================================================

        gmsh.model.mesh.generate(2)

        for level in range(num_refinements + 1):

            level_filename = filename.replace(".msh", f"_level{level}.msh")

            if diagnostics:
                print(f"Writing mesh level {level}: {level_filename}")

            gmsh.write(level_filename)

            if level < num_refinements:
                gmsh.model.mesh.refine()

        gmsh.write(filename)

        if diagnostics:
            print("\n66 single-cell periodic mesh completed.")
            print(f"  L = {L}")
            print(f"  normalized l, R, t = {l}, {R}, {t}")
            print(f"  fiber surfaces  : {len(fiber_surfaces)}")
            print(f"  matrix surfaces : {len(matrix_surfaces)}")

    finally:
        gmsh.finalize()
###################################################
# AFTER MESH GENERATION:
    # print("\nGmsh periodicity check")
    # print("---------------------------")

    # for slave_curve in right_curves + top_curves:

    #     master_curve, node_tags, node_tags_master, affine = \
    #         gmsh.model.mesh.getPeriodicNodes(
    #             1,
    #             slave_curve,
    #             False,
    #         )

    #     print(
    #         f"  slave curve {slave_curve} "
    #         f"-> master curve {master_curve}: "
    #         f"{len(node_tags)} node pairs"
    #     )

    #     if len(node_tags) == 0:
    #         print("  [FAIL] No periodic node correspondence.")
    #     else:
    #         print("  [OK]")
#AAAAAAAAAAAAAAAAAAAAAAAA

###################################################   
    # # ------------------------------------------------------------
    # # Smoothen the corners
    # # ------------------------------------------------------------

    # d = math.sqrt(2.0) * R
    # q = R / math.sqrt(2.0)

    # corner_offset = math.sqrt(2.0) * h

    # sharp_corners = [
    #     # left
    #     (0.5 - corner_offset, 0.5),

    #     # right
    #     (0.5 + corner_offset, 0.5),

    #     # bottom
    #     (0.5, 0.5 - corner_offset),

    #     # top
    #     (0.5, 0.5 + corner_offset),
    # ]

    # rounding_regions = []

    # # ============================================================
    # # LEFT CORNER
    # # ============================================================

    # xc, yc = sharp_corners[0]

    # # Circle center O: to the left of V
    # xo = xc - d
    # yo = yc

    # # Tangency points
    # xa = xc - q
    # ya = yc - q

    # xb = xc - q
    # yb = yc + q

    # V = gmsh.model.occ.addPoint(xc, yc, 0.0)
    # A = gmsh.model.occ.addPoint(xa, ya, 0.0)
    # B = gmsh.model.occ.addPoint(xb, yb, 0.0)
    # O = gmsh.model.occ.addPoint(xo, yo, 0.0)

    # VA = gmsh.model.occ.addLine(V, A)

    # AB = gmsh.model.occ.addCircleArc(
    #     A,
    #     O,
    #     B,
    # )

    # BV = gmsh.model.occ.addLine(B, V)

    # loop = gmsh.model.occ.addCurveLoop([
    #     VA,
    #     AB,
    #     BV,
    # ])

    # region = gmsh.model.occ.addPlaneSurface([loop])

    # rounding_regions.append((2, region))

    # # ============================================================
    # # RIGHT CORNER
    # # ============================================================

    # xc, yc = sharp_corners[1]

    # # Circle center O: to the right of V
    # xo = xc + d
    # yo = yc

    # # Tangency points
    # xa = xc + q
    # ya = yc - q

    # xb = xc + q
    # yb = yc + q

    # V = gmsh.model.occ.addPoint(xc, yc, 0.0)
    # A = gmsh.model.occ.addPoint(xa, ya, 0.0)
    # B = gmsh.model.occ.addPoint(xb, yb, 0.0)
    # O = gmsh.model.occ.addPoint(xo, yo, 0.0)

    # VA = gmsh.model.occ.addLine(V, A)

    # AB = gmsh.model.occ.addCircleArc(
    #     A,
    #     O,
    #     B,
    # )

    # BV = gmsh.model.occ.addLine(B, V)

    # loop = gmsh.model.occ.addCurveLoop([
    #     VA,
    #     AB,
    #     BV,
    # ])

    # region = gmsh.model.occ.addPlaneSurface([loop])

    # rounding_regions.append((2, region))

    # # ============================================================
    # # BOTTOM CORNER
    # # ============================================================

    # xc, yc = sharp_corners[2]

    # # Circle center O: below V
    # xo = xc
    # yo = yc - d

    # # Tangency points
    # xa = xc - q
    # ya = yc - q

    # xb = xc + q
    # yb = yc - q

    # V = gmsh.model.occ.addPoint(xc, yc, 0.0)
    # A = gmsh.model.occ.addPoint(xa, ya, 0.0)
    # B = gmsh.model.occ.addPoint(xb, yb, 0.0)
    # O = gmsh.model.occ.addPoint(xo, yo, 0.0)

    # VA = gmsh.model.occ.addLine(V, A)

    # AB = gmsh.model.occ.addCircleArc(
    #     A,
    #     O,
    #     B,
    # )

    # BV = gmsh.model.occ.addLine(B, V)

    # loop = gmsh.model.occ.addCurveLoop([
    #     VA,
    #     AB,
    #     BV,
    # ])

    # region = gmsh.model.occ.addPlaneSurface([loop])

    # rounding_regions.append((2, region))

    # # ============================================================
    # # TOP CORNER
    # # ============================================================

    # xc, yc = sharp_corners[3]

    # # Circle center O: above V
    # xo = xc
    # yo = yc + d

    # # Tangency points
    # xa = xc - q
    # ya = yc + q

    # xb = xc + q
    # yb = yc + q

    # V = gmsh.model.occ.addPoint(xc, yc, 0.0)
    # A = gmsh.model.occ.addPoint(xa, ya, 0.0)
    # B = gmsh.model.occ.addPoint(xb, yb, 0.0)
    # O = gmsh.model.occ.addPoint(xo, yo, 0.0)

    # VA = gmsh.model.occ.addLine(V, A)

    # AB = gmsh.model.occ.addCircleArc(
    #     A,
    #     O,
    #     B,
    # )

    # BV = gmsh.model.occ.addLine(B, V)

    # loop = gmsh.model.occ.addCurveLoop([
    #     VA,
    #     AB,
    #     BV,
    # ])

    # region = gmsh.model.occ.addPlaneSurface([loop])

    # rounding_regions.append((2, region))

    # gmsh.model.occ.synchronize()

    # # ------------------------------------------------------------
    # # ADD the four V-A-B regions to the fibers
    # #
    # # IMPORTANT:
    # # The disks are NOT present here.
    # # Only the four curvilinear V-A-B surfaces are added.
    # # ------------------------------------------------------------

    # fiber_union, _ = gmsh.model.occ.fuse(
    #     fiber_union,
    #     rounding_regions,
    #     removeObject=True,
    #     removeTool=True,
    # )

    # gmsh.model.occ.synchronize()




    # FINESCALE:
    
        # d = math.sqrt(2.0) * R
        # q = R / math.sqrt(2.0)

        # corner_offset = math.sqrt(2.0) * h

        # xc = x0 + 0.5 * l_micro
        # yc = y0 + 0.5 * l_micro

        # sharp_corners = [
        #     # left
        #     (xc - corner_offset, yc),

        #     # right
        #     (xc + corner_offset, yc),

        #     # bottom
        #     (xc, yc - corner_offset),

        #     # top
        #     (xc, yc + corner_offset),
        # ]

        # rounding_regions = []

        # # ============================================================
        # # LEFT CORNER
        # # ============================================================

        # xc, yc = sharp_corners[0]

        # # Circle center O: to the left of V
        # xo = xc - d
        # yo = yc

        # # Tangency points
        # xa = xc - q
        # ya = yc - q

        # xb = xc - q
        # yb = yc + q

        # V = gmsh.model.occ.addPoint(xc, yc, 0.0)
        # A = gmsh.model.occ.addPoint(xa, ya, 0.0)
        # B = gmsh.model.occ.addPoint(xb, yb, 0.0)
        # O = gmsh.model.occ.addPoint(xo, yo, 0.0)

        # VA = gmsh.model.occ.addLine(V, A)

        # AB = gmsh.model.occ.addCircleArc(
        #     A,
        #     O,
        #     B,
        # )

        # BV = gmsh.model.occ.addLine(B, V)

        # loop = gmsh.model.occ.addCurveLoop([
        #     VA,
        #     AB,
        #     BV,
        # ])

        # region = gmsh.model.occ.addPlaneSurface([loop])

        # rounding_regions.append((2, region))

        # # ============================================================
        # # RIGHT CORNER
        # # ============================================================

        # xc, yc = sharp_corners[1]

        # # Circle center O: to the right of V
        # xo = xc + d
        # yo = yc

        # # Tangency points
        # xa = xc + q
        # ya = yc - q

        # xb = xc + q
        # yb = yc + q

        # V = gmsh.model.occ.addPoint(xc, yc, 0.0)
        # A = gmsh.model.occ.addPoint(xa, ya, 0.0)
        # B = gmsh.model.occ.addPoint(xb, yb, 0.0)
        # O = gmsh.model.occ.addPoint(xo, yo, 0.0)

        # VA = gmsh.model.occ.addLine(V, A)

        # AB = gmsh.model.occ.addCircleArc(
        #     A,
        #     O,
        #     B,
        # )

        # BV = gmsh.model.occ.addLine(B, V)

        # loop = gmsh.model.occ.addCurveLoop([
        #     VA,
        #     AB,
        #     BV,
        # ])

        # region = gmsh.model.occ.addPlaneSurface([loop])

        # rounding_regions.append((2, region))

        # # ============================================================
        # # BOTTOM CORNER
        # # ============================================================

        # xc, yc = sharp_corners[2]

        # # Circle center O: below V
        # xo = xc
        # yo = yc - d

        # # Tangency points
        # xa = xc - q
        # ya = yc - q

        # xb = xc + q
        # yb = yc - q

        # V = gmsh.model.occ.addPoint(xc, yc, 0.0)
        # A = gmsh.model.occ.addPoint(xa, ya, 0.0)
        # B = gmsh.model.occ.addPoint(xb, yb, 0.0)
        # O = gmsh.model.occ.addPoint(xo, yo, 0.0)

        # VA = gmsh.model.occ.addLine(V, A)

        # AB = gmsh.model.occ.addCircleArc(
        #     A,
        #     O,
        #     B,
        # )

        # BV = gmsh.model.occ.addLine(B, V)

        # loop = gmsh.model.occ.addCurveLoop([
        #     VA,
        #     AB,
        #     BV,
        # ])

        # region = gmsh.model.occ.addPlaneSurface([loop])

        # rounding_regions.append((2, region))

        # # ============================================================
        # # TOP CORNER
        # # ============================================================

        # xc, yc = sharp_corners[3]

        # # Circle center O: above V
        # xo = xc
        # yo = yc + d

        # # Tangency points
        # xa = xc - q
        # ya = yc + q

        # xb = xc + q
        # yb = yc + q

        # V = gmsh.model.occ.addPoint(xc, yc, 0.0)
        # A = gmsh.model.occ.addPoint(xa, ya, 0.0)
        # B = gmsh.model.occ.addPoint(xb, yb, 0.0)
        # O = gmsh.model.occ.addPoint(xo, yo, 0.0)

        # VA = gmsh.model.occ.addLine(V, A)

        # AB = gmsh.model.occ.addCircleArc(
        #     A,
        #     O,
        #     B,
        # )

        # BV = gmsh.model.occ.addLine(B, V)

        # loop = gmsh.model.occ.addCurveLoop([
        #     VA,
        #     AB,
        #     BV,
        # ])

        # region = gmsh.model.occ.addPlaneSurface([loop])

        # rounding_regions.append((2, region))

        # gmsh.model.occ.synchronize()

        # # ------------------------------------------------------------
        # # ADD the four V-A-B regions to the fibers
        # #
        # # IMPORTANT:
        # # The disks are NOT present here.
        # # Only the four curvilinear V-A-B surfaces are added.
        # # ------------------------------------------------------------

        # fiber_union, _ = gmsh.model.occ.fuse(
        #     fiber_union,
        #     rounding_regions,
        #     removeObject=True,
        #     removeTool=True,
        # )

        # gmsh.model.occ.synchronize()

        # return fiber_union