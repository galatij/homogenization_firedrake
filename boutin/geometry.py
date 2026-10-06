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
    err = 0.02
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

    a = 2*np.sqrt(l*l/4 + R*R*2)
    l = l/a
    R = R/a
    t = t/a
    a = a/a

    xc = a/2
    yc = a/2

    if data_geometry["chiral"]["mirror"]:
        # Reflect x across the local cell center axis (xc = 0.5)
        x = 2 * xc - x  

    theta = pi/2 - np.arctan(l/(2*R)) - err #if mirror == False else pi/2 + np.arctan(l/(2*R)) + err


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

def generate_XX_mesh(
    t,
    mesh_size_matrix,
    mesh_size_fiber,
    filename="output/X_mesh_cell.msh",
    num_refinements = 0
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

    corners = [
        (0.0, -h),
        (sqrt2, -h),
        (sqrt2, +h),
        (0.0, +h),
    ]

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

    square = gmsh.model.occ.addRectangle(
        0.0,
        0.0,
        0.0,
        L,
        L,
    )

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

    fiber_surfaces = [
        tag for dim, tag in fiber_union
        if dim == 2
    ]

    matrix_surfaces = [
        tag for dim, tag in matrix
        if dim == 2
    ]

    if fiber_surfaces:
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

    if matrix_surfaces:
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

    # ------------------------------------------------------------
    # Mesh-size control
    # ------------------------------------------------------------

    fiber_points = []

    for dim, tag in fiber_union:

        boundary = gmsh.model.getBoundary(
            [(dim, tag)],
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

    gmsh.option.setNumber(
        "Mesh.MeshSizeMax",
        mesh_size_matrix,
    )

    # ============================================================
    # Identify periodic boundary curves
    # ============================================================

    tol = 1e-6

    # Get all curves bounding the final matrix surfaces.
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


    # Classify curves from their endpoint coordinates.
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

            xyz = gmsh.model.getValue(
                0,
                point_tag,
                [],
            )

            coordinates.append(
                (xyz[0], xyz[1])
            )

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

        coords = [
            gmsh.model.getValue(0, p, [])
            for p in point_tags
        ]

        x = 0.5 * (coords[0][0] + coords[1][0])
        y = 0.5 * (coords[0][1] + coords[1][1])

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
    gmsh.model.mesh.setPeriodic(
        1,
        right_curves,
        left_curves,
        translation_x,
    )

    # Top -> Bottom
    gmsh.model.mesh.setPeriodic(
        1,
        top_curves,
        bottom_curves,
        translation_y,
    )

    print("\n[OK] Periodic boundary conditions defined.")

    # ============================================================
    # Generate mesh
    # ============================================================
    # gmsh.option.setNumber("Mesh.MshFileVersion", 4.1)
    gmsh.model.mesh.generate(2)

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

        corners = [
            (0.0, -h),
            (smax, -h),
            (smax, +h),
            (0.0, +h),
        ]

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
                gmsh.model.occ.addLine(
                    fiber1_pts[i],
                    fiber1_pts[(i + 1) % 4],
                )
            )

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
                gmsh.model.occ.addLine(
                    fiber2_pts[i],
                    fiber2_pts[(i + 1) % 4],
                )
            )

        loop2 = gmsh.model.occ.addCurveLoop(lines2)
        surf2 = gmsh.model.occ.addPlaneSurface([loop2])

        # ----------------------------------------------------
        # Local cell
        # ----------------------------------------------------

        square = gmsh.model.occ.addRectangle(
            x0,
            y0,
            0,
            l_micro,
            l_micro,
        )

        gmsh.model.occ.synchronize()

        # ----------------------------------------------------
        # Clip first diagonal to cell
        # ----------------------------------------------------

        fiber1_cut, _ = gmsh.model.occ.intersect(
            [(2, surf1)],
            [(2, square)],
            removeObject=True,
            removeTool=False,
        )

        gmsh.model.occ.synchronize()

        # ----------------------------------------------------
        # Clip second diagonal to cell
        # ----------------------------------------------------

        fiber2_cut, _ = gmsh.model.occ.intersect(
            [(2, surf2)],
            [(2, square)],
            removeObject=True,
            removeTool=False,
        )

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

    domain = gmsh.model.occ.addRectangle(
        0,
        0,
        0,
        Lx,
        Ly,
    )

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

    # ========================================================
    # Mesh-size control
    # ========================================================

    fiber_points = []

    for dim, tag in fiber_union:

        boundary = gmsh.model.getBoundary(
            [(dim, tag)],
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

    gmsh.option.setNumber(
        "Mesh.MeshSizeMax",
        mesh_size_matrix,
    )
    # ============================================================
    # Identify periodic boundary curves of the fine-scale domain
    # ============================================================

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
            tag for dim, tag in endpoints
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

        coords = [
            gmsh.model.getValue(0, p, [])
            for p in point_tags
        ]

        x = 0.5 * (coords[0][0] + coords[1][0])
        y = 0.5 * (coords[0][1] + coords[1][1])

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