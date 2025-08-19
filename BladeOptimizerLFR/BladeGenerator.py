import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt
import trimesh


# ---------- shape primitives (gamma/tau) ----------
def beta_distribution(x, alpha):
    """
    Basic normalized camber envelope γ(x).
    Formula:
        γ(x) = [x^α * (1-x)^(1-α)] / [α^α * (1-α)^(1-α)]
    Peak occurs at x=α, max(γ)=1.
    """
    x = np.asarray(x)
    eps = 1e-9
    a = np.clip(alpha, eps, 1.0 - eps)
    num = (x**a) * ((1.0 - x)**(1.0 - a))
    den = (a**a) * ((1.0 - a)**(1.0 - a)) + eps
    return num / den


def beta_distribution_extended(x, alpha, kappa=1.0):
    """
    Extended β-based camber envelope γ_ext(x).
    Formula:
        γ_ext(x) = [x^(κ·α) * (1-x)^(κ·(1-α))] /
                   [(α^α * (1-α)^(1-α))^κ]
    - kappa > 1 sharpens the peak
    - kappa < 1 flattens the curve
    """
    x = np.asarray(x)
    eps = 1e-9
    a = np.clip(alpha, eps, 1.0 - eps)
    num = (x ** (kappa * a)) * ((1.0 - x) ** (kappa * (1.0 - a)))
    den = ((a**a) * ((1.0 - a)**(1.0 - a)) + eps) ** kappa
    return num / den


def thickness_distribution(x, a=0.2, b=0.8, beta=0.3):
    """
    Basic thickness distribution τ(x).
    Piecewise:
      - [0,a]: cosine ramp up
      - (a,b): flat
      - [b,1]: cosine decay
    """
    x = np.asarray(x)
    a = np.clip(a, 0.1, 0.3)
    b = np.clip(b, 0.7, 0.9)
    if not (a <= b):
        a, b = min(a, b), max(a, b)

    tau = np.ones_like(x)
    idx_le = x <= a
    if a > 0:
        tau[idx_le] = ((1.0 - np.cos(np.pi * x[idx_le] / a)) / 2.0) ** beta
    else:
        tau[idx_le] = 0.0
    idx_te = x >= b
    if b < 1.0:
        tau[idx_te] = ((1.0 + np.cos(np.pi * (x[idx_te] - b) / (1.0 - b))) / 2.0) ** beta
    else:
        tau[idx_te] = 0.0
    return tau


def tapered_thickness_distribution(x, a=0.2, b=0.8, beta=0.3, taper=0.5):
    """
    Tapered thickness τ_tape(x).
    Formula:
        τ_tape(x) = τ(x) * (1 - taper·x)
    - taper ∈ [0,1]; higher values reduce thickness at the trailing tip
    - prevents excessive curling at the blade tip
    """
    base = thickness_distribution(x, a, b, beta)
    return base * (1.0 - taper * x)


# ---------- blade generator ----------
class Blade3D:
    """
    3D blade generation from spanwise layered definitions.
    """

    def __init__(
        self,
        span_layers, Theta=np.pi / 6, H=1.0,
        z0=0.0, hub_radius=1.0, shroud_radius=2.0,
    ):
        self.layers = span_layers
        self.Theta = float(Theta)
        self.H = float(H)
        self.z0 = float(z0)
        self.hub_radius = float(hub_radius)
        self.shroud_radius = float(shroud_radius)

        self.vertices_upper = None
        self.vertices_lower = None
        self.vertices_center = None
        self.faces_upper = None
        self.faces_lower = None
        self.faces_center = None

    def _layer_radius(self, i, n):
        li = self.layers[i]
        if 'radius' in li and li['radius'] is not None:
            return float(li['radius'])
        w = 0.0 if n <= 1 else i / (n - 1)
        return (1.0 - w) * self.hub_radius + w * self.shroud_radius

    def generate_surface(self, points_per_chord=300):
        num_layers = len(self.layers)
        N = int(points_per_chord)
        xi = np.linspace(0.0, 1.0, N)

        verts_upper = []
        verts_lower = []
        verts_center = []

        for i in range(num_layers):
            prm = self.layers[i]
            theta0 = float(prm['theta0'])
            h_max = float(prm['h_max'])
            t_max = float(prm['t_max'])
            alpha = float(prm['alpha'])
            a = float(prm['a'])
            b = float(prm['b'])
            beta = float(prm['beta'])
            mode = prm.get('mode', 'basic')

            # choose gamma
            if mode == 'extended':
                kappa = float(prm.get('kappa', 1.5))
                gamma = beta_distribution_extended(xi, alpha, kappa)
            else:
                gamma = beta_distribution(xi, alpha)

            # choose tau
            if mode == 'tapered':
                taper = float(prm.get('taper', 0.5))
                tau = tapered_thickness_distribution(xi, a, b, beta, taper)
            else:
                tau = thickness_distribution(xi, a, b, beta)

            theta = theta0 + xi * self.Theta
            z_center = self.z0 + xi * self.H - h_max * gamma
            z_upper = z_center + t_max * tau
            z_lower = z_center - t_max * tau

            R = self._layer_radius(i, num_layers)
            x_circ = np.cos(theta) * R
            y_circ = np.sin(theta) * R

            verts_upper.append(np.stack([x_circ, y_circ, z_upper], axis=-1))
            verts_lower.append(np.stack([x_circ, y_circ, z_lower], axis=-1))
            verts_center.append(np.stack([x_circ, y_circ, z_center], axis=-1))

        self.vertices_upper = np.vstack(verts_upper)
        self.vertices_lower = np.vstack(verts_lower)
        self.vertices_center = np.vstack(verts_center)

        def build_faces(verts, N, num_layers):
            faces = []
            for i in range(num_layers - 1):
                base0 = i * N
                base1 = (i + 1) * N
                for j in range(N - 1):
                    v0 = base0 + j
                    v1 = base0 + j + 1
                    v2 = base1 + j + 1
                    v3 = base1 + j
                    faces.append([4, v0, v1, v2, v3])
            return np.hstack(faces).astype(np.int64)

        self.faces_upper = build_faces(self.vertices_upper, N, num_layers)
        self.faces_lower = build_faces(self.vertices_lower, N, num_layers)
        self.faces_center = build_faces(self.vertices_center, N, num_layers)

    def to_pyvista_mesh(self, mode="both"):
        if self.vertices_upper is None:
            self.generate_surface()
        meshes = []
        if mode in ["upper", "both"]:
            meshes.append(pv.PolyData(self.vertices_upper, self.faces_upper))
        if mode in ["lower", "both"]:
            meshes.append(pv.PolyData(self.vertices_lower, self.faces_lower))
        if mode == "center":
            meshes.append(pv.PolyData(self.vertices_center, self.faces_center))
        if len(meshes) == 1:
            return meshes[0]
        return meshes[0].merge(meshes[1:])

    def visualize(self, mode="both"):
        mesh = self.to_pyvista_mesh(mode)
        plotter = pv.Plotter()
        plotter.add_mesh(mesh, color='lightblue', show_edges=True)
        plotter.show()

    def _generate_solid_from_surfaces(self, mode="both"):
        """
        Generate a solid mesh by connecting upper and lower surfaces.
        - mode: "upper", "lower", "both" (used for surface selection)
        Returns:
            trimesh.Trimesh object
        """
        # Ensure surfaces are generated
        if self.vertices_upper is None:
            self.generate_surface()

        # Number of layers and points per chord
        num_layers = len(self.layers)
        N = len(self.vertices_upper) // num_layers

        # Build vertices
        vertices = []
        if mode in ["upper", "both"]:
            vertices.extend(self.vertices_upper.tolist())
        if mode in ["lower", "both"]:
            vertices.extend(self.vertices_lower.tolist())

        vertices = np.array(vertices)

        # Build faces
        faces = []

        # Upper surface faces
        if mode in ["upper", "both"]:
            for i in range(num_layers - 1):
                base0 = i * N
                base1 = (i + 1) * N
                for j in range(N - 1):
                    v0 = base0 + j
                    v1 = base0 + j + 1
                    v2 = base1 + j + 1
                    v3 = base1 + j
                    faces.append([v0, v1, v2])
                    faces.append([v0, v2, v3])

        # Lower surface faces (offset by upper vertices)
        offset = len(self.vertices_upper) if mode in ["both"] else 0
        if mode in ["lower", "both"]:
            for i in range(num_layers - 1):
                base0 = i * N + offset
                base1 = (i + 1) * N + offset
                for j in range(N - 1):
                    v0 = base0 + j
                    v1 = base0 + j + 1
                    v2 = base1 + j + 1
                    v3 = base1 + j
                    faces.append([v0, v2, v1])  # reverse order to flip normals
                    faces.append([v0, v3, v2])

        # Side faces connecting upper and lower surfaces
        if mode == "both":
            for i in range(num_layers):
                base_upper = i * N
                base_lower = base_upper + offset
                for j in range(N - 1):
                    v0 = base_upper + j
                    v1 = base_upper + j + 1
                    v2 = base_lower + j + 1
                    v3 = base_lower + j
                    faces.append([v0, v1, v2])
                    faces.append([v0, v2, v3])

        # Create trimesh
        solid_mesh = trimesh.Trimesh(vertices=vertices, faces=np.array(faces))
        solid_mesh.remove_duplicate_faces()
        solid_mesh.remove_unreferenced_vertices()
        solid_mesh.fill_holes()
        return solid_mesh

    def export_mesh(self, filename: str, mode: str = "both", as_solid: bool = False):
        if as_solid:
            mesh = self._generate_solid_from_surfaces(mode)
            mesh.export(filename)
            return
        mesh = self.to_pyvista_mesh(mode)
        mesh.save(filename)


# ------------- example -------------
if __name__ == "__main__":
    layers_params = [
        {'theta0': 0.0, 'h_max': 0.02, 't_max': 0.01, 'alpha': 0.4, 'a': 0.2, 'b': 0.8, 'beta': 0.3,
         'mode': 'extended', 'kappa': 1.5},
        {'theta0': 0.005, 'h_max': 0.022, 't_max': 0.011, 'alpha': 0.41, 'a': 0.2, 'b': 0.8, 'beta': 0.3,
         'mode': 'extended', 'kappa': 1.55},
        {'theta0': 0.01, 'h_max': 0.024, 't_max': 0.012, 'alpha': 0.42, 'a': 0.2, 'b': 0.8, 'beta': 0.3,
         'mode': 'extended', 'kappa': 1.6},
        {'theta0': 0.015, 'h_max': 0.026, 't_max': 0.013, 'alpha': 0.43, 'a': 0.2, 'b': 0.8, 'beta': 0.3,
         'mode': 'extended', 'kappa': 1.65},
        {'theta0': 0.02, 'h_max': 0.028, 't_max': 0.014, 'alpha': 0.44, 'a': 0.2, 'b': 0.8, 'beta': 0.3,
         'mode': 'extended', 'kappa': 1.7},
    ]

    blade = Blade3D(
        span_layers=layers_params,
        Theta=np.pi / 6,
        H=0.21,
        z0=-0.1,
        hub_radius=0.121,
        shroud_radius=0.16,
    )
    blade.generate_surface(points_per_chord=300)

    blade.visualize(mode="both")
    blade.export_mesh("blade_example_solid.stl", mode="both", as_solid=True)
    blade.export_mesh("blade_example_surface.stl", mode="both", as_solid=False)

    # -------- validation script: plot one cross-section --------
    test_layer_index = 1
    test_layer = layers_params[test_layer_index]
    N = 200
    xi = np.linspace(0.0, 1.0, N)

    mode = test_layer.get('mode', 'basic')
    if mode == 'extended':
        gamma = beta_distribution_extended(xi, test_layer['alpha'], test_layer.get('kappa', 1.5))
    else:
        gamma = beta_distribution(xi, test_layer['alpha'])

    if mode == 'tapered':
        tau = tapered_thickness_distribution(xi, test_layer['a'], test_layer['b'], test_layer['beta'], test_layer.get('taper', 0.5))
    else:
        tau = thickness_distribution(xi, test_layer['a'], test_layer['b'], test_layer['beta'])

    theta0 = float(test_layer['theta0'])
    h_max = float(test_layer['h_max'])
    t_max = float(test_layer['t_max'])
    theta = theta0 + xi * blade.Theta

    z_center = blade.z0 + xi * blade.H - h_max * gamma
    z_upper = z_center + t_max * tau
    z_lower = z_center - t_max * tau

    plt.figure(figsize=(6, 4))
    plt.plot(xi, z_center, 'k-', label="camber line")
    plt.plot(xi, z_upper, 'r-', label="upper surface")
    plt.plot(xi, z_lower, 'b-', label="lower surface")
    plt.xlabel("Relative chord position ξ")
    plt.ylabel("z (relative)")
    plt.title(f"Blade cross-section (span={test_layer_index/(len(layers_params)-1)} ({mode} mode)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
