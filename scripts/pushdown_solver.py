"""Second-order plastic-hinge pushdown solver for the certificate cross-check.

Formulation
-----------
Plane frame.  Every frame node has three degrees of freedom
``[u_x, u_y, theta]``; fixed bases are fully restrained.  The unknown vector also
carries the **element-internal end rotations** of every member, two per member, so
that a member end rotation and the joint rotation it frames are distinct
quantities connected by an explicit rotational hinge element,

    M = k_h (theta_joint - theta_member - theta_p),        |M| <= M_y ,

with the plastic rotation ``theta_p`` produced by a return mapping.  This is the
standard concentrated-plasticity frame idealization: the member is elastic, the
plasticity is a zero-length rotational spring at its ends, and the joint rotation
is shared by every member that frames it.

Geometric stiffness
-------------------
The consistent P-Delta geometric stiffness of a beam-column under axial force
``N`` is added to the material stiffness,

    k_g = N / (30 L) * [[0,0,0,0,0,0],
                        [0,36,3L,0,-36,3L],
                        [0,3L,4L^2,0,-3L,-L^2],
                        [0,0,0,0,0,0],
                        [0,-36,-3L,0,36,-3L],
                        [0,3L,-L^2,0,-3L,4L^2]] .

``second_order=False`` reproduces the small-displacement idealization of the
certificate; ``second_order=True`` adds the P-Delta effect.

Solution scheme
---------------
The pushdown multiplier ``lambda`` is an unknown and the vertical displacement of
the node above the cut is prescribed in equal steps.  Each step solves the
bordered equilibrium system

    [ K_ff   -p_f ] [ du_f ]   [ r_f ]
    [ K_pf   -p_p ] [ dlam ] = [ r_p ]

with an initial-stiffness operator and a return mapping on every hinge.  A step
that cannot be equilibrated is the discrete limit point; the last converged
``lambda`` is the pushdown multiplier, which for the elastic-perfectly-plastic
idealization equals the rigid-plastic collapse multiplier of the same frame.
"""
from __future__ import annotations

import math

import numpy as np

from engine import Frame  # noqa: E402

HINGE_REFERENCE_STIFFNESS = 5.0e3
NUM_STEPS = 240
NEWTON_ABS_TOL = 1.0e-8
MAX_ITER = 60
# bilinear softening backbone of the degrading lane
SOFTENING_ONSET = 6.0
SOFTENING_SLOPE = 0.05
SOFTENING_RESIDUAL = 0.2


def frame_topology(f: Frame, removal: tuple[int, int]) -> dict:
    """Return node coordinates, member table and the cut-node index."""
    s, g = removal
    yy = np.concatenate([[0.0], np.cumsum(list(f.heights))])
    xx = np.concatenate([[0.0], np.cumsum(list(f.spans))])

    survivor_joints: set[tuple[int, int]] = set()
    removed_joints: set[tuple[int, int]] = set()
    for m in f.members:
        kind, story, grid = m[0], int(m[1]), int(m[2])
        touched = ({(grid, story), (grid + 1, story)} if kind == "b"
                   else {(grid, story - 1), (grid, story)})
        if kind == "c" and story == s and grid == g:
            removed_joints |= touched
        else:
            survivor_joints |= touched
    removed_joints -= survivor_joints
    keep = survivor_joints | (removed_joints & {(j, 0) for j in range(f.B + 1)})

    node_of: dict[tuple[int, int], int] = {}
    coords: list[tuple[float, float]] = []
    for r in range(f.H + 1):
        for j in range(f.B + 1):
            if (j, r) not in keep:
                continue
            node_of[(j, r)] = len(coords)
            coords.append((float(xx[j]), float(yy[r])))

    members = []
    for k, m in enumerate(f.members):
        kind, story, grid = m[0], int(m[1]), int(m[2])
        if kind == "c" and story == s and grid == g:
            continue
        if kind == "b":
            ni, nj = node_of[(grid, story)], node_of[(grid + 1, story)]
        else:
            ni, nj = node_of[(grid, story - 1)], node_of[(grid, story)]
        members.append({"index": k, "kind": kind, "nodes": (int(ni), int(nj)),
                        "My": (float(m[3]), float(m[4]))})
    fixed = [node_of[(j, 0)] for j in range(f.B + 1) if (j, 0) in node_of]
    return {
        "coords": coords,
        "members": members,
        "fixed": fixed,
        "cut_node": int(node_of[(g, s)]),
        "node_of": {f"{j}_{r}": int(v) for (j, r), v in node_of.items()},
    }


def elastic_properties(member_index: int, i_scale: float) -> tuple[float, float, float]:
    e_mod = 2.0e7
    i_sec = i_scale * (1.0 + 0.05 * member_index)
    a_sec = 12.0 * i_sec
    return e_mod, a_sec, i_sec


class Pushdown:
    """Displacement-controlled second-order plastic-hinge pushdown."""

    def __init__(self, f: Frame, x: np.ndarray, removal: tuple[int, int],
                 second_order: bool = False, degrading: bool = False,
                 hinge_factor: float = 1.0, i_scale: float = 5.0e-5,
                 gravity_factor: float = 0.0):
        self.f = f
        self.x = np.asarray(x, float)
        self.removal = removal
        self.second_order = bool(second_order)
        self.degrading = bool(degrading)
        self.gravity_factor = float(gravity_factor)
        self.i_scale = float(i_scale)
        topo = frame_topology(f, removal)
        self.coords = np.asarray(topo["coords"], float)
        self.members = topo["members"]
        self.node_of = topo["node_of"]
        self.fixed = list(topo["fixed"])
        self.cut_node = int(topo["cut_node"])
        self.n_node = len(self.coords)
        self.n_member = len(self.members)
        self.n_struct = 3 * self.n_node
        self.n_dof = self.n_struct + 2 * self.n_member
        self.k_h = float(hinge_factor) * HINGE_REFERENCE_STIFFNESS

        self.joint_dof = np.empty((self.n_member, 2), dtype=int)
        self.hinge_dof = np.empty((self.n_member, 2), dtype=int)
        self.geom: list[dict] = []
        for i, member in enumerate(self.members):
            ni, nj = member["nodes"]
            xi, yi = self.coords[ni]
            xj, yj = self.coords[nj]
            dx, dy = xj - xi, yj - yi
            length = float(math.hypot(dx, dy))
            c, s = dx / length, dy / length
            e_mod, a_sec, i_sec = elastic_properties(member["index"], self.i_scale)
            ei = e_mod * i_sec
            ea = e_mod * a_sec
            kl = np.zeros((6, 6))
            kl[0, 0] = kl[3, 3] = ea / length
            kl[0, 3] = kl[3, 0] = -ea / length
            kl[1, 1] = kl[4, 4] = 12.0 * ei / length ** 3
            kl[1, 4] = kl[4, 1] = -12.0 * ei / length ** 3
            kl[1, 2] = kl[2, 1] = 6.0 * ei / length ** 2
            kl[1, 5] = kl[5, 1] = 6.0 * ei / length ** 2
            kl[4, 2] = kl[2, 4] = -6.0 * ei / length ** 2
            kl[4, 5] = kl[5, 4] = -6.0 * ei / length ** 2
            kl[2, 2] = kl[5, 5] = 4.0 * ei / length
            kl[2, 5] = kl[5, 2] = 2.0 * ei / length
            t = np.zeros((6, 6))
            t[0, 0] = t[1, 1] = c
            t[0, 1] = s
            t[1, 0] = -s
            t[3, 3] = t[4, 4] = c
            t[3, 4] = s
            t[4, 3] = -s
            t[2, 2] = t[5, 5] = 1.0
            self.geom.append({"L": length, "kl": kl, "T": t, "ei": ei})
            self.joint_dof[i] = [3 * ni + 2, 3 * nj + 2]
            self.hinge_dof[i] = [self.n_struct + 2 * i, self.n_struct + 2 * i + 1]

        self.theta_p_old = np.zeros((self.n_member, 2))
        self.theta_p = np.zeros((self.n_member, 2))
        self.moment = np.zeros((self.n_member, 2))
        self.yielded = np.zeros((self.n_member, 2), dtype=bool)
        self.max_plastic_rotation = 0.0
        self.fixed_dofs = set(3 * n + d for n in self.fixed for d in range(3))

    # -- hinge constitutive law -------------------------------------------
    def hinge_capacity(self, i: int, end: int) -> float:
        return self.members[i]["My"][end] * self.x[self.members[i]["index"]]

    def hinge_response(self, relative: float, theta_p: float, my: float):
        """Moment and algorithmic tangent of one hinge.

        Two backbones are available.  With ``degrading=False`` the hinge is
        elastic-perfectly-plastic.  With ``degrading=True`` it is bilinear: a
        plastic plateau up to ``SOFTENING_ONSET`` times the yield rotation,
        then a negative post-peak slope, which is the standard concentrated
        plasticity idealization used when a softening branch is required.
        """
        trial = self.k_h * (relative - theta_p)
        if abs(trial) <= my:
            return trial, self.k_h
        sign = 1.0 if trial > 0 else -1.0
        if not self.degrading:
            return sign * my, self.k_h
        theta_y = my / self.k_h
        theta_u = SOFTENING_ONSET * theta_y
        plastic = theta_p + sign * (abs(trial) - my) / self.k_h
        if abs(plastic) <= theta_u:
            return sign * my, self.k_h
        residual = self.k_h * SOFTENING_SLOPE
        cap = max(my * SOFTENING_RESIDUAL, my - residual * (abs(plastic) - theta_u))
        return sign * cap, residual

    # -- assembly ----------------------------------------------------------
    def assemble(self, u: np.ndarray, with_plastic: bool = True):
        """Return the tangent matrix and internal force vector."""
        k = np.zeros((self.n_dof, self.n_dof))
        f_int = np.zeros(self.n_dof)
        for i in range(self.n_member):
            g = self.geom[i]
            t = g["T"]
            hinge = self.hinge_dof[i]
            ni, nj = self.members[i]["nodes"]
            # member end i: two translations of the joint node plus its own
            # internal rotation; likewise for end j
            rows = np.array([3 * ni, 3 * ni + 1, int(hinge[0]),
                             3 * nj, 3 * nj + 1, int(hinge[1])])
            ul = t @ u[rows]
            kg = np.zeros((6, 6))
            if self.second_order:
                axial = g["kl"][0, 0] * (ul[3] - ul[0])
                length = g["L"]
                factor = axial / (30.0 * length)
                kg[1, 1] = kg[4, 4] = 36.0
                kg[1, 4] = kg[4, 1] = -36.0
                kg[1, 2] = kg[2, 1] = 3.0 * length
                kg[1, 5] = kg[5, 1] = 3.0 * length
                kg[4, 2] = kg[2, 4] = -3.0 * length
                kg[4, 5] = kg[5, 4] = -3.0 * length
                kg[2, 2] = kg[5, 5] = 4.0 * length ** 2
                kg[2, 5] = kg[5, 2] = -length ** 2
                kg *= factor
            tkt = t.T @ (g["kl"] + kg)
            k[np.ix_(rows, rows)] += tkt @ t
            f_int[rows] += tkt @ ul
            # hinge between the joint rotation and the internal member rotation
            for end in (0, 1):
                jd = int(self.joint_dof[i, end])
                hd = int(hinge[end])
                my = self.hinge_capacity(i, end)
                theta_p = self.theta_p_old[i, end] if with_plastic else 0.0
                moment, kappa = self.hinge_response(u[jd] - u[hd], theta_p, my)
                k[jd, jd] += kappa
                k[hd, hd] += kappa
                k[jd, hd] -= kappa
                k[hd, jd] -= kappa
                f_int[jd] += moment
                f_int[hd] -= moment
        return k, f_int

    # -- loads -------------------------------------------------------------
    def ext_force(self, lam: float) -> np.ndarray:
        load = np.zeros(self.n_dof)
        s, g = self.removal
        for r in range(s, self.f.H + 1):
            node = self.node_of.get(f"{g}_{r}")
            if node is None:
                continue
            load[3 * node + 1] -= float(self.f.loads[r - 1][g]) * lam
        if self.gravity_factor > 0.0:
            for r in range(1, self.f.H + 1):
                for j in range(self.f.B + 1):
                    node = self.node_of.get(f"{j}_{r}")
                    if node is None:
                        continue
                    load[3 * node + 1] -= (self.gravity_factor
                                           * float(self.f.loads[r - 1][j]) * lam)
        return load

    def load_direction(self) -> np.ndarray:
        return self.ext_force(1.0)

    def free_dofs(self) -> np.ndarray:
        return np.array([d for d in range(self.n_dof)
                         if d not in self.fixed_dofs], dtype=int)

    # -- return mapping ----------------------------------------------------
    def apply_return_map(self, u: np.ndarray) -> None:
        """Return-map every hinge at the current displacement state.

        The plastic rotation is obtained by inverting the active backbone branch,
        so perfect plasticity and the bilinear softening backbone share one code
        path and the converged state satisfies ``M = k_h(rel - theta_p)`` with the
        backbone restriction.
        """
        for i in range(self.n_member):
            for end in (0, 1):
                jd = int(self.joint_dof[i, end])
                hd = int(self.hinge_dof[i, end])
                my = self.hinge_capacity(i, end)
                relative = u[jd] - u[hd]
                moment, _ = self.hinge_response(relative, self.theta_p_old[i, end], my)
                elastic_trial = self.k_h * (relative - self.theta_p_old[i, end])
                if abs(elastic_trial) <= my:
                    # still elastic: no plastic rotation is generated
                    self.theta_p[i, end] = self.theta_p_old[i, end]
                    self.moment[i, end] = elastic_trial
                    continue
                sign = 1.0 if moment >= 0 else -1.0
                if not self.degrading:
                    self.theta_p[i, end] = relative - sign * my / self.k_h
                else:
                    # invert the bilinear backbone at |M| = |moment|
                    cap = abs(moment)
                    residual = self.k_h * SOFTENING_SLOPE
                    if cap >= my:
                        plastic = my / self.k_h + (my - cap) / residual
                    else:
                        plastic = cap / self.k_h
                    self.theta_p[i, end] = relative - sign * plastic
                self.moment[i, end] = moment
                self.yielded[i, end] = True

    def commit(self) -> None:
        self.theta_p_old = self.theta_p.copy()
        self.max_plastic_rotation = max(
            self.max_plastic_rotation, float(np.max(np.abs(self.theta_p))))

    # -- step solution -----------------------------------------------------
    def step_load(self, u0: np.ndarray, lam_target: float):
        """Load-controlled step with the plastic return mapping.

        Returns ``(u, converged, iterations)``.  The load factor is prescribed, so
        a non-converged step means the applied load exceeds the capacity and the
        last converged factor is the collapse multiplier.
        """
        u = u0.copy()
        free = self.free_dofs()
        k_el, _ = self.assemble(u, with_plastic=False)
        try:
            factor = np.linalg.cholesky(k_el[np.ix_(free, free)])
        except np.linalg.LinAlgError:
            return u, False, -1
        for iteration in range(1, MAX_ITER + 1):
            self.apply_return_map(u)
            _, f_int = self.assemble(u, with_plastic=True)
            r = (self.ext_force(lam_target) - f_int)[free]
            if np.max(np.abs(r)) <= NEWTON_ABS_TOL:
                self.commit()
                return u, True, iteration
            try:
                du = np.linalg.solve(factor.T, np.linalg.solve(factor, r))
            except np.linalg.LinAlgError:
                return u, False, -1
            if not np.all(np.isfinite(du)):
                return u, False, -1
            u[free] += du
        self.commit()
        return u, False, MAX_ITER

    def step(self, u0: np.ndarray, lam0: float, prescribed_dof: int,
             prescribed_value: float):
        """Displacement-controlled step (retained for auxiliary checks)."""
        u = u0.copy()
        u[prescribed_dof] = prescribed_value
        lam = float(lam0)
        free = self.free_dofs()
        free = free[free != prescribed_dof]
        k_el, _ = self.assemble(u, with_plastic=False)
        n = free.size
        border = np.zeros((n + 1, n + 1))
        border[:n, :n] = k_el[np.ix_(free, free)]
        p = self.load_direction()
        border[:n, n] = -p[free]
        border[n, :n] = k_el[prescribed_dof, free]
        border[n, n] = -p[prescribed_dof]
        try:
            np.linalg.cholesky(border[:n, :n])
        except np.linalg.LinAlgError:
            return u, lam, False, -1
        for iteration in range(1, MAX_ITER + 1):
            self.apply_return_map(u)
            _, f_int = self.assemble(u, with_plastic=True)
            residual = self.ext_force(lam) - f_int
            r = np.concatenate([residual[free], [residual[prescribed_dof]]])
            if np.max(np.abs(r)) <= NEWTON_ABS_TOL:
                self.commit()
                return u, lam, True, iteration
            try:
                delta = np.linalg.solve(border, r)
            except np.linalg.LinAlgError:
                return u, lam, False, -1
            if not np.all(np.isfinite(delta)):
                return u, lam, False, -1
            u[free] += delta[:n]
            lam += delta[n]
            u[prescribed_dof] = prescribed_value
        self.commit()
        return u, lam, False, MAX_ITER

    def reference_displacement(self) -> float:
        u = np.zeros(self.n_dof)
        k, _ = self.assemble(u, with_plastic=False)
        free = self.free_dofs()
        position = {int(d): i for i, d in enumerate(free)}
        rhs = self.load_direction()[free]
        solution = np.linalg.solve(k[np.ix_(free, free)], rhs)
        return abs(float(solution[position[3 * self.cut_node + 1]]))

    def run(self, num_steps: int = NUM_STEPS, drift_multiplier: float = 6.0,
            lam_max: float | None = None) -> dict:
        """Load-controlled pushdown.

        ``lambda`` is incremented in equal steps and the analysis stops at the
        first step that cannot be equilibrated.  The increment is sized from the
        elastic reference displacement and ``drift_multiplier`` so that the
        collapse displacement is reached within ``num_steps`` steps.
        """
        u = np.zeros(self.n_dof)
        self.theta_p_old[:] = 0.0
        self.theta_p[:] = 0.0
        self.yielded[:] = False
        reference = self.reference_displacement()
        prescribed = 3 * self.cut_node + 1
        if lam_max is None:
            lam_max = 64.0
        dlam = lam_max / num_steps
        history = []
        lam = 0.0
        for step_index in range(1, num_steps + 1):
            target = dlam * step_index
            u_new, ok, iterations = self.step_load(u, target)
            if not ok:
                history.append({"step": step_index, "lambda": None, "ok": False,
                                "iterations": iterations,
                                "disp": float(u[prescribed])})
                break
            u = u_new
            lam = target
            history.append({"step": step_index, "lambda": float(lam),
                            "disp": float(u[prescribed]), "ok": True,
                            "iterations": iterations})
        return {"converged": True, "history": history,
                "reference_displacement": float(reference),
                "lambda_increment": float(dlam),
                "lambda_max": float(lam)}
