#!/usr/bin/env python3
"""
Supra-Hodge Laplacians for Multi-View Reasoning — reference implementation.

This module mirrors the constructions in "Supra-Hodge Laplacians for Multi-View
Reasoning" (spectral operator L_p, energy identity (9), Algorithms 1–2).

Linear algebra is implemented from first principles (no NumPy/SciPy). Symmetric
eigenproblems use a classical Jacobi diagonalization. The OpenAI call uses only
the Python standard library (urllib + json).

Sections:
  1. Dense matrix utilities
  2. Jacobi eigendecomposition (symmetric)
  3. Simplicial boundary operators and Hodge Laplacians H_p
  4. Supra-Hodge block assembly (Definition 3.2)
  5. Energy, Rayleigh quotient, spectral projector
  6. Algorithm 1: simplicial lifting from scores
  7. Algorithm 2: reasoning pipeline + soft LLM augmentation
  8. Mock eight-node scientific-planning scenario (Section 7.1 narrative)
"""

from __future__ import annotations

import json
import math
import os
import random
import warnings
from pathlib import Path
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# 1. Dense matrix layer (real coefficients)
# ---------------------------------------------------------------------------

Matrix = List[List[float]]
Vector = List[float]


def mat_zeros(rows: int, cols: int) -> Matrix:
    return [[0.0 for _ in range(cols)] for _ in range(rows)]


def mat_identity(n: int) -> Matrix:
    m = mat_zeros(n, n)
    for i in range(n):
        m[i][i] = 1.0
    return m


def mat_copy(a: Matrix) -> Matrix:
    return [row[:] for row in a]


def mat_transpose(a: Matrix) -> Matrix:
    if not a:
        return []
    r, c = len(a), len(a[0])
    return [[a[i][j] for i in range(r)] for j in range(c)]


def mat_add(a: Matrix, b: Matrix) -> Matrix:
    r, c = len(a), len(a[0])
    return [[a[i][j] + b[i][j] for j in range(c)] for i in range(r)]


def mat_sub(a: Matrix, b: Matrix) -> Matrix:
    r, c = len(a), len(a[0])
    return [[a[i][j] - b[i][j] for j in range(c)] for i in range(r)]


def mat_scale(a: Matrix, s: float) -> Matrix:
    return [[a[i][j] * s for j in range(len(a[0]))] for i in range(len(a))]


def mat_mul(a: Matrix, b: Matrix) -> Matrix:
    """Matrix multiply: (r1 x c) * (c x c2) -> (r1 x c2)."""
    if not a or not b:
        return []
    r1, c = len(a), len(a[0])
    if len(b) != c:
        raise ValueError("mat_mul: incompatible inner dimensions.")
    c2 = len(b[0])
    out = mat_zeros(r1, c2)
    for i in range(r1):
        for k in range(c):
            aik = a[i][k]
            if aik == 0.0:
                continue
            bk = b[k]
            for j in range(c2):
                out[i][j] += aik * bk[j]
    return out


def mat_vec_mul(a: Matrix, x: Vector) -> Vector:
    if not a:
        return []
    if len(x) != len(a[0]):
        raise ValueError("Vector length must match column count of the matrix.")
    return [sum(a[i][j] * x[j] for j in range(len(x))) for i in range(len(a))]


def vec_dot(u: Vector, v: Vector) -> float:
    if len(u) != len(v):
        raise ValueError("vec_dot: vectors must have the same length.")
    return sum(ui * vi for ui, vi in zip(u, v))


def vec_sub(u: Vector, v: Vector) -> Vector:
    if len(u) != len(v):
        raise ValueError("vec_sub: vectors must have the same length.")
    return [a - b for a, b in zip(u, v)]


def vec_norm(u: Vector) -> float:
    return math.sqrt(vec_dot(u, u))


def vec_scale(u: Vector, s: float) -> Vector:
    return [ui * s for ui in u]


def vec_normalize(u: Vector) -> Vector:
    n = vec_norm(u)
    if n == 0.0:
        return u[:]
    return [ui / n for ui in u]


def block_matrix(blocks: List[List[Matrix]]) -> Matrix:
    """Assemble a block matrix from rectangular grid of blocks."""
    row_heights = [len(blocks[i][0]) for i in range(len(blocks))]
    col_widths = [len(blocks[0][j]) for j in range(len(blocks[0]))]
    for ri, row in enumerate(blocks):
        for j, blk in enumerate(row):
            if len(blk) != row_heights[ri] or len(blk[0]) != col_widths[j]:
                raise ValueError("Incompatible block sizes.")
    rtot = sum(len(blocks[i][0]) for i in range(len(blocks)))
    ctot = sum(len(blocks[0][j][0]) for j in range(len(blocks[0])))
    out = mat_zeros(rtot, ctot)
    ri = 0
    for i, row in enumerate(blocks):
        rh = len(row[0])
        cj = 0
        for j, blk in enumerate(row):
            cw = len(blk[0])
            for bi in range(rh):
                for bj in range(cw):
                    out[ri + bi][cj + bj] = blk[bi][bj]
            cj += cw
        ri += rh
    return out


def frobenius_offdiag_norm(a: Matrix) -> float:
    """Frobenius norm of strictly upper-triangle part (symmetry diagnostic)."""
    n = len(a)
    s = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            v = a[i][j]
            s += v * v
    return math.sqrt(2.0 * s)


# ---------------------------------------------------------------------------
# 2. Jacobi eigendecomposition for real symmetric matrices
# ---------------------------------------------------------------------------


def jacobi_symmetric_eigen(
    a_in: Matrix,
    tol: float = 1e-10,
    max_sweeps: int = 150,
) -> Tuple[List[float], Matrix]:
    """
    Classical Jacobi eigenvalue algorithm for real symmetric A.

    Returns:
      eigenvalues (ascending),
      orthogonal matrix V whose columns are eigenvectors (A V = V Λ).
    """
    n = len(a_in)
    if n == 0:
        return [], []
    a = mat_copy(a_in)
    v = mat_identity(n)

    converged = False
    for _ in range(max_sweeps):
        if frobenius_offdiag_norm(a) < tol:
            converged = True
            break
        for p in range(n - 1):
            for q in range(p + 1, n):
                apq = a[p][q]
                # Do not skip small |apq| using the same `tol` as the global stop criterion:
                # many modest entries can keep Frobenius off-norm above `tol` otherwise.
                if abs(apq) <= 1e-15 * max(1.0, abs(a[p][p]), abs(a[q][q])):
                    continue
                app, aqq = a[p][p], a[q][q]
                tau = (aqq - app) / (2.0 * apq)
                t = math.copysign(1.0, tau) / (abs(tau) + math.sqrt(1.0 + tau * tau))
                c = 1.0 / math.sqrt(1.0 + t * t)
                s = t * c

                # Rotate A <- J^T A J (symmetric update)
                for i in range(n):
                    if i != p and i != q:
                        aip, aiq = a[i][p], a[i][q]
                        a[i][p] = c * aip - s * aiq
                        a[p][i] = a[i][p]
                        a[i][q] = c * aiq + s * aip
                        a[q][i] = a[i][q]
                app, apq, aqp, aqq = a[p][p], a[p][q], a[q][p], a[q][q]
                a[p][p] = c * c * app - 2.0 * s * c * apq + s * s * aqq
                a[q][q] = s * s * app + 2.0 * s * c * apq + c * c * aqq
                a[p][q] = 0.0
                a[q][p] = 0.0

                # V <- V J
                for i in range(n):
                    vip, viq = v[i][p], v[i][q]
                    v[i][p] = c * vip - s * viq
                    v[i][q] = s * vip + c * viq

    if not converged and frobenius_offdiag_norm(a) >= tol:
        warnings.warn(
            "jacobi_symmetric_eigen: reached max_sweeps without full off-diagonal convergence; "
            "eigenvalues may be inaccurate for ill-conditioned inputs.",
            RuntimeWarning,
            stacklevel=2,
        )

    eigenvalues = [a[i][i] for i in range(n)]
    # Sort ascending and reorder columns of V
    idx = sorted(range(n), key=lambda i: eigenvalues[i])
    evals_sorted = [eigenvalues[i] for i in idx]
    v_sorted = mat_zeros(n, n)
    for j, old_j in enumerate(idx):
        for i in range(n):
            v_sorted[i][j] = v[i][old_j]
    return evals_sorted, v_sorted


# ---------------------------------------------------------------------------
# 3. Combinatorics: edges and triangles as aligned reference sets
# ---------------------------------------------------------------------------


def all_edges(n: int) -> List[Tuple[int, int]]:
    return [(i, j) for i in range(n) for j in range(i + 1, n)]


def all_triangles(n: int) -> List[Tuple[int, int, int]]:
    return [(i, j, k) for i in range(n) for j in range(i + 1, n) for k in range(j + 1, n)]


def index_map(items: Sequence[Tuple[int, ...]]) -> Dict[Tuple[int, ...], int]:
    return {t: i for i, t in enumerate(items)}


# ---------------------------------------------------------------------------
# 4. Boundary operators B1 (edges -> vertices) and B2 (triangles -> edges)
# ---------------------------------------------------------------------------


def build_boundary1(n_vertices: int, edge_index: Dict[Tuple[int, int], int]) -> Matrix:
    """
    Oriented incidence: column for edge (u,v), u < v, is +1 at v, -1 at u.
    Matches ∂[u,v] = v - u as a map C1 -> C0.
    """
    m = len(edge_index)
    b1 = mat_zeros(n_vertices, m)
    for (u, v), j in edge_index.items():
        b1[u][j] -= 1.0
        b1[v][j] += 1.0
    return b1


def build_boundary2(
    edge_index: Dict[Tuple[int, int], int],
    triangle_index: Dict[Tuple[int, int, int], int],
) -> Matrix:
    """
    ∂[v0,v1,v2] = [v1,v2] - [v0,v2] + [v0,v1] with increasing vertex order v0<v1<v2.
    """
    m = len(edge_index)
    t = len(triangle_index)
    b2 = mat_zeros(m, t)
    for (v0, v1, v2), ti in triangle_index.items():
        e01 = edge_index[(v0, v1)]
        e02 = edge_index[(v0, v2)]
        e12 = edge_index[(v1, v2)]
        b2[e01][ti] += 1.0
        b2[e02][ti] -= 1.0
        b2[e12][ti] += 1.0
    return b2


def hodge_laplacian_0(b1: Matrix) -> Matrix:
    """H0 = B1 B1^T (combinatorial graph Laplacian on vertices)."""
    b1t = mat_transpose(b1)
    return mat_mul(b1, b1t)


def hodge_laplacian_1(b1: Matrix, b2: Matrix) -> Matrix:
    """H1 = B1^T B1 + B2 B2^T (Definition 2.2, p=1)."""
    b1t = mat_transpose(b1)
    term1 = mat_mul(b1t, b1)
    if len(b2) == 0 or len(b2[0]) == 0:
        return term1
    b2t = mat_transpose(b2)
    term2 = mat_mul(b2, b2t)
    return mat_add(term1, term2)


def curl_component_energy(b2: Matrix, edge_signal: Vector) -> float:
    """v^T B2 B2^T v — highlights triangle / cycle inconsistency at order 1."""
    if len(b2) == 0 or len(b2[0]) == 0:
        return 0.0
    b2t = mat_transpose(b2)
    tmp = mat_vec_mul(b2t, edge_signal)
    return vec_dot(tmp, tmp)


# ---------------------------------------------------------------------------
# 5. Supra-Hodge Laplacian L_p (Definition 3.2)
# ---------------------------------------------------------------------------


def supra_hodge_laplacian(
    layer_hodge_matrices: Sequence[Matrix],
    omega: Matrix,
) -> Matrix:
    """
    Build L_p from layerwise H_p^{(i)} and scalar couplings ω_ij (symmetric).

    Diagonal block i:  H^{(i)} + (∑_{j≠i} ω_ij) I
    Off-diagonal (i,j):  -ω_ij I
    """
    k = len(layer_hodge_matrices)
    n = len(layer_hodge_matrices[0])
    for h in layer_hodge_matrices:
        if len(h) != n or len(h[0]) != n:
            raise ValueError("All H^{(i)} must share the same square dimension.")
    if len(omega) != k or any(len(row) != k for row in omega):
        raise ValueError("omega must be k x k.")

    blocks: List[List[Matrix]] = []
    for i in range(k):
        row_blocks: List[Matrix] = []
        sum_off = sum(omega[i][j] for j in range(k) if j != i)
        diag_base = mat_add(layer_hodge_matrices[i], mat_scale(mat_identity(n), sum_off))
        for j in range(k):
            if i == j:
                row_blocks.append(diag_base)
            else:
                row_blocks.append(mat_scale(mat_identity(n), -omega[i][j]))
        blocks.append(row_blocks)
    return block_matrix(blocks)


def aggregated_supra_hodge_operator(
    supra_blocks: Sequence[Matrix],
    alphas: Sequence[float],
) -> Matrix:
    """
    Definition 3.3: L_SH = diag(α₀L₀, α₁L₁, …, α_P L_P).
    Each L_p is already the full Supra-Hodge matrix at order p (same k-layer stack).
    """
    if len(supra_blocks) != len(alphas):
        raise ValueError("alphas must match number of blocks.")
    scaled = [mat_scale(blk, alphas[i]) for i, blk in enumerate(supra_blocks)]
    n_total = sum(len(b) for b in scaled)
    out = mat_zeros(n_total, n_total)
    off = 0
    for blk in scaled:
        d = len(blk)
        for i in range(d):
            for j in range(d):
                out[off + i][off + j] = blk[i][j]
        off += d
    return out


def supra_energy_quadratic_form(
    layer_signals: Sequence[Vector],
    layer_hodge: Sequence[Matrix],
    omega: Matrix,
) -> float:
    """Equation (9): sum_i v^{(i)T} H^{(i)} v^{(i)} + sum_{i<j} ω_ij ||v^{(i)}-v^{(j)}||^2."""
    k = len(layer_signals)
    e = 0.0
    for i in range(k):
        hi = layer_hodge[i]
        vi = layer_signals[i]
        e += vec_dot(vi, mat_vec_mul(hi, vi))
    for i in range(k):
        for j in range(i + 1, k):
            diff = vec_sub(layer_signals[i], layer_signals[j])
            e += omega[i][j] * vec_dot(diff, diff)
    return e


def rayleigh_quotient(l: Matrix, v: Vector) -> float:
    num = vec_dot(v, mat_vec_mul(l, v))
    den = vec_dot(v, v)
    return num / den if den != 0.0 else float("nan")


# ---------------------------------------------------------------------------
# 6. Algorithm 1 — simplicial lifting from embedding-derived scores
# ---------------------------------------------------------------------------


def cosine_similarity(a: Vector, b: Vector) -> float:
    na, nb = vec_norm(a), vec_norm(b)
    if na == 0.0 or nb == 0.0:
        return 0.0
    return vec_dot(a, b) / (na * nb)


def lift_simplices(
    embeddings: Sequence[Vector],
    tau_edge: float,
    tau_triangle: float,
    triplet_score: Callable[[int, int, int], float],
) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int, int]]]:
    """
    Algorithm 1 (paper): threshold pairwise similarity for edges, triplet rule for triangles.

    Here relation_rule1 is cosine similarity between vertex embeddings;
    relation_rule2 is supplied by the caller (e.g. minimum pairwise cosine minus slack).
    """
    n = len(embeddings)
    edges: List[Tuple[int, int]] = []
    for u in range(n):
        for v in range(u + 1, n):
            if cosine_similarity(embeddings[u], embeddings[v]) >= tau_edge:
                edges.append((u, v))
    triangles: List[Tuple[int, int, int]] = []
    for u in range(n):
        for v in range(u + 1, n):
            for w in range(v + 1, n):
                if triplet_score(u, v, w) >= tau_triangle:
                    triangles.append((u, v, w))
    return edges, triangles


# ---------------------------------------------------------------------------
# 7. Layer bundle: aligned incidence matrices per view
# ---------------------------------------------------------------------------


@dataclass
class LayerComplex:
    """One structural view: Hodge matrices embedded in the global simplex basis."""

    name: str
    description: str
    h0: Matrix
    h1: Matrix
    b1: Matrix
    b2: Matrix


def assemble_layer(
    name: str,
    description: str,
    n_vertices: int,
    edge_index: Dict[Tuple[int, int], int],
    triangle_index: Dict[Tuple[int, int, int], int],
    active_edges: Iterable[Tuple[int, int]],
    active_triangles: Iterable[Tuple[int, int, int]],
) -> LayerComplex:
    """
    Build B1, B2 with zero columns / rows for simplices absent in this layer
    (Assumption 3.1 — zero padding in the ambient reference basis).
    """
    m = len(edge_index)
    t = len(triangle_index)
    b1 = mat_zeros(n_vertices, m)
    b2 = mat_zeros(m, t)
    ae = set(active_edges)
    at = set(active_triangles)
    for e in ae:
        if e not in edge_index:
            raise ValueError(f"Edge {e} not in reference edge_index.")
        j = edge_index[e]
        u, v = e
        b1[u][j] -= 1.0
        b1[v][j] += 1.0
    for tri in at:
        if tri not in triangle_index:
            raise ValueError(f"Triangle {tri} not in reference triangle_index.")
        v0, v1, v2 = tri
        ti = triangle_index[tri]
        b2[edge_index[(v0, v1)]][ti] += 1.0
        b2[edge_index[(v0, v2)]][ti] -= 1.0
        b2[edge_index[(v1, v2)]][ti] += 1.0
    h0 = hodge_laplacian_0(b1)
    h1 = hodge_laplacian_1(b1, b2)
    return LayerComplex(name=name, description=description, h0=h0, h1=h1, b1=b1, b2=b2)


# ---------------------------------------------------------------------------
# 8. Spectral diagnostics (Section 4)
# ---------------------------------------------------------------------------


@dataclass
class SpectralDiagnostics:
    eigenvalues: List[float]
    eigenvectors: Matrix  # columns u_j
    fiedler_value: float  # λ_2 (second smallest)
    fiedler_vector: Vector
    spectral_gap: float  # λ_3 - λ_2 (if available)


def top_eigenpairs(
    l_matrix: Matrix,
    num_smallest: int = 5,
) -> SpectralDiagnostics:
    evals, evecs = jacobi_symmetric_eigen(l_matrix)
    if len(evals) < 2:
        return SpectralDiagnostics(evals, evecs, evals[0], [1.0], 0.0)
    f_val = evals[1]
    f_vec = [evecs[i][1] for i in range(len(evecs))]
    gap = evals[2] - evals[1] if len(evals) > 2 else 0.0
    return SpectralDiagnostics(
        eigenvalues=evals[: min(num_smallest, len(evals))],
        eigenvectors=evecs,
        fiedler_value=f_val,
        fiedler_vector=f_vec,
        spectral_gap=gap,
    )


def spectral_projector_apply(
    eigenvectors: Matrix,
    r: int,
    v: Vector,
) -> Vector:
    """Equation (11): hat v = P_{p,r} v with P = U_r U_r^T."""
    n = len(v)
    out = [0.0] * n
    if not eigenvectors or not eigenvectors[0]:
        return out
    for j in range(min(r, len(eigenvectors[0]))):
        uj = [eigenvectors[i][j] for i in range(n)]
        coeff = vec_dot(uj, v)
        for i in range(n):
            out[i] += coeff * uj[i]
    return out


# ---------------------------------------------------------------------------
# 9. OpenAI chat completion (stdlib only)
# ---------------------------------------------------------------------------


def load_env_file(path: Optional[Path] = None) -> None:
    """
    Load KEY=VALUE pairs from a .env file into os.environ.
    Does not override variables already set in the environment.
    Skips blank lines and lines starting with #.
    """
    candidates = []
    if path is not None:
        candidates.append(path)
    else:
        here = Path(__file__).resolve().parent
        candidates.extend([here / ".env", Path.cwd() / ".env"])
    for p in candidates:
        if not p.is_file():
            continue
        try:
            raw = p.read_text(encoding="utf-8")
        except OSError:
            continue
        for line in raw.splitlines():
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            if "=" not in s:
                continue
            key, _, value = s.partition("=")
            key = key.strip()
            if not key:
                continue
            value = value.strip().strip("'").strip('"')
            if key not in os.environ:
                os.environ[key] = value
        break


def openai_chat_completion(
    messages: List[Dict[str, str]],
    model: str = "gpt-4o-mini",
    api_key: Optional[str] = None,
    timeout_s: float = 120.0,
) -> str:
    """
    POST /v1/chat/completions. Set OPENAI_API_KEY or pass api_key=.

    Reads optional project `.env` (same directory as this file, or cwd) unless
    the variable is already set in the environment.

    If no key is present, returns a clear placeholder string instead of failing.
    """
    load_env_file()
    key = api_key or os.environ.get("OPENAI_API_KEY")
    if not key:
        return (
            "[OPENAI_API_KEY not set — skipping live API call.]\n\n"
            "Mock assistant reply: acknowledge the spectral summary and suggest "
            "resolving the evidential gap for the Query–Hypothesis–Fact triangle."
        )

    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.2,
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        "https://api.openai.com/v1/chat/completions",
        data=data,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {key}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        choices = body.get("choices") or []
        if not choices:
            return f"[OpenAI response missing choices] {json.dumps(body)[:500]}"
        msg = (choices[0].get("message") or {}) if isinstance(choices[0], dict) else {}
        content = msg.get("content")
        if content is None:
            return f"[OpenAI response missing message.content] {json.dumps(body)[:500]}"
        return str(content)
    except urllib.error.HTTPError as e:
        err_txt = e.read().decode("utf-8", errors="replace")
        return f"[HTTP {e.code}] {err_txt}"
    except Exception as e:
        return f"[OpenAI request error] {e!s}"


# ---------------------------------------------------------------------------
# 10. Algorithm 2 — Supra-Hodge reasoning pipeline (soft augmentation)
# ---------------------------------------------------------------------------


@dataclass
class ReasoningPipelineResult:
    entity_names: List[str]
    layer_names: List[str]
    diagnostics_order_p: str
    fiedler_value: float
    spectral_gap: float
    curl_by_layer: List[float]
    augmented_prompt: str
    llm_response: str


def format_spectral_augmentation(
    entity_names: Sequence[str],
    layer_labels: Sequence[str],
    diag: SpectralDiagnostics,
    stacked_edge_signal: Vector,
    layers: Sequence[LayerComplex],
    block_dim: int,
    k: int,
    edge_list: Optional[Sequence[Tuple[int, int]]] = None,
    order_p: int = 1,
) -> str:
    """Human-readable spectral summary (Section 6.5 template)."""
    n = len(entity_names)
    fvec = diag.fiedler_vector
    # Stacked L0: k blocks of length n (vertices)
    if order_p == 0 and len(fvec) == k * n:
        best_layer = 0
        best_idx = 0
        best_abs = -1.0
        for li in range(k):
            offset = li * n
            for i in range(n):
                val = abs(fvec[offset + i])
                if val > best_abs:
                    best_abs = val
                    best_idx = i
                    best_layer = li
        frontier = (
            f"primary conflict frontier emphasis on entity "
            f"'{entity_names[best_idx]}' (layer view '{layer_labels[best_layer]}')"
        )
    # Stacked L1: k blocks of length m (aligned edges)
    elif order_p == 1 and edge_list is not None and len(fvec) == k * len(edge_list):
        m = len(edge_list)
        best_layer = 0
        best_ei = 0
        best_abs = -1.0
        for li in range(k):
            offset = li * m
            for ei in range(m):
                val = abs(fvec[offset + ei])
                if val > best_abs:
                    best_abs = val
                    best_ei = ei
                    best_layer = li
        u, v = edge_list[best_ei]
        frontier = (
            f"primary conflict frontier on oriented edge ({entity_names[u]}–{entity_names[v]}) "
            f"in layer '{layer_labels[best_layer]}'"
        )
    else:
        frontier = "Fiedler structure available (inspect stacked vector coordinates)."

    curl_bits = []
    off = 0
    for li, layer in enumerate(layers):
        seg = stacked_edge_signal[off : off + block_dim]
        curl_bits.append(f"{layer.name}: curl energy ≈ {curl_component_energy(layer.b2, seg):.4f}")
        off += block_dim

    lines = [
        "Spectral analysis (Supra-Hodge soft augmentation):",
        f"- {frontier}",
        f"- Supra-Hodge Fiedler value λ₂ ≈ {diag.fiedler_value:.4f}",
        f"- Spectral gap (λ₃−λ₂) ≈ {diag.spectral_gap:.4f}",
        "- Per-layer curl indicator (edge signal against B₂ B₂ᵀ):",
    ]
    lines += [f"  * {c}" for c in curl_bits]
    return "\n".join(lines)


def supra_hodge_pipeline(
    entity_names: List[str],
    layers: List[LayerComplex],
    omega: Matrix,
    order_p: int,
    mock_edge_signal_factory: Callable[[LayerComplex], Vector],
    model: str = "gpt-4o-mini",
    edge_list: Optional[Sequence[Tuple[int, int]]] = None,
) -> ReasoningPipelineResult:
    """
    Algorithm 2:
      • assemble H^{(i)}_p (already in layers),
      • form L_p,
      • partial eigendecomposition (here: full Jacobi — small matrices),
      • diagnostics + soft prompt augmentation,
      • LLM call.
    """
    k = len(layers)
    if order_p == 0:
        h_mats = [ly.h0 for ly in layers]
    elif order_p == 1:
        h_mats = [ly.h1 for ly in layers]
    else:
        raise ValueError("This reference file implements p in {0, 1}.")

    l_supra = supra_hodge_laplacian(h_mats, omega)
    diag = top_eigenpairs(l_supra)

    block_dim = len(h_mats[0])
    stacked_signal: List[float] = []
    curl_energies: List[float] = []
    for ly in layers:
        sig = mock_edge_signal_factory(ly)
        stacked_signal.extend(sig)
        curl_energies.append(curl_component_energy(ly.b2, sig))

    aug = format_spectral_augmentation(
        entity_names,
        [ly.name for ly in layers],
        diag,
        stacked_signal,
        layers,
        block_dim,
        k,
        edge_list=list(edge_list) if edge_list is not None else None,
        order_p=order_p,
    )

    user_prompt = (
        f"You are assisting with multi-view scientific planning. Entities: {', '.join(entity_names)}.\n"
        f"The planner encodes {k} structural views (layers). Below is a mathematical spectral summary "
        f"of coupled higher-order consistency (Supra-Hodge). Use it as soft structural context — it does "
        f"not override factual reasoning.\n\n{aug}\n\n"
        "Question: In one short paragraph, where is the reasoning trace most fragile, and what evidence "
        "would most reduce that fragility?"
    )

    messages = [
        {"role": "system", "content": "You are a careful reasoning assistant."},
        {"role": "user", "content": user_prompt},
    ]
    reply = openai_chat_completion(messages, model=model)

    return ReasoningPipelineResult(
        entity_names=entity_names,
        layer_names=[ly.name for ly in layers],
        diagnostics_order_p=f"p={order_p}",
        fiedler_value=diag.fiedler_value,
        spectral_gap=diag.spectral_gap,
        curl_by_layer=curl_energies,
        augmented_prompt=user_prompt,
        llm_response=reply,
    )


# ---------------------------------------------------------------------------
# 11. Mock eight-node scientific planning scenario (Section 7.1 style)
# ---------------------------------------------------------------------------

ENTITY_NAMES_EIGHT = [
    "Query",
    "FactA",
    "FactB",
    "HypH",
    "ToolS",
    "ToolC",
    "State1",
    "State2",
]


def mock_embeddings_eight_node(seed: int = 42) -> List[Vector]:
    """Fabricated 8-D embeddings — not from a real encoder."""
    rnd = random.Random(seed)
    dim = 8
    names_n = len(ENTITY_NAMES_EIGHT)
    raw = [[rnd.gauss(0, 1) for _ in range(dim)] for _ in range(names_n)]
    # Nudge semantic neighbors: Query, FactA, HypH mutually close
    for idx in (0, 1, 3):
        for d in range(dim):
            raw[idx][d] += 0.35
    return raw


def triplet_rule_factory(
    emb: Sequence[Vector],
    slack: float,
) -> Callable[[int, int, int], float]:
    """Heuristic triplet score: min pairwise cosine minus slack."""

    def score(u: int, v: int, w: int) -> float:
        c = [
            cosine_similarity(emb[u], emb[v]),
            cosine_similarity(emb[u], emb[w]),
            cosine_similarity(emb[v], emb[w]),
        ]
        return min(c) - slack

    return score


def build_eight_node_case_study(
    include_evidential_triangle: bool,
) -> Tuple[List[LayerComplex], Matrix, List[Tuple[int, int]]]:
    """
    Four layers (semantic, evidential, task dependency, execution), shared vertices.

    Narrative: the semantic view contains a filled triangle on (Query, FactA, HypH).
    The evidential view may omit that 2-simplex until `include_evidential_triangle` is True,
    modeling higher-order disagreement between views (Section 7.1).
    """
    n = len(ENTITY_NAMES_EIGHT)
    emb = mock_embeddings_eight_node()

    tau_e, tau_t = 0.15, 0.02
    triplet_fn = triplet_rule_factory(emb, slack=0.05)

    # --- per-layer simplicial lifts (Algorithm 1) ---
    sem_edges, sem_tris = lift_simplices(emb, tau_e, tau_t, triplet_fn)
    evi_edges, evi_tris = lift_simplices(emb, tau_e + 0.02, tau_t + 0.03, triplet_fn)
    task_edges, task_tris = lift_simplices(emb, tau_e - 0.02, tau_t, triplet_fn)
    exe_edges, exe_tris = lift_simplices(emb, tau_e, tau_t - 0.01, triplet_fn)

    # Force narrative triangle (Query=0, FactA=1, HypH=3) in semantic layer
    key_tri = (0, 1, 3)
    key_edges = {(0, 1), (0, 3), (1, 3)}
    for e in key_edges:
        if e not in sem_edges:
            sem_edges.append(e)
    if key_tri not in sem_tris:
        sem_tris.append(key_tri)

    # Evidential: keep edges but optionally withhold triangle
    for e in key_edges:
        if e not in evi_edges:
            evi_edges.append(e)
    if include_evidential_triangle and key_tri not in evi_tris:
        evi_tris.append(key_tri)
    if not include_evidential_triangle:
        evi_tris = [t for t in evi_tris if t != key_tri]

    # Reference sets = union across layers. Every triangle boundary references three edges;
    # those edges must exist in the reference edge list even if no single layer listed them alone.
    edge_set = set(sem_edges) | set(evi_edges) | set(task_edges) | set(exe_edges)
    tri_set = set(sem_tris) | set(evi_tris) | set(task_tris) | set(exe_tris)
    for a, b, c in tri_set:
        edge_set.update([(a, b), (a, c), (b, c)])
    edge_list = sorted(edge_set)
    tri_list = sorted(tri_set)
    edge_index = index_map(edge_list)
    triangle_index = index_map(tri_list)

    omega = [
        [0.0, 0.35, 0.25, 0.25],
        [0.35, 0.0, 0.30, 0.30],
        [0.25, 0.30, 0.0, 0.28],
        [0.25, 0.30, 0.28, 0.0],
    ]

    layers = [
        assemble_layer(
            "semantic_similarity",
            "Conceptual neighborhood / embedding coherence",
            n,
            edge_index,
            triangle_index,
            sem_edges,
            sem_tris,
        ),
        assemble_layer(
            "evidential_support",
            "Citation-like reinforcement among facts and hypotheses",
            n,
            edge_index,
            triangle_index,
            evi_edges,
            evi_tris,
        ),
        assemble_layer(
            "task_dependency",
            "Which subgoals precede others",
            n,
            edge_index,
            triangle_index,
            task_edges,
            task_tris,
        ),
        assemble_layer(
            "execution_order",
            "Temporal / tool invocation structure",
            n,
            edge_index,
            triangle_index,
            exe_edges,
            exe_tris,
        ),
    ]
    return layers, omega, edge_list


def build_two_layer_focused_case_study(
    include_evidential_triangle: bool,
) -> Tuple[List[LayerComplex], Matrix, List[Tuple[int, int]]]:
    """
    Same narrative as the eight-node study, but only semantic vs. evidential layers
    with strong cross-coupling so λ₂ movement from closing the evidential 2-simplex is visible.
    """
    n = len(ENTITY_NAMES_EIGHT)
    emb = mock_embeddings_eight_node()
    tau_e, tau_t = 0.15, 0.02
    triplet_fn = triplet_rule_factory(emb, slack=0.05)
    sem_edges, sem_tris = lift_simplices(emb, tau_e, tau_t, triplet_fn)
    evi_edges, evi_tris = lift_simplices(emb, tau_e + 0.02, tau_t + 0.03, triplet_fn)
    key_tri = (0, 1, 3)
    key_edges = {(0, 1), (0, 3), (1, 3)}
    for e in key_edges:
        if e not in sem_edges:
            sem_edges.append(e)
    if key_tri not in sem_tris:
        sem_tris.append(key_tri)
    for e in key_edges:
        if e not in evi_edges:
            evi_edges.append(e)
    if include_evidential_triangle and key_tri not in evi_tris:
        evi_tris.append(key_tri)
    if not include_evidential_triangle:
        evi_tris = [t for t in evi_tris if t != key_tri]
    edge_set = set(sem_edges) | set(evi_edges)
    tri_set = set(sem_tris) | set(evi_tris)
    for a, b, c in tri_set:
        edge_set.update([(a, b), (a, c), (b, c)])
    edge_list = sorted(edge_set)
    tri_list = sorted(tri_set)
    edge_index = index_map(edge_list)
    triangle_index = index_map(tri_list)
    omega = [
        [0.0, 0.85],
        [0.85, 0.0],
    ]
    layers = [
        assemble_layer(
            "semantic_similarity",
            "Conceptual neighborhood / embedding coherence",
            n,
            edge_index,
            triangle_index,
            sem_edges,
            sem_tris,
        ),
        assemble_layer(
            "evidential_support",
            "Citation-like reinforcement among facts and hypotheses",
            n,
            edge_index,
            triangle_index,
            evi_edges,
            evi_tris,
        ),
    ]
    return layers, omega, edge_list


def uniform_edge_signal(layer: LayerComplex) -> Vector:
    """Fabricated 1-chain: unit flow on each edge column (length = #reference edges)."""
    m = len(layer.b1[0])
    if m == 0:
        return []
    return [1.0 for _ in range(m)]


def main() -> None:
    print("=== Supra-Hodge — two-layer focused comparison (semantic vs evidential, p=1) ===\n")

    ly_b, om2, _edge_ref = build_two_layer_focused_case_study(include_evidential_triangle=False)
    ly_a, om2_after, edge_ref = build_two_layer_focused_case_study(include_evidential_triangle=True)
    assert om2 == om2_after
    h1_b = [ly.h1 for ly in ly_b]
    h1_a = [ly.h1 for ly in ly_a]
    evals_b, _ = jacobi_symmetric_eigen(supra_hodge_laplacian(h1_b, om2))
    evals_a, _ = jacobi_symmetric_eigen(supra_hodge_laplacian(h1_a, om2))
    mu2_b = evals_b[1] if len(evals_b) > 1 else float("nan")
    mu2_a = evals_a[1] if len(evals_a) > 1 else float("nan")
    print(f"Before evidential 2-simplex on (Query, FactA, HypH): λ₂ ≈ {mu2_b:.4f}")
    print(f"After  evidential triangle closure:               λ₂ ≈ {mu2_a:.4f}")
    # Cross-layer disagreement ‖v^(1)−v^(2)‖ on the Supra-Hodge Fiedler vector — often
    # shrinks when views align on higher-order structure (Section 4 / Eq. 9).
    def _split_fiedler(l_mat: Matrix) -> Tuple[float, float]:
        d, vecs = jacobi_symmetric_eigen(l_mat)
        if len(d) < 2 or not vecs:
            return float("nan"), float("nan")
        f = [vecs[i][1] for i in range(len(vecs))]
        mblk = len(h1_b[0])
        return vec_norm(vec_sub(f[0:mblk], f[mblk : 2 * mblk])), d[1]

    dis_b, _ = _split_fiedler(supra_hodge_laplacian(h1_b, om2))
    dis_a, _ = _split_fiedler(supra_hodge_laplacian(h1_a, om2))
    print(f"  Fiedler cross-layer ‖v^(sem)−v^(evi)‖: {dis_b:.4f} → {dis_a:.4f}")
    sig2 = uniform_edge_signal(ly_b[1])
    curl_b = curl_component_energy(ly_b[1].b2, sig2)
    curl_a = curl_component_energy(ly_a[1].b2, sig2)
    print(f"  Evidential curl energy ‖B₂ᵀv‖² (uniform 1-chain): {curl_b:.4f} → {curl_a:.4f}")
    print(
        "  Section 7.1 reports μ₂ dropping from 0.14 to 0.03; reproducing those numbers "
        "requires the paper’s exact lifted complexes and couplings.\n"
    )

    print("=== Full four-layer eight-node run (p=1) ===\n")
    layers_before, omega, _ = build_eight_node_case_study(include_evidential_triangle=False)
    layers_after, omega2, edge_full = build_eight_node_case_study(include_evidential_triangle=True)
    assert omega == omega2
    h1_before = [ly.h1 for ly in layers_before]
    evals_full_b, _ = jacobi_symmetric_eigen(supra_hodge_laplacian(h1_before, omega))
    evals_full_a, _ = jacobi_symmetric_eigen(supra_hodge_laplacian([ly.h1 for ly in layers_after], omega))
    print(f"Four-layer λ₂ before / after evidential fix: {evals_full_b[1]:.4f} / {evals_full_a[1]:.4f}")
    print(
        "  (When many layers are stacked, a single local Hodge update can be spectrally muted — "
        "the two-layer block above isolates the effect.)\n"
    )

    # Energy identity check (9) on random stacked signal
    rnd = random.Random(0)
    k = len(layers_before)
    m = len(h1_before[0])
    vecs = [[rnd.gauss(0, 1) for _ in range(m)] for _ in range(k)]
    e_direct = vec_dot(
        _stack(vecs),
        mat_vec_mul(supra_hodge_laplacian(h1_before, omega), _stack(vecs)),
    )
    e_formula = supra_energy_quadratic_form(vecs, h1_before, omega)
    print(f"Energy consistency check |direct - formula| (Eq. 9): {abs(e_direct - e_formula):.2e}\n")

    # Full LLM pipeline (mock spectral signal)
    print("=== Algorithm 2 — soft augmentation + OpenAI ===\n")
    res = supra_hodge_pipeline(
        ENTITY_NAMES_EIGHT,
        layers_after,
        omega,
        order_p=1,
        mock_edge_signal_factory=uniform_edge_signal,
        edge_list=edge_full,
    )
    print(res.augmented_prompt[:1200] + ("...\n" if len(res.augmented_prompt) > 1200 else "\n"))
    print("--- LLM response ---\n")
    print(res.llm_response)


def _stack(chunks: List[Vector]) -> Vector:
    out: Vector = []
    for c in chunks:
        out.extend(c)
    return out


if __name__ == "__main__":
    main()
