"""Pairwise interaction features for Particle Transformer-style attention.

This module computes pairwise geometric features between particles and embeds
them into attention bias format. The features are added to attention scores
before softmax, implementing: attention = softmax((QK^T)/sqrt(d) + U) @ V

The feature computer supports both same-type blocks (a collection with itself,
e.g. Jets-Jets) and cross-type blocks (two different collections, e.g.
Jets-BoostedJets) so that resonances built from more than one kind of jet can
receive proper pairwise features. Each block is embedded by its own MLP.

Reference: "Particle Transformer for Jet Tagging" - https://arxiv.org/abs/2202.03772
"""

import math
from typing import Dict, List, Optional, Tuple

import torch
from torch import nn, Tensor


def sincos_to_phi(sinphi: Tensor, cosphi: Tensor) -> Tensor:
    """Convert sin(phi) and cos(phi) to phi angle using atan2."""
    return torch.atan2(sinphi, cosphi)


def delta_phi(phi1: Tensor, phi2: Tensor) -> Tensor:
    """Compute delta phi with proper wrapping to [-pi, pi]."""
    dphi = phi1 - phi2
    dphi = torch.fmod(dphi + math.pi, 2 * math.pi) - math.pi
    return dphi


def _matches(name: str, token: str) -> bool:
    """Whether a (lowercased) feature name refers to a given kinematic token.

    Matches an exact name (``pt``) or a prefixed name (``fj_pt``, ``ak8_pt``)
    so that different jet collections that decorate their features with a
    collection-specific prefix are all detected. The ``_`` separator guards
    against accidental substring hits such as ``sdmass`` matching ``mass``.
    """
    return name == token or name.endswith("_" + token)


def auto_detect_kinematic_features(
    feature_names: List[str],
    input_source: str = "",
) -> Dict:
    """Auto-detect pt, eta, phi (or sinphi/cosphi), and mass indices.

    Handles both bare feature names (``pt``, ``eta``, ``mass``) and names that
    carry a collection-specific prefix (``fj_pt``, ``fj_eta``, ``fj_mass``),
    which is required to build pairwise features for boosted-jet collections.
    """
    feature_names_lower = [f.lower() for f in feature_names]

    result = {
        "pt_idx": -1,
        "eta_idx": -1,
        "phi_idx": -1,
        "sinphi_idx": -1,
        "cosphi_idx": -1,
        "mass_idx": -1,
        "use_sincos_phi": False,
    }

    for i, name in enumerate(feature_names_lower):
        if _matches(name, "pt"):
            result["pt_idx"] = i
            break

    for i, name in enumerate(feature_names_lower):
        if _matches(name, "eta"):
            result["eta_idx"] = i
            break

    for i, name in enumerate(feature_names_lower):
        if _matches(name, "phi"):
            result["phi_idx"] = i
            break

    if result["phi_idx"] == -1:
        for i, name in enumerate(feature_names_lower):
            if _matches(name, "sinphi") or _matches(name, "sin_phi"):
                result["sinphi_idx"] = i
            elif _matches(name, "cosphi") or _matches(name, "cos_phi"):
                result["cosphi_idx"] = i

        if result["sinphi_idx"] != -1 and result["cosphi_idx"] != -1:
            result["use_sincos_phi"] = True

    # Prefer a plain mass ("mass"/"fj_mass") over derived masses such as
    # "sdmass" (soft-drop mass); "_mass" suffix matching already excludes
    # sdmass, but we keep the first plain-mass hit for clarity.
    for i, name in enumerate(feature_names_lower):
        if _matches(name, "mass"):
            result["mass_idx"] = i
            break

    if result["pt_idx"] == -1:
        raise ValueError(f"Could not find 'pt' feature in {input_source}: {feature_names}")
    if result["eta_idx"] == -1:
        raise ValueError(f"Could not find 'eta' feature in {input_source}: {feature_names}")
    if result["phi_idx"] == -1 and not result["use_sincos_phi"]:
        raise ValueError(
            f"Could not find 'phi' or 'sinphi'/'cosphi' features in {input_source}: {feature_names}"
        )

    return result


class PairwiseFeatureComputer(nn.Module):
    """Compute Particle Transformer-style pairwise interaction features.

    Works for a collection with itself (same-type) or for two different
    collections (cross-type). Given the physical kinematics of set ``i``
    (rows) and set ``j`` (columns) it returns an ``(B, Ni, Nj, F)`` tensor of
    log-scaled interaction features ``ln(kT), ln(z), ln(deltaR), ln(m^2)``.
    """

    def __init__(self, num_features: int = 4, eps: float = 1e-7):
        super().__init__()
        self.num_features = num_features
        self.eps = eps

    @staticmethod
    def _four_vector(pt: Tensor, eta: Tensor, phi: Tensor, mass: Tensor, eps: float):
        px = pt * torch.cos(phi)
        py = pt * torch.sin(phi)
        pz = pt * torch.sinh(eta)
        energy = torch.sqrt((pt * torch.cosh(eta)) ** 2 + mass ** 2 + eps)
        return px, py, pz, energy

    def forward(
        self,
        kinematics_i: Tuple[Tensor, Tensor, Tensor, Optional[Tensor]],
        kinematics_j: Tuple[Tensor, Tensor, Tensor, Optional[Tensor]],
        mask_i: Optional[Tensor] = None,
        mask_j: Optional[Tensor] = None,
        remove_self_pair: bool = False,
    ) -> Tensor:
        """Compute pairwise features between set i (rows) and set j (columns).

        Parameters
        ----------
        kinematics_i, kinematics_j: (pt, eta, phi, mass)
            Physical (de-normalized) kinematics. ``pt`` has shape ``(B, N)``.
            ``mass`` may be ``None``.
        mask_i, mask_j: [B, N], optional
            Positive masks (True = real vector). Padding pairs are zeroed.
        remove_self_pair: bool
            If True (same-type blocks), zero the diagonal i == j so we do not
            inject spurious biases from self-pairs.

        Returns
        -------
        [B, Ni, Nj, F]
        """
        pt_i, eta_i, phi_i, mass_i = kinematics_i
        pt_j, eta_j, phi_j, mass_j = kinematics_j

        if mass_i is None:
            mass_i = torch.zeros_like(pt_i)
        if mass_j is None:
            mass_j = torch.zeros_like(pt_j)

        # Following ParT: use ln(kt), ln(z), ln(delta), ln(m^2) built from kinematics.
        # We approximate rapidity with pseudorapidity (eta), reasonable for boosted jets.
        px_i, py_i, pz_i, e_i = self._four_vector(pt_i, eta_i, phi_i, mass_i, self.eps)
        px_j, py_j, pz_j, e_j = self._four_vector(pt_j, eta_j, phi_j, mass_j, self.eps)

        # Broadcast rows (i, dim 2) against columns (j, dim 1) -> (B, Ni, Nj).
        pt_i, pt_j = pt_i.unsqueeze(2), pt_j.unsqueeze(1)
        eta_i, eta_j = eta_i.unsqueeze(2), eta_j.unsqueeze(1)
        phi_i, phi_j = phi_i.unsqueeze(2), phi_j.unsqueeze(1)
        px_i, px_j = px_i.unsqueeze(2), px_j.unsqueeze(1)
        py_i, py_j = py_i.unsqueeze(2), py_j.unsqueeze(1)
        pz_i, pz_j = pz_i.unsqueeze(2), pz_j.unsqueeze(1)
        e_i, e_j = e_i.unsqueeze(2), e_j.unsqueeze(1)

        features = []

        d_eta = eta_i - eta_j
        d_phi = delta_phi(phi_i, phi_j)
        delta_r = torch.sqrt(d_eta ** 2 + d_phi ** 2 + self.eps)
        pt_min = torch.minimum(pt_i, pt_j)

        kt = pt_min * delta_r
        features.append(torch.log(kt.clamp(min=self.eps)))

        if self.num_features >= 2:
            z = pt_min / (pt_i + pt_j + self.eps)
            features.append(torch.log(z.clamp(min=self.eps)))

        if self.num_features >= 3:
            features.append(torch.log(delta_r.clamp(min=self.eps)))

        if self.num_features >= 4:
            m2 = (e_i + e_j) ** 2 - (px_i + px_j) ** 2 - (py_i + py_j) ** 2 - (pz_i + pz_j) ** 2
            m2 = m2.clamp(min=self.eps)
            features.append(torch.log(m2))

        pairwise_features = torch.stack(features, dim=-1)

        # Neutralize self-pairs (i == j) on same-type blocks, like ParT's
        # optional remove_self_pair. Avoids large negative logs on the diagonal.
        if remove_self_pair and pairwise_features.shape[1] == pairwise_features.shape[2]:
            num_particles = pairwise_features.shape[1]
            diag = torch.arange(num_particles, device=pairwise_features.device)
            pairwise_features = pairwise_features.clone()
            pairwise_features[:, diag, diag, :] = 0.0

        if mask_i is not None and mask_j is not None:
            pair_mask = mask_i.unsqueeze(2) & mask_j.unsqueeze(1)
            pairwise_features = pairwise_features * pair_mask.unsqueeze(-1)

        return pairwise_features


class PairwiseEmbedding(nn.Module):
    """Embed a block of pairwise features into per-head attention bias (ParT-style).

    ParT uses BatchNorm + 1x1 Conv stacks to embed pairwise features into a
    per-head attention bias. We implement the same idea here. The block may be
    rectangular (``R != C``) to support cross-type collections.
    """

    def __init__(self, num_features: int, num_heads: int, embed_dim: int = 8):
        super().__init__()
        self.num_heads = num_heads
        self.embed_dim = embed_dim

        # Work on flattened pairs: (B, F, R*C). Use BN over feature channels like ParT.
        self.input_bn = nn.BatchNorm1d(num_features)
        self.embed = nn.Sequential(
            nn.Conv1d(num_features, embed_dim, kernel_size=1),
            nn.BatchNorm1d(embed_dim),
            nn.GELU(),
            nn.Conv1d(embed_dim, embed_dim, kernel_size=1),
            nn.BatchNorm1d(embed_dim),
            nn.GELU(),
            nn.Conv1d(embed_dim, num_heads, kernel_size=1),
        )

    def forward(self, pairwise_features: Tensor) -> Tensor:
        batch_size, num_rows, num_cols, num_features = pairwise_features.shape

        # (B, R, C, F) -> (B, F, R*C)
        x = pairwise_features.permute(0, 3, 1, 2).contiguous().view(batch_size, num_features, num_rows * num_cols)
        x = self.input_bn(x)
        x = self.embed(x)  # (B, H, R*C)
        x = x.view(batch_size, self.num_heads, num_rows, num_cols)
        x = x.reshape(batch_size * self.num_heads, num_rows, num_cols)
        return x
