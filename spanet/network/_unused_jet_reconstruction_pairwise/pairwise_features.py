"""Pairwise interaction features for Particle Transformer-style attention.

This module computes pairwise geometric features between particles and embeds
them into attention bias format. The features are added to attention scores
before softmax, implementing: attention = softmax((QK^T)/sqrt(d) + U) @ V

Reference: "Particle Transformer for Jet Tagging" - https://arxiv.org/abs/2202.03772
"""
import math
from typing import Dict, List, Optional, Tuple

import torch
from torch import nn, Tensor


def sincos_to_phi(sinphi: Tensor, cosphi: Tensor) -> Tensor:
    """Convert sin(phi) and cos(phi) to phi angle using atan2.

    Parameters
    ----------
    sinphi : Tensor
        Sine of phi angle
    cosphi : Tensor
        Cosine of phi angle

    Returns
    -------
    Tensor
        Phi angle in range [-pi, pi]
    """
    return torch.atan2(sinphi, cosphi)


def delta_phi(phi1: Tensor, phi2: Tensor) -> Tensor:
    """Compute delta phi with proper wrapping to [-pi, pi].

    Parameters
    ----------
    phi1, phi2 : Tensor
        Phi angles

    Returns
    -------
    Tensor
        Delta phi in range [-pi, pi]
    """
    dphi = phi1 - phi2
    dphi = torch.fmod(dphi + math.pi, 2 * math.pi) - math.pi
    return dphi


def auto_detect_kinematic_features(
    feature_names: List[str],
    input_source: str = ""
) -> Dict:
    """Auto-detect kinematic feature indices from feature names.

    Searches for pt, eta, phi (or sinphi+cosphi), and mass in the feature list.

    Parameters
    ----------
    feature_names : List[str]
        List of feature names from the event info
    input_source : str
        Name of input source (for error messages)

    Returns
    -------
    Dict
        Dictionary with keys:
        - pt_idx: index of pt feature
        - eta_idx: index of eta feature
        - phi_idx: index of phi feature (if direct phi available)
        - sinphi_idx: index of sinphi feature (if sin/cos representation)
        - cosphi_idx: index of cosphi feature (if sin/cos representation)
        - mass_idx: index of mass feature (-1 if not found)
        - use_sincos_phi: bool, True if using sinphi/cosphi
    """
    feature_names_lower = [f.lower() for f in feature_names]

    result = {
        'pt_idx': -1,
        'eta_idx': -1,
        'phi_idx': -1,
        'sinphi_idx': -1,
        'cosphi_idx': -1,
        'mass_idx': -1,
        'use_sincos_phi': False
    }

    # Find pt
    for i, name in enumerate(feature_names_lower):
        if name == 'pt':
            result['pt_idx'] = i
            break

    # Find eta
    for i, name in enumerate(feature_names_lower):
        if name == 'eta':
            result['eta_idx'] = i
            break

    # Find phi - prefer direct phi, fall back to sinphi/cosphi
    for i, name in enumerate(feature_names_lower):
        if name == 'phi':
            result['phi_idx'] = i
            break

    if result['phi_idx'] == -1:
        # Look for sinphi/cosphi
        for i, name in enumerate(feature_names_lower):
            if name == 'sinphi' or name == 'sin_phi':
                result['sinphi_idx'] = i
            elif name == 'cosphi' or name == 'cos_phi':
                result['cosphi_idx'] = i

        if result['sinphi_idx'] != -1 and result['cosphi_idx'] != -1:
            result['use_sincos_phi'] = True

    # Find mass
    for i, name in enumerate(feature_names_lower):
        if name == 'mass':
            result['mass_idx'] = i
            break

    # Validate required features
    if result['pt_idx'] == -1:
        raise ValueError(f"Could not find 'pt' feature in {input_source}: {feature_names}")
    if result['eta_idx'] == -1:
        raise ValueError(f"Could not find 'eta' feature in {input_source}: {feature_names}")
    if result['phi_idx'] == -1 and not result['use_sincos_phi']:
        raise ValueError(f"Could not find 'phi' or 'sinphi'/'cosphi' features in {input_source}: {feature_names}")

    return result


class PairwiseFeatureComputer(nn.Module):
    """Compute Particle Transformer-style pairwise interaction features.

    Computes pairwise geometric features between all particle pairs:
    - ln(kT): transverse momentum scale = log(min(pt_i, pt_j) * deltaR)
    - ln(z): momentum fraction = log(min(pt_i, pt_j) / (pt_i + pt_j))
    - ln(deltaR): angular distance = log(sqrt(deta^2 + dphi^2))
    - ln(m^2): invariant mass squared = log((E_i + E_j)^2 - |p_i + p_j|^2)

    These features encode physics-aware geometric relationships.

    Parameters
    ----------
    num_features : int
        Number of pairwise features to compute (1-4)
    eps : float
        Small constant for numerical stability in log operations
    """

    def __init__(self, num_features: int = 4, eps: float = 1e-7):
        super().__init__()
        self.num_features = num_features
        self.eps = eps

    def forward(
        self,
        pt: Tensor,
        eta: Tensor,
        phi: Tensor,
        mass: Optional[Tensor] = None,
        mask: Optional[Tensor] = None
    ) -> Tensor:
        """Compute pairwise features.

        Parameters
        ----------
        pt : Tensor [B, N]
            Transverse momentum
        eta : Tensor [B, N]
            Pseudorapidity
        phi : Tensor [B, N]
            Azimuthal angle
        mass : Tensor [B, N], optional
            Particle mass (defaults to 0 if not provided)
        mask : Tensor [B, N], optional
            Boolean mask, True for valid particles

        Returns
        -------
        Tensor [B, N, N, num_features]
            Pairwise features for all particle pairs
        """
        batch_size, num_particles = pt.shape

        if mass is None:
            mass = torch.zeros_like(pt)

        # Compute 4-vectors from (pt, eta, phi, mass)
        px = pt * torch.cos(phi)
        py = pt * torch.sin(phi)
        pz = pt * torch.sinh(eta)
        energy = torch.sqrt(pt**2 * torch.cosh(eta)**2 + mass**2 + self.eps)

        # Expand for pairwise computation: [B, N, 1] and [B, 1, N]
        pt_i, pt_j = pt.unsqueeze(2), pt.unsqueeze(1)
        eta_i, eta_j = eta.unsqueeze(2), eta.unsqueeze(1)
        phi_i, phi_j = phi.unsqueeze(2), phi.unsqueeze(1)
        px_i, px_j = px.unsqueeze(2), px.unsqueeze(1)
        py_i, py_j = py.unsqueeze(2), py.unsqueeze(1)
        pz_i, pz_j = pz.unsqueeze(2), pz.unsqueeze(1)
        e_i, e_j = energy.unsqueeze(2), energy.unsqueeze(1)

        features = []

        # Common computations
        d_eta = eta_i - eta_j
        d_phi = delta_phi(phi_i, phi_j)
        delta_r = torch.sqrt(d_eta**2 + d_phi**2 + self.eps)
        pt_min = torch.minimum(pt_i, pt_j)

        # Feature 1: ln(kT) - transverse momentum scale
        # kT = min(pT_i, pT_j) * deltaR
        kt = pt_min * delta_r
        features.append(torch.log(kt.clamp(min=self.eps)))

        if self.num_features >= 2:
            # Feature 2: ln(z) - momentum fraction
            # z = min(pT_i, pT_j) / (pT_i + pT_j)
            z = pt_min / (pt_i + pt_j + self.eps)
            features.append(torch.log(z.clamp(min=self.eps)))

        if self.num_features >= 3:
            # Feature 3: ln(deltaR) - angular distance
            features.append(torch.log(delta_r.clamp(min=self.eps)))

        if self.num_features >= 4:
            # Feature 4: ln(m^2) - invariant mass squared
            # m^2 = (E_i + E_j)^2 - (px_i + px_j)^2 - (py_i + py_j)^2 - (pz_i + pz_j)^2
            m2 = (e_i + e_j)**2 - (px_i + px_j)**2 - (py_i + py_j)**2 - (pz_i + pz_j)**2
            m2 = m2.clamp(min=self.eps)  # Ensure positive
            features.append(torch.log(m2))

        # Stack features: [B, N, N, num_features]
        pairwise_features = torch.stack(features, dim=-1)

        # Apply mask: set invalid pairs to 0
        if mask is not None:
            pair_mask = mask.unsqueeze(2) & mask.unsqueeze(1)  # [B, N, N]
            pairwise_features = pairwise_features * pair_mask.unsqueeze(-1)

        return pairwise_features


class PairwiseEmbedding(nn.Module):
    """Embed pairwise features into attention bias format.

    Follows Particle Transformer's PairEmbed approach:
    - Input: [B, N, N, num_features]
    - Output: [B * num_heads, N, N] (to be added to attention scores)

    Parameters
    ----------
    num_features : int
        Number of input pairwise features
    num_heads : int
        Number of attention heads
    embed_dim : int
        Hidden dimension for embedding MLP
    """

    def __init__(self, num_features: int, num_heads: int, embed_dim: int = 8):
        super().__init__()
        self.num_heads = num_heads
        self.embed_dim = embed_dim

        # MLP to embed pairwise features
        self.embed = nn.Sequential(
            nn.Linear(num_features, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, num_heads)
        )

    def forward(self, pairwise_features: Tensor) -> Tensor:
        """Embed pairwise features into attention bias.

        Parameters
        ----------
        pairwise_features : Tensor [B, N, N, num_features]
            Pairwise features from PairwiseFeatureComputer

        Returns
        -------
        Tensor [B * num_heads, N, N]
            Attention bias to add to attention scores
        """
        batch_size, seq_len, _, _ = pairwise_features.shape

        # Embed: [B, N, N, num_features] -> [B, N, N, num_heads]
        bias = self.embed(pairwise_features)

        # Reshape for attention: [B, N, N, H] -> [B, H, N, N] -> [B*H, N, N]
        bias = bias.permute(0, 3, 1, 2).contiguous()
        bias = bias.view(batch_size * self.num_heads, seq_len, seq_len)

        return bias
