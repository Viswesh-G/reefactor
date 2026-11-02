"""Forced surface heat/salinity budgets in physical units on a spherical grid.

These are reduced mixed-layer constraints, not a closed coral physiology model.
Unresolved entrainment, vertical mixing and reef exchange remain model discrepancy.
"""
import torch
from torch import nn

EARTH_RADIUS = 6_371_000.0


def transport_terms(field, u, v, latitude, longitude, diffusivity):
    """Return interior advection and div(K grad(field)); lat/lon increase."""
    if field.shape[-2] < 3 or field.shape[-1] < 3:
        raise ValueError("Physics stencil needs at least 3 x 3 cells")
    lat = torch.as_tensor(latitude, dtype=field.dtype, device=field.device)
    lon = torch.as_tensor(longitude, dtype=field.dtype, device=field.device)
    if lat.ndim != 1 or lon.ndim != 1 or len(lat) != field.shape[-2] or len(lon) != field.shape[-1]:
        raise ValueError("Coordinate axes must match the field")
    dlat, dlon = torch.diff(torch.deg2rad(lat)), torch.diff(torch.deg2rad(lon))
    if not bool((dlat > 0).all() and (dlon > 0).all()) or not (
            torch.allclose(dlat, dlat[0].expand_as(dlat), rtol=1e-4, atol=1e-8) and
            torch.allclose(dlon, dlon[0].expand_as(dlon), rtol=1e-4, atol=1e-8)):
        raise ValueError("Physics requires increasing regular coordinates")
    phi = torch.deg2rad(lat)
    cosine = torch.cos(phi[1:-1])[:, None]
    if bool((cosine.abs() < 0.01).any()):
        raise ValueError("Use a polar grid near the poles")
    dx = EARTH_RADIUS * cosine * dlon[0]
    dy = EARTH_RADIUS * dlat[0]
    center = field[..., 1:-1, 1:-1]
    east, west = field[..., 1:-1, 2:], field[..., 1:-1, :-2]
    north, south = field[..., 2:, 1:-1], field[..., :-2, 1:-1]
    k = torch.as_tensor(diffusivity, device=field.device, dtype=field.dtype) + torch.zeros_like(field)
    if bool((k <= 0).any()):
        raise ValueError("Diffusivity must be positive")
    kc = k[..., 1:-1, 1:-1]
    ke, kw = (kc+k[..., 1:-1, 2:])/2, (kc+k[..., 1:-1, :-2])/2
    kn, ks = (kc+k[..., 2:, 1:-1])/2, (kc+k[..., :-2, 1:-1])/2
    cos_n = torch.cos((phi[1:-1]+phi[2:])/2)[:, None]
    cos_s = torch.cos((phi[1:-1]+phi[:-2])/2)[:, None]
    diffusion = (ke*(east-center)-kw*(center-west))/dx.square()
    diffusion = diffusion + (cos_n*kn*(north-center)-cos_s*ks*(center-south))/(cosine*dy.square())
    advection = u[..., 1:-1, 1:-1]*(east-west)/(2*dx) + v[..., 1:-1, 1:-1]*(north-south)/(2*dy)
    return advection, diffusion


def stencil_mask(valid):
    return (valid[..., 1:-1, 1:-1] & valid[..., 2:, 1:-1] & valid[..., :-2, 1:-1]
            & valid[..., 1:-1, 2:] & valid[..., 1:-1, :-2])


def masked_mse(residual, valid, scale):
    if scale <= 0 or not bool(valid.any()):
        raise ValueError("Need a positive residual scale and at least one valid stencil")
    return (residual[valid]/scale).square().mean()


class ForcedSurfaceBudget(nn.Module):
    def __init__(self, diffusivity=500.0, rho=1025.0, cp=3990.0, dt=86400.0, learn_k=False):
        super().__init__()
        if min(diffusivity, rho, cp, dt) <= 0:
            raise ValueError("Physical constants must be positive")
        self.log_k = nn.Parameter(torch.tensor(float(diffusivity)).log(), requires_grad=learn_k)
        self.rho, self.cp, self.dt = rho, cp, dt

    def forward(self, prediction, forcing, latitude, longitude, kind="heat", scale=1e-5):
        """prediction: B,T,H,W in Celsius or practical salinity, including t0.

        Forcing arrays have B,T-1,H,W. Use issuance-available forcing for forecasts.
        No future reanalysis may enter an operational forecast's input channel.
        """
        if prediction.ndim != 4 or prediction.shape[1] < 2:
            raise ValueError("Expected B,T,H,W including at least one transition")
        prev, current = prediction[:, :-1], prediction[:, 1:]
        needed = ["water_u", "water_v", "mixed_layer_depth"]
        needed += ["net_heat_flux"] if kind == "heat" else ["freshwater_flux"]
        if kind not in ("heat", "salt"):
            raise ValueError("kind must be heat or salt")
        for name in needed:
            if name not in forcing or forcing[name].shape != prev.shape:
                raise ValueError(f"{name} must have shape {tuple(prev.shape)}")
        valid = torch.isfinite(prev) & torch.isfinite(current)
        for name in needed:
            valid = valid & torch.isfinite(forcing[name])
        valid = valid & (forcing["mixed_layer_depth"] > 0)
        if "valid" in forcing:
            valid = valid & forcing["valid"].bool()
        # Replace only masked values before arithmetic; retain masks in loss.
        p, c = torch.nan_to_num(prev), torch.nan_to_num(current)
        safe = {name: torch.nan_to_num(forcing[name]) for name in needed}
        k = self.log_k.exp().clamp(10, 5000)
        adv, diffusion = transport_terms(p, safe["water_u"], safe["water_v"], latitude, longitude, k)
        h = safe["mixed_layer_depth"][..., 1:-1, 1:-1].clamp_min(0.01)
        residual = (c-p)[..., 1:-1, 1:-1]/self.dt + adv - diffusion
        if kind == "heat":
            residual = residual - safe["net_heat_flux"][..., 1:-1, 1:-1]/(self.rho*self.cp*h)
        else:
            residual = residual + p[..., 1:-1, 1:-1]*safe["freshwater_flux"][..., 1:-1, 1:-1]/(self.rho*h)
        return masked_mse(residual, stencil_mask(valid), scale)
