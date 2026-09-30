"""Compatibility seam between ArviZ 0.x and ArviZ 1.x.

ArviZ 1.0 removed `arviz.InferenceData` and adopted `xarray.DataTree` as the
container for grouped inference output. The two lines are pinned as matched
pairs against PyMC in `pyproject.toml`, because PyMC 5 cannot import at all
under ArviZ 1 and PyMC 6 requires ArviZ 1:

| Python    | PyMC     | ArviZ      | Container              |
| --------- | -------- | ---------- | ---------------------- |
| 3.11      | 5.x      | 0.x        | `arviz.InferenceData`  |
| >=3.12    | 6.x      | 1.x        | `xarray.DataTree`      |

This module is the only place in Impulso allowed to branch on the installed
ArviZ version. Everything else works against the helpers exported here, so the
rest of the codebase stays version-agnostic.

Two deliberate boundaries:

- The public `idata` attribute is the *upstream-native* container for the
  installed stack, so users keep the full upstream API rather than an Impulso
  wrapper. Code that mutates `idata` must follow the installed upstream
  convention (`InferenceData.extend` on ArviZ 0, `DataTree.update` on ArviZ 1).
- Internally, every group is normalised to `xarray.Dataset` via
  `get_group_dataset`. `Prior`, `Sampler`, `IdentificationScheme`,
  `ErrorDistribution`, and `VolatilityProcess` implementations therefore keep
  receiving `xarray.Dataset` on both stacks, and third-party implementations of
  those protocols do not break.
"""

from importlib.metadata import version
from typing import TypeAlias

import arviz as az
import xarray as xr

__all__ = [
    "ARVIZ_V1",
    "InferenceDataLike",
    "get_group_dataset",
    "hdi_bounds",
    "make_idata",
]


def _detect_arviz_v1() -> bool:
    """Return True when the installed ArviZ is 1.0 or newer.

    Read the distribution metadata rather than probing `az.InferenceData`.
    ArviZ 1.x still exposes that name through a module-level `__getattr__`
    migration shim that emits `arviz.MigrationWarning`, so probing it would
    both mis-detect the version and pollute `import impulso` with warnings.
    """
    return int(version("arviz").split(".")[0]) >= 1


ARVIZ_V1: bool = _detect_arviz_v1()

if ARVIZ_V1:
    InferenceDataLike: TypeAlias = xr.DataTree
else:  # pragma: no cover - branch taken on the legacy stack only
    # Bound once here so no other module has to touch the name; on ArviZ 1 this
    # attribute only exists behind a MigrationWarning-emitting shim.
    InferenceDataLike: TypeAlias = az.InferenceData


def make_idata(**groups: xr.Dataset) -> InferenceDataLike:
    """Build the installed stack's InferenceData-schema container.

    Args:
        **groups: Mapping of schema group name (`posterior`,
            `posterior_predictive`, `observed_data`, ...) to the
            `xarray.Dataset` holding that group.

    Returns:
        An `xarray.DataTree` on ArviZ 1 or an `arviz.InferenceData` on
        ArviZ 0.

    Raises:
        TypeError: If any group is not an `xarray.Dataset`. `DataTree` nodes
            are rejected explicitly: callers must extract a Dataset first, so
            that group ownership and copy semantics stay obvious at the call
            site.
    """
    for name, group in groups.items():
        if not isinstance(group, xr.Dataset):
            raise TypeError(
                f"make_idata() group {name!r} must be an xarray.Dataset, got {type(group).__name__}. "
                "Use get_group_dataset() to extract a Dataset before rebuilding a container."
            )
    if ARVIZ_V1:
        return xr.DataTree.from_dict(dict(groups))
    return InferenceDataLike(**groups)


def get_group_dataset(idata: InferenceDataLike, name: str) -> xr.Dataset:
    """Return an InferenceData-schema group as an `xarray.Dataset`.

    On ArviZ 1 the group is a `DataTree` node, which is *not* an
    `xarray.Dataset`; passing it into Impulso's algorithms or protocol
    implementations would silently violate their `xr.Dataset` contract.

    `.to_dataset()` builds a new object on every call, so callers that reuse a
    group across a loop (or hand it to the memoised posterior helpers) must
    bind the result to a local once and pass that local through, rather than
    re-reading the group inside the loop.

    Args:
        idata: The container holding the group.
        name: Schema group name, e.g. `"posterior"`.

    Returns:
        The group as an `xarray.Dataset`, carrying the group's attrs.

    Raises:
        KeyError: If the container has no such group.
    """
    if ARVIZ_V1:
        if name not in idata.children:
            raise KeyError(f"InferenceData container has no group {name!r}. Available groups: {sorted(idata.children)}")
        return idata[name].to_dataset()
    if name not in idata.groups():
        raise KeyError(f"InferenceData container has no group {name!r}. Available groups: {sorted(idata.groups())}")
    return getattr(idata, name)


def hdi_bounds(da: xr.DataArray, prob: float) -> tuple[xr.DataArray, xr.DataArray]:
    """Compute HDI lower/upper bounds across the `chain` and `draw` dims.

    Normalises the two ArviZ APIs, which differ in both the probability
    keyword (`hdi_prob=` versus `prob=`) and the bound coordinate
    (`hdi` with `lower`/`higher` versus `ci_bound` with `lower`/`upper`).

    Only named `DataArray`s carrying both `chain` and `draw` are accepted, and
    those dims are always reduced explicitly. This is deliberate: given a bare
    3-D ndarray, ArviZ 0 reduces the leading sample axes while ArviZ 1 reduces
    only the trailing axis, so an ndarray-based call would silently return
    differently-shaped — and wrong — intervals across the two stacks.

    Args:
        da: Named posterior draws carrying `chain` and `draw` dims.
        prob: Probability mass for the interval, in (0, 1).

    Returns:
        A `(lower, upper)` pair of `DataArray`s with `chain`, `draw`, and the
        bound coordinate removed, and the remaining dims in their input order.

    Raises:
        TypeError: If `da` is not an `xarray.DataArray`.
        ValueError: If `prob` is outside (0, 1), or if `da` is unnamed or is
            missing `chain` or `draw`.
    """
    if not 0.0 < prob < 1.0:
        raise ValueError(f"hdi_bounds() prob must be in (0, 1), got {prob}.")
    if not isinstance(da, xr.DataArray):
        raise TypeError(f"hdi_bounds() expects an xarray.DataArray, got {type(da).__name__}.")
    if da.name is None:
        raise ValueError(
            "hdi_bounds() requires a named DataArray so the result can be identified; set `da.name` or use `da.rename(...)`."
        )
    missing = [dim for dim in ("chain", "draw") if dim not in da.dims]
    if missing:
        raise ValueError(
            f"hdi_bounds() requires the sampling dims 'chain' and 'draw'; {da.name!r} is missing {missing} (dims: {list(da.dims)})."
        )

    name = str(da.name)
    dataset = xr.Dataset({name: da})
    if ARVIZ_V1:
        result = az.hdi(dataset, prob=prob, dim=["chain", "draw"])[name]
        bound_dim, upper_label = "ci_bound", "upper"
    else:
        result = az.hdi(dataset, hdi_prob=prob)[name]
        bound_dim, upper_label = "hdi", "higher"

    lower = result.sel({bound_dim: "lower"}).drop_vars(bound_dim)
    upper = result.sel({bound_dim: upper_label}).drop_vars(bound_dim)
    return lower, upper
