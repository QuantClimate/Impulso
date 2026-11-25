"""Helpers for importing optional third-party dependencies.

Impulso keeps its core install lean. Features that need a heavier library
(for example the stationarity diagnostics, which need `statsmodels`) declare
an extra in `pyproject.toml` and import the dependency lazily through
`require`, so that a core-only install never pays the import cost and never
fails at `import impulso` time.
"""

import importlib
from types import ModuleType


def require(module: str, *, extra: str) -> ModuleType:
    """Import an optional dependency or raise an actionable error.

    Args:
        module: Importable module name, e.g. `"statsmodels"`.
        extra: Name of the Impulso extra that provides it, e.g.
            `"diagnostics"`.

    Returns:
        The imported module.

    Raises:
        ImportError: If the module is not installed. The message names the
            extra and gives copy-pasteable install commands.
    """
    try:
        return importlib.import_module(module)
    except ImportError as exc:
        raise ImportError(
            f"{module} is required for this feature. "
            f'Install it with: pip install "impulso[{extra}]" or: uv add "impulso[{extra}]"'
        ) from exc


def require_all(*modules: str, extra: str) -> tuple[ModuleType, ...]:
    """Import several optional dependencies, or raise one error naming every missing package.

    A failed import is reported under the package its `ImportError` names as
    missing, so a package that fails because another one it imports is
    missing (NumPyro without JAX) is reported as that other package, once.

    Args:
        *modules: Importable module names, e.g. `"jax", "numpyro"`.
        extra: Name of the Impulso extra that provides them, e.g.
            `"numpyro"`.

    Returns:
        The imported modules, in the order given.

    Raises:
        ImportError: If any module fails to import. The message names every
            missing package and the extra, and gives copy-pasteable install
            commands.
    """
    imported: list[ModuleType] = []
    missing: list[str] = []
    first_error: ImportError | None = None
    for module in modules:
        try:
            imported.append(importlib.import_module(module))
        except ImportError as exc:
            package = (exc.name or module).partition(".")[0]
            if package not in missing:
                missing.append(package)
            first_error = first_error or exc
    if not missing:
        return tuple(imported)
    names = missing[0] if len(missing) == 1 else f"{', '.join(missing[:-1])} and {missing[-1]}"
    verb, pronoun = ("is", "it") if len(missing) == 1 else ("are", "them")
    raise ImportError(
        f"{names} {verb} required for this feature. "
        f'Install {pronoun} with: pip install "impulso[{extra}]" or: uv add "impulso[{extra}]"'
    ) from first_error
