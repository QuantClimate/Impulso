"""Log-density terms whose gradient a PyMC model sets to zero with a gradient override.

A term that is constant wherever it is finite, such as the `latent_stationarity`
Potential, has gradient zero. When PyTensor cannot differentiate the graph that
computes such a term (the non-symmetric `eig`), the graph is wrapped in an
`OpFromGraph` whose gradient is overridden with zeros (`zero_gradient_op`).

PyTensor's JAX conversion inlines the `OpFromGraph` and drops the override, so
compiled JAX code would differentiate the graph after all. The code that builds
such a term therefore also declares it (`declare_zero_gradient`), and
`CompiledNumPyroBlock.from_pymc_model` adds every declared term to the
log-density with a zero gradient (`has_zero_gradient`). The declaration is a
flag on the term's PyTensor variable, so it travels with the variable PyMC
keeps in `model.potentials` or `model.basic_RVs`.
"""

import inspect
from typing import Any

_ZERO_GRADIENT_FLAG = "impulso_zero_gradient"
"""Name of the attribute of a PyTensor variable's `tag` that declares a zero gradient."""


def zero_gradient_op(inputs: list[Any], outputs: list[Any], *, name: str) -> Any:
    """An `OpFromGraph` that computes `outputs` from `inputs` and whose gradient is overridden with zeros.

    Args:
        inputs: Symbolic PyTensor inputs of the graph.
        outputs: Symbolic PyTensor outputs computed from `inputs`.
        name: Name of the `OpFromGraph`.

    Returns:
        The `OpFromGraph`. Applied to variables, it gives `outputs` with
        gradient zero with respect to every input.
    """
    import pytensor.tensor as pt
    from pytensor.compile.builders import OpFromGraph

    # PyTensor calls the override with the inputs, the outputs and the
    # gradients of the outputs; only the inputs' shapes are needed.
    def zero_gradients(op_inputs: list[Any], *_: Any) -> list[Any]:
        return [pt.zeros_like(op_input) for op_input in op_inputs]

    # Newer PyTensor renamed `lop_overrides` to `pullback`.
    override = "pullback" if "pullback" in inspect.signature(OpFromGraph.__init__).parameters else "lop_overrides"
    return OpFromGraph(inputs, outputs, name=name, **{override: zero_gradients})


def declare_zero_gradient(term: Any) -> Any:
    """Declare that `term`, a log-density term of a PyMC model, has gradient zero through a gradient override.

    Call it where the term is built, on the variable PyMC registers (the
    return value of `pm.Potential`, or a random variable).

    Args:
        term: The PyTensor variable of the term.

    Returns:
        `term`, declared.
    """
    setattr(term.tag, _ZERO_GRADIENT_FLAG, True)
    return term


def has_zero_gradient(term: Any) -> bool:
    """Whether `declare_zero_gradient` declared `term`.

    Args:
        term: The PyTensor variable of a log-density term.

    Returns:
        `True` if the term was declared to have gradient zero.
    """
    return getattr(term.tag, _ZERO_GRADIENT_FLAG, False)
