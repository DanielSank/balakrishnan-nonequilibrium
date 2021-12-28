"""
This module computes and animates the solutions to exercise 8.4.1 in the book
Elements of Nonequilibrium Statistical Mechanics by Balakrishnan.

"""

from typing import overload, Union
import numpy as np

import attr
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


PI = np.pi
TWOPI = 2 * PI


@attr.dataclass
class Parameters:
    """Parameters for the simulation.

    Attributes:
        x_0: The starting position of the delta function source.
        d: The diffusion constant.
        a: The half-width of the domain. Boundary conditions are enforced at
            +a and -a.
        n_max: The maximum spatial mode index.
    """
    x_0: float
    d: float
    a: float
    n_max: int
    duration_ms: int
    times: np.ndarray


@overload
def space_mode(n: int, x: float, a: float) -> float:
    ...


@overload
def space_mode(n: int, x: np.ndarray, a: float) -> np.ndarray:
    ...


def space_mode(
        n: int,
        x: Union[float, np.ndarray],
        a: float,
) -> Union[float, np.ndarray]:
    """A single spatial mode of the problem

    Args:
        n: mode index
        x: position
        a: half-width of the domain

    Returns:
        The nth space mode amplitude at position(s) x.
    """
    return np.cos(((PI * x) / (2 * a) + PI / 2) * n)


def pdf_reflecting(
        x: np.ndarray,
        t: float,
        parameters: Parameters,
) -> np.ndarray:
    """The probability density function with reflecting boundaries.

    We express the PDF as a sum over spatial wave numbers, including up to
    parameters.n_max terms.

    Args:
        x: Spatial axis. We compute the PDF for each value in this array.
        t: Value of time at which to compute the pdf.
        parameters: Parameters of the simulation.

    Returns:
        pdf evaluated at positions x and time t.
    """
    a = parameters.a
    n_max = parameters.n_max
    x_0 = parameters.x_0
    d = parameters.d

    k = np.linspace(1, n_max, n_max + 1)
    x_grid, k_grid = np.meshgrid(x, k, indexing='ij')
    return (1 / (2 * a)) + (1 / a) * np.sum(
            space_mode(k_grid, x_grid, a) * space_mode(k_grid, x_0, a) * np.exp(
                -d * (PI * k_grid / (2 * a))**2 * t
            ),
            axis=1,
    )


class Diffusion:
    def __init__(
            self,
            pdf_function,
            ax: plt.Axes,
            x: np.ndarray,
            parameters: Parameters,
) -> None:
        self.pdf_function = pdf_function
        self.ax = ax
        self.x = x
        self.parameters=parameters
        self.line, = ax.plot([], [], 'k-')

    def __call__(self, t: float):
        y = self.pdf_function(
                x=self.x,
                t=t,
                parameters=self.parameters,
        )
        self.line.set_data(self.x, y)
        self.ax.set_title(f"D={self.parameters.d:.2f}, x0={self.parameters.x_0:.2f}, {t=:.3f}")
        return self.line


DEFAULT_PARAMETERS = Parameters(
        x_0=0.125,
        d=0.5,
        a=0.5,
        n_max=200,
        duration_ms=100000,
        times=np.linspace(0, 1, 1001),
)


def make_animation(parameters: Parameters):
    fig, ax = plt.subplots()
    ax.grid()
    ax.set_xlim(-parameters.a, parameters.a)
    ax.set_ylim(0, 8)
    ax.set_ylabel("PDF")
    ax.set_xlabel("Position")
    diff = Diffusion(
            pdf_function=pdf_reflecting,
            ax=ax,
            x=np.linspace(-parameters.a, parameters.a, 51),
            parameters=parameters,
    )
    _anim = FuncAnimation(
            fig,
            diff,
            parameters.times,
            interval=parameters.duration_ms / len(parameters.times),
    )
    plt.show()


if __name__ == "__main__":
    make_animation(parameters=DEFAULT_PARAMETERS)
