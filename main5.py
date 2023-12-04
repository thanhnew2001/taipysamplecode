"""
Taipy app to generate mandelbrot fractals
"""

from taipy import Gui

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

WINDOW_SIZE = 500

cm = plt.cm.get_cmap("viridis")


def generate_mandelbrot(
    center: int = WINDOW_SIZE / 2,
    dx_range: int = 1000,
    dx_start: float = -0.12,
    dy_range: float = 1000,
    dy_start: float = -0.82,
    iterations: int = 50,
    max_value: int = 200,
    i: int = 0,
) -> str:
    mat = np.zeros((WINDOW_SIZE, WINDOW_SIZE))
    for y in range(WINDOW_SIZE):
        for x in range(WINDOW_SIZE):
            dx = (x - center) / dx_range + dx_start
            dy = (y - center) / dy_range + dy_start
            a = dx
            b = dy
            for t in range(iterations):
                d = (a * a) - (b * b) + dx
                b = 2 * (a * b) + dy
                a = d
                h = d > max_value
                if h is True:
                    mat[x, y] = t

    colored_mat = cm(mat / mat.max())
    im = Image.fromarray((colored_mat * 255).astype(np.uint8))
    path = f"mandelbrot_{i}.png"
    im.save(path)

    return path


def generate(state):
    state.i = state.i + 1
    state.path = generate_mandelbrot(
        dx_start=-state.dx_start / 100,
        dy_start=(state.dy_start - 100) / 100,
        iterations=state.iterations,
        i=state.i,
    )


i = 0
dx_start = 11
dy_start = 17
iterations = 50

path = generate_mandelbrot(
    dx_start=-dx_start / 100,
    dy_start=(dy_start - 100) / 100,
)

page = """
# Mandelbrot Generator

<|layout|columns=35 65|
Display image from path
<|{path}|image|width=500px|height=500px|class_name=img|>

Iterations:<br />
Create a slider to select iterations
<|{iterations}|slider|min=10|max=50|continuous=False|on_change=generate|><br />
X Position:<br />
<|{dy_start}|slider|min=0|max=100|continuous=False|on_change=generate|><br />
Y Position:<br />

Slider dx_start
<|{dx_start}|slider|min=0|max=100|continuous=False|on_change=generate|><br />
|>
"""

Gui(page).run(title="Mandelbrot Generator")
