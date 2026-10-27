"""Cartesian geometry helpers: shapes → fluid/solid mask at cell centers."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal, Sequence, Union

import numpy as np

PaintMode = Literal["solid", "fluid"]


@dataclass
class Rectangle:
    x0: float
    y0: float
    x1: float
    y1: float
    mode: PaintMode = "solid"
    kind: str = "rectangle"

    def contains(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        xa, xb = sorted((self.x0, self.x1))
        ya, yb = sorted((self.y0, self.y1))
        return (x >= xa) & (x <= xb) & (y >= ya) & (y <= yb)


@dataclass
class Circle:
    cx: float
    cy: float
    radius: float
    mode: PaintMode = "solid"
    kind: str = "circle"

    def contains(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return (x - self.cx) ** 2 + (y - self.cy) ** 2 <= self.radius**2


@dataclass
class Polygon:
    points: list[tuple[float, float]]
    mode: PaintMode = "solid"
    kind: str = "polygon"

    def contains(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        # Ray-casting point-in-polygon on a flat array; reshape at the end.
        pts = self.points
        n = len(pts)
        if n < 3:
            return np.zeros_like(x, dtype=bool)
        xf = np.asarray(x, dtype=float).ravel()
        yf = np.asarray(y, dtype=float).ravel()
        inside = np.zeros(xf.shape, dtype=bool)
        for i in range(n):
            x1, y1 = pts[i]
            x2, y2 = pts[(i + 1) % n]
            cond = ((y1 > yf) != (y2 > yf)) & (
                xf < (x2 - x1) * (yf - y1) / (y2 - y1 + 1e-300) + x1
            )
            inside ^= cond
        return inside.reshape(np.asarray(x).shape)


Shape = Union[Rectangle, Circle, Polygon]


@dataclass
class GeometryScene:
    shapes: list = field(default_factory=list)

    def add(self, shape: Shape) -> None:
        self.shapes.append(shape)

    def clear(self) -> None:
        self.shapes.clear()

    def to_json(self) -> dict[str, Any]:
        return {"shapes": [asdict(s) for s in self.shapes]}

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> GeometryScene:
        scene = cls()
        for raw in data.get("shapes", []):
            kind = raw.get("kind", "")
            mode: PaintMode = raw.get("mode", "solid")
            if kind == "rectangle":
                scene.add(
                    Rectangle(
                        float(raw["x0"]),
                        float(raw["y0"]),
                        float(raw["x1"]),
                        float(raw["y1"]),
                        mode=mode,
                    )
                )
            elif kind == "circle":
                scene.add(
                    Circle(
                        float(raw["cx"]),
                        float(raw["cy"]),
                        float(raw["radius"]),
                        mode=mode,
                    )
                )
            elif kind == "polygon":
                pts = [(float(p[0]), float(p[1])) for p in raw["points"]]
                scene.add(Polygon(pts, mode=mode))
            else:
                raise ValueError(f"Unknown shape kind: {kind}")
        return scene


def cell_centers(
    nx: int,
    ny: int,
    x_min: float = 0.0,
    x_max: float = 1.0,
    y_min: float = 0.0,
    y_max: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return meshgrid of cell-center coordinates with shape (ny, nx)."""
    dx = (x_max - x_min) / float(nx)
    dy = (y_max - y_min) / float(ny)
    xs = x_min + (np.arange(nx, dtype=float) + 0.5) * dx
    ys = y_min + (np.arange(ny, dtype=float) + 0.5) * dy
    return np.meshgrid(xs, ys, indexing="xy")


def rasterize(
    shapes: Sequence[Shape],
    nx: int,
    ny: int,
    x_min: float = 0.0,
    x_max: float = 1.0,
    y_min: float = 0.0,
    y_max: float = 1.0,
    *,
    base_fluid: bool = True,
) -> np.ndarray:
    """
    Rasterize shapes onto an interior mask at cell centers.

    Returns uint8 array shape (ny, nx): 1 = fluid, 0 = solid.
    Shapes are applied in order; later shapes overwrite earlier paint.
    """
    mask = np.full((ny, nx), 1 if base_fluid else 0, dtype=np.uint8)
    if not shapes:
        return mask
    x, y = cell_centers(nx, ny, x_min, x_max, y_min, y_max)
    for shape in shapes:
        hit = shape.contains(x, y)
        value = np.uint8(0 if shape.mode == "solid" else 1)
        mask[hit] = value
    return mask


def rasterize_scene(
    scene: GeometryScene | dict[str, Any],
    nx: int,
    ny: int,
    x_min: float = 0.0,
    x_max: float = 1.0,
    y_min: float = 0.0,
    y_max: float = 1.0,
    *,
    base_fluid: bool = True,
) -> np.ndarray:
    if isinstance(scene, dict):
        scene = GeometryScene.from_json(scene)
    return rasterize(
        scene.shapes, nx, ny, x_min, x_max, y_min, y_max, base_fluid=base_fluid
    )
