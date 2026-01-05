"""
Example script to visualize all available topology species.

Each topology species is implemented as a Python class with a common interface.
This script instantiates each topology once using default parameters and renders
the resulting 2D geometry. The output images allow users to quickly inspect and
compare the different topology families.
"""

from shapes import *

"""
List of topology species to visualize.

Each entry is a class representing a specific 2D topology family. All classes
share the same base interface and can be instantiated and rendered in an
identical way.
"""
shape_classes = [
    Cross,
    Rectangle,
    Ellipse,
    SplitRing,
    VShape,
    LShape,
    Bezier,
    BezierFlower,
    BezierStar,
    NeedleDrop,
    HeightmapSlice,
    DiffusionAggregation,
    WaveInterference,
    CellularAutomata,
]

"""
Instantiate and render each topology species.

For each class, a single instance is created using default parameters.
The resulting geometry is drawn and saved to disk using the class name
as the output filename.
"""
for ShapeClass in shape_classes:
    print(ShapeClass)
    shape = ShapeClass()
    shape.draw_shape(savefig=True, name=ShapeClass.__name__)
