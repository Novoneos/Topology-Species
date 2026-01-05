from shapes import *

# List of topology classes to visualize.
# Each class represents a different 2D topology species.
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

# Iterate over all topology species
for ShapeClass in shape_classes:
    # Print the class name to the console for tracking
    print(ShapeClass)

    # Instantiate the topology with default parameters
    shape = ShapeClass()

    # Render and save the generated topology
    # The filename is automatically derived from the class name
    shape.draw_shape(savefig=True, name=ShapeClass.__name__)
