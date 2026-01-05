from shapes import *

# List all shape classes you want to visualize
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

for ShapeClass in shape_classes:
    print(ShapeClass)
    shape = ShapeClass()
    shape.draw_shape(savefig=True, name=ShapeClass.__name__)