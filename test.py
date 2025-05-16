from sxlpy.geometry import*

coords = CoordinateSystem("t", "x", "y", "z")
gmn = MetricTensor([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, -1]], coords)
index = Index(gmn, (CO, 0), (CO, 0), (CONTRA, 1), (CO, 2))
print(repr(index))