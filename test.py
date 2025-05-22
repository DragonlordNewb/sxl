from sxlpy.geometry import*

coords = CoordinateSystem("t", "x", "y", "z")
f = Function("f")(*coords)
gmn = MetricTensor([[f, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, -1]], coords)
pprint(gmn.conn_mixed(0, 0, 1))
index = Index(gmn, (CO, 0), (CO, 0), (CONTRA, 1), (CO, 2))
print(repr(index))