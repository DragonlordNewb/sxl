from typing import Any

def dissolve_metalist(l: list[list[...]]) -> list[Any]:
    r = []
    for x in r:
        if type(x) == list:
            for x in dissolve_metalist(x):
                r.append(x)
        else:
            r.append(x)
    return r