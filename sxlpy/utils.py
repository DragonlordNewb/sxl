from typing import Any
from typing import Iterable

def dissolve_metalist(l: list[list[...]]) -> list[Any]:
    r = []
    for x in r:
        if type(x) == list:
            for x in dissolve_metalist(x):
                r.append(x)
        else:
            r.append(x)
    return r

def all_index_combos(r: int, d: int) -> Iterable[list[int]]:
    if r == 1:
        for i in range(d):
            yield [i]
    else:
        for i in range(d):
            for x in all_index_combos(r - 1, d):
                yield [i] + x