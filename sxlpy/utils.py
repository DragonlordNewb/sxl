from typing import Any
from typing import Iterable

def dissolve_metalist(l: list[list[...]]) -> Iterable[Any]:
    for x in r:
        if type(x) == list:
            for x in dissolve_metalist(x):
                yield x
        else:
            yield x

def all_index_combos(r: int, d: int) -> Iterable[list[int]]:
    if r == 1:
        for i in range(d):
            yield [i]
    else:
        for i in range(d):
            for x in all_index_combos(r - 1, d):
                yield [i] + x

def empty_value_block(r: int, d: int) -> list[list[...]]:
    if r == 1:
        return [None]*d
    else:
        return [empty_value_block(r - 1, d)]*d