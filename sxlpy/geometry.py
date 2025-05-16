from sympy import Symbol
from sympy import symbols
from sympy import Function
from sympy import Matrix
from sympy import Expr
from sympy import pprint
from typing import Iterable

E0 = lambda v, i: f"[E0] Bad index formation (variances {v}, indices {i})"
E1 = "[E1] Can\'t contract two indices of the same variance (raise first)"

COVARIANT = CO = "d"
CONTRAVARIANT = CONTRA = "u"

def variance_to_metaindex(variance: tuple[str]) -> int:
    """
    Convert a variance identifier (e.g. the (CO, CONTRA, CO) tuple) into an int
    which refers to the index at which values of that version of the tensor are
    stored internally (see Tensor class).

    The variance string is represented as a binary string, which thus maps every
    possible variance to a number from 0 to 2^n - 1. Then a list of those 2^n
    different tensors are stored, and this function is used to index them.
    """

    result = 0
    for index_number, index_type in enumerate(variance):
        if index_type == CONTRA:
            result += pow(2, index_number)
        elif index_type == CO:
            pass
        else:
            raise IndexError(E1)
    return result

class CoordinateSystem:

    def __init__(self, *labels: list[str]) -> None:
        self.labels = labels
        self.symbols = symbols(labels)

    def __iter__(self) -> Iterable[Symbol]:
        return iter(self.symbols)

    def len(self) -> int:
        return len(self.labels)

class MetricTensor:

    def __init__(self, values: list[list[Expr]], cs: CoordinateSystem) -> None:
        self.mat = Matrix(values)
        self.mt_dd = values
        self.mt_uu = self.mat.inv.tolist()

    def dd(self, i, j):
        return self.mt_dd[i][j]

    def uu(self, i, j):
        return self.mt_uu[i][j]

class Index:

    def __init__(self, m: MetricTensor, *indices: list[tuple[str, int]]) -> None:
        self._init = (m, *indices)
        self.metric = m
        self.variances = self.indices = []
        for v, i in indices:
            self.variances.append(v)
            self.indices.append(i)

        for v, i in self:
            if v not in (CO, CONTRA) or i < 0 or i >= len(self.metric):
                raise IndexError(E0(self.variances, self.indices))

    def __len__(self) -> int:
        return len(self.metric)
    
    def find_metaindex(self) -> int:
        return self.variance_to_metaindex(self.variances)
    
    def copy(self) -> "Index":
        return Index(*self._init)

    def substitute_index(self, metaindex: int, new_value: int) -> None:
        self.indices[metaindex] = new_value

    def substitute_variance(self, metaindex: int, new_value: str) -> None:
        self.variances[metaindex] = new_value

    def single_contracted(self, metaindex: int) -> list["Index"]:
        idxs = []
        for i in range(len(self)):
            idx = self.copy()
            idx.substitute_index(metaindex, i)
            idxs.append(idx)
        return idxs

    def double_contracted(self, mi1: int, mi2: int) -> list["Index"]:
        if self.variances[mi1] == self.variances[mi2]:
            raise IndexError(E1)
        
        idxs = []
        for i in range(len(self)):
            idx = self.copy()
            idx.substitute_index(mi1, i)
            idx.substitute_index(mi2, i)
            idxs.append(idx)
        return idxs

    def raise_sum(self, metaindex: int) -> list[tuple[Expr, "Index"]]:
        r = self.indices[metaindex]
        return zip(self.single_contracted(metaindex), [self.metric.uu(r, i) for i in range(len(self))])

    def lower_sum(self, metaindex):
        r = self.indices[metaindex]
        return zip(self.single_contracted(metaindex), [self.metric.dd(r, i) for i in range(len(self))])

    @classmethod
    def co(cls, m: MetricTensor, *indices) -> "Index":
        return cls(m, *[(CO, i) for i in indices])
    
    @classmethod
    def contra(cls, m: MetricTensor, *indices) -> "Index":
        return cls(m, *[(CONTRA, i) for i in indices])
    
    @classmethod
    def mixed(cls, m: MetricTensor, *indices) -> "Index":
        j = [(CONTRA, indices[0])]
        for k in indices[1:]:
            j.append((CO, k))
        return cls(m, *j) 

class Tensor:

    rank: int
    metric: MetricTensor
    values: list

    def __init__(self, rank: int) -> None:
        self.values = [None for _ in range(pow(2, rank) - 1)]

    def get(self, variance: tuple[str], *indices: list[int]) -> Expr:
        if len(indices) != self.rank:
            raise IndexError(E2(self.rank, len(indices)))
        tn = self.values[variance_to_metaindex(variance)]
        
        ...

        return tn

    def set(self, variance: tuple[str], val: Expr, *indices: list[int]) -> None:
        if len(indices) != self.rank:
            raise IndexError(E4(self.rank, len(indices)))
        tn = self.values[variance_to_metaindex(variance)]
        
        ...

        tn[indices[-1]] = val

    def raise_index(self, mi: int, v: tuple[str], *indices: list[int],) -> Expr:
        """
        Given the tensor with that variance, raise the index located at the
        metaindex. For example, on a rank-3 tensor T,

            T.raise_index(0, (CO, CO, CO), 1, 2, 3)

        yields T^1_{23} from T_{123}.
        """

