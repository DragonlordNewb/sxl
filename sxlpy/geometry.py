"""
I had a semi-working library before, but it was not generalizable enough; it
was impossible to work with a tensor I hadn't hard-coded, and I couldn't use
most sets of indices so I couldn't do any of the really cool stuff.

I have added compatibility for tensors of any rank and dimension along with
any and all indices, at the cost of memory. Good luck.
"""

from sympy import Symbol
from sympy import symbols
from sympy import Function
from sympy import Matrix
from sympy import Expr
from sympy import pprint
from typing import Iterable

E0 = lambda v, i: f"[E0] Bad index formation (variances {v}, indices {i})"
E1 = "[E1] Bad variance"
E2 = "[E2] Cannot raise/lower an index of the wrong variance"
E3A = "[E3A] Bad index rank when getting"
E3B = "[E3B] Bad index rank while setting"
E4A = "[E4A] Tensor incomplete/corrupted while getting"
E4B = "[E4B] Tensor corrupted while setting"
E5 = "[E5] Cannot trace with indices of different variance (music first)"
E6 = "[E6] Cannot expand non-dummy index"

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
            if v not in (CO, CONTRA) or i < -1 or i >= len(self.metric):
                # Dummy indices of -1 are allowed
                raise IndexError(E0(self.variances, self.indices))

        self.rank = len(self.indices)

    def __len__(self) -> int:
        return len(self.metric)
    
    def find_metaindex(self) -> int:
        return self.variance_to_metaindex(self.variances)

    def get_dummy_metaindices(self) -> int:
        r = []
        for mi, i in enumerate(self.indices):
            if i == -1:
                r.append(mi)
    
    def copy(self) -> "Index":
        return Index(*self._init)

    def substitute_index(self, metaindex: int, new_value: int) -> None:
        self.indices[metaindex] = new_value

    def substitute_variance(self, metaindex: int, new_value: str) -> None:
        self.variances[metaindex] = new_value

    def contracted(self, metaindex: int) -> list["Index"]:
        idxs = []
        idx = self.copy()
        for i in range(len(self)):
            idx.substitute_index(metaindex, i)
            idxs.append(idx.copy())
        return idxs

    def expanded(self, *over_dummies: list[int]) -> list["Index"]:
        for i in over_dummies:
            if self.indices[i] != -1:
                raise IndexError(E6)

        r = []

        if len(over_dummies) == 1:
            idx = self.copy()
            for i in range(len(self)):
                idx.substitute(over_dummies[0], i)
                r.append(idx.copy())
        else:
            idx = self.copy()
            for i in range(len(self)):
                idx.substitute(over_dummies[0], i)
                for nidx in idx.expand(over_dummies[1:]):
                    r.append(nidx)
        
        return r

    def trace_sum(self, mi1: int, mi2: int) -> list[tuple[Expr, "Index"]]:
        idxs = []
        idx = self.copy()
        for i in range(len(self)):
            idx.substitute_index(mi1, i)
            idx.substitute_index(mi2, i)
            idxs.append(idx.copy())
        if self.vaiances[mi1] == self.variances[mi2] == CO:
            return zip(idxs, [self.metric.uu(i, i) for i in range(len(self))])
        elif self.variaces[mi1] == self.variances[mi2] == CONTRA:
            return zip(idxs, [self.metric.dd(i, i) for i in range(len(self))])
        else:
            return zip(idxs, [1]*len(self))

    def raise_sum(self, metaindex: int) -> tuple["Index", list[tuple[Expr, "Index"]]]:
        if self.variances[metaindex] == CONTRA:
            raise IndexError(E2)
        r = self.indices[metaindex]
        nidx = self.copy()
        nidx.substitute_variance(metaindex, CONTRA)
        return nidx, zip(self.contracted(metaindex), [self.metric.uu(r, i) for i in range(len(self))])

    def lower_sum(self, metaindex):
        if self.variances[metaindex] == CO:
            raise IndexError(E2)
        r = self.indices[metaindex]
        nidx = self.copy()
        nidx.substitute_variance(metaindex, CO)
        return nidx, zip(self.contracted(metaindex), [self.metric.dd(r, i) for i in range(len(self))])

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

    @classmethod
    def all(cls, m: MetricTensor, *variance) -> list["Index"]:
        

class Tensor:

    rank: int
    metric: MetricTensor
    values: list

    def __init__(self, rank: int) -> None:
        self.values = [None for _ in range(pow(2, rank) - 1)]

    def __len__(self) -> int:
        return len(self.metric)

    def get(self, index: Index) -> Expr:
        if -1 in index.indices:


        if index.rank != self.rank:
            raise IndexError(E3A)
        
        mi = index.find_metaindex()
        tsr = values[mi]
        for i in index.indices:
            if tsr is None:
                raise IndexError(E4A)
            tsr = tsr[i]
        return tsr # Should be it!

    def set(self, index: Index, val: Expr) -> None:
        if index.rank != self.rank:
            raise IndexError(E3B)

        mi = index.find_metaindex()
        tsr = values[mi]
        for i in index.indices[:-1]:
            if tsr is None:
                raise IndexError(E4B)
            tsr = tsr[i]

        tsr[indices[-1]] = val

    def raise_index(self, index: Index, mi: int) -> Expr:
        """
        Given the tensor with that variance, raise the index located at the
        metaindex. For example, on a rank-3 tensor T,

            T.raise_index(Index(m, (CO, 0), (CO, 1), (CO, 2)), 0)

        yields T^0_{12} from T_{012}.
        """

        nidx, s = index.raise_sum(mi)
        val = sum([self.get(idx) * gmn for idx, gmn in s])
        self.set(nidx, val)
        return val

    def lower_index(self, index: Index, mi: int) -> Expr:
        """
        Same as raise_index, but lowers instead.
        """

        nidx, s = index.lower_sum(mi)
        val = sum([self.get(idx) * gmn for idx, gmn in s])
        self.set(nidx, val)
        return val

    def trace(self, index: Index, mi1: int, mi2: int) -> Expr:
        """
        Compute the trace of two given indices. For example, on a rank-4
        tensor (e.g. the Riemann tensor),

            T.trace(Index.co(m, -1, 0, -1, 1), 0, 2)

        runs a trace over the first and third indices. (if this were the
        Riemann tensor, it would give R_{01}). Works on indices of any
        variance.
        """

        return sum([self.get(idx) * gmn for idx, gmn in index.trace_sum(mi1, mi2)])