from abc import ABC
from abc import abstractmethod
from typing import Iterable
from typing import Callable
from typing import Union
from sxlpy.geometry import Index
from sxlpy.geometry import Tensor

class TensorProperty(ABC):

    @abstractmethod
    def update(self, tensor: Tensor, *args, **kwargs) -> None:
        raise NotImplementedError("Subclasses must implement this method")

class TensorZeros(TensorProperty):

    def __init__(self, *variance) -> None:
        self.variance = variance

    @abstractmethod
    def condition(self, index: Index) -> bool:
        raise NotImplementedError("Subclasses must implement this method.")

    def update(self, tensor: Tensor) -> None:
        for index in Index.all(tensor.metric, self.variance):
            if self.condition(index):
                tensor.set(index, 0)

class TensorSymmetry(TensorProperty):

    def __init__(self, variance: list[str], target_metaindices: list[int]) -> None:
        self.targets = target_metaindices
        self.variance = variance

    @abstractmethod
    def symmetric_components(self, index: Index) -> Iterable[tuple[Index, int]]:
        raise NotImplementedError("Subclasses must implement this method.")

    def independent_components(self, tensor: Tensor) -> list[Index]:
        already = []
        for index in Index.all(tensor.metric, self.variance):
            if index not in already:
                yield index
                already.append(index)
                for sindex, _ in self.symmetric_components(index):
                    already.append(sindex)
    
    def independent_nonzero_components(self, tensor: Tensor, *zeros: TensorZeros) -> list[Index]:
        flag = False
        for iindex in self.independent_components(tensor):
            flag = False
            for zero in zeros:
                if zero.condition(index):
                    flag = True
                    break
            if not flag:
                yield iindex
                    
    def update(self, tensor: Tensor) -> None:
        for index in Index.all(tensor.metric, self.variance):
            val = tensor.get(index)
            if val is not None:
                for sindex, factor in self.symmetric_components(index):
                    tensor.set(sindex, val * factor)

class TensorProperties(TensorProperty):

    def __init__(self, *props: list[TensorProperty]) -> None:
        self.props = props

    def __iter__(self) -> Iterable[TensorProperty]:
        return iter(self.props)

    def update(self, tensor: Tensor) -> None:
        for prop in self:
            prop.update(tensor)