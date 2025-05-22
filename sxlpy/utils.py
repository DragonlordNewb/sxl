from typing import Any
from typing import Iterable
from math import floor

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
		return [None for _ in range(d)]
	else:
		return [empty_value_block(r - 1, d)]*d

def symmetric_with_diag(d: int) -> Iterable[tuple[int, int]]:
	for i in range(d):
		for j in range(i, d):
			yield (i, j)

def symmetric_without_diag(d: int) -> Iterable[tuple[int, int]]:
	for i in range(d):
		for j in range(i + 1, d):
			yield (i, j)

class Progress:

	desc = ""
	total = 0
	current = 0

	def __init__(self, desc: str, total: int, length: int=50) -> None:
		self.total = total
		self.desc = desc
		self.length = length

	def _bar(self) -> str:
		filled_count = floor(self.current * self.length / self.total)
		unfilled_count = self.length - filled_count
		percent = str(self.current * 100 / self.total)[:5] + "%"
		return "[SXL] ... " + self.desc + " "*(50 - len(self.desc)) + "<" + ("#"*filled_count) + ("-"*unfilled_count) + "> " + percent

	def __enter__(self) -> None:
		self.current = -1
		self.done()
		return self
	
	def done(self) -> None:
		self.current += 1
		print(self._bar(), end="\r")

	def __exit__(self, *_):
		print("[SXL] !!! ")