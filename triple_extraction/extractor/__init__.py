"""Extractor protocol — เผื่อ backend อื่น (GLiREL/GLiDRE) ในอนาคต"""

from typing import Protocol

Triple = dict  # {head, relation, tail, ...}


class Extractor(Protocol):
    def extract(self, text: str) -> list[Triple]: ...
