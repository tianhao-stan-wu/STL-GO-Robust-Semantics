"""
Algebraic structures for STL-GO semantics.

Each algebra defines the operations used by the evaluator for Boolean/temporal
operators. New algebras just subclass Algebra and implement the abstract methods.
"""

from abc import ABC, abstractmethod


class Algebra(ABC):
    """Abstract base: defines the interface for any STL-GO algebra."""

    @abstractmethod
    def top(self) -> float:
        """Identity for ⊕ (or-like op), i.e. ⊤."""
        pass

    @abstractmethod
    def bot(self) -> float:
        """Identity for ⊗ (and-like op), i.e. ⊥."""
        pass

    @abstractmethod
    def and_op(self, a, b):
        """φ1 ∧ φ2"""
        pass

    @abstractmethod
    def or_op(self, a, b):
        """φ1 ∨ φ2"""
        pass

    @abstractmethod
    def neg_op(self, a):
        """¬φ"""
        pass


class MinMaxAlgebra(Algebra):
    """
    Min-max (standard STL robustness) algebra.
      and → min,  or → max,  neg → negation,  top → +∞,  bot → -∞
    """

    def top(self):
        return float('inf')

    def bot(self):
        return float('-inf')

    def and_op(self, a, b):
        return min(a, b)

    def or_op(self, a, b):
        return max(a, b)

    def neg_op(self, a):
        return -a