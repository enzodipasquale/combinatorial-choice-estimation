"""Convenience accessors for scenario factory builders."""

from __future__ import annotations

from . import greedy, linear_knapsack, plain_single_item, quadratic_knapsack, supermodular


class ScenarioLibrary:
    """Entry point for scenario builders."""

    @staticmethod
    def greedy():
        return greedy.build()

    @staticmethod
    def plain_single_item():
        return plain_single_item.build()

    @staticmethod
    def linear_knapsack():
        return linear_knapsack.build()

    @staticmethod
    def quadratic_knapsack():
        return quadratic_knapsack.build()

    @staticmethod
    def quadratic_supermodular():
        return supermodular.build()


__all__ = ["ScenarioLibrary"]


