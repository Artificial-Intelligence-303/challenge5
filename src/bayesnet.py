"""Bayesian networks: variables, their tables, and assignments.

Provided. You do not change this file, but you use every part of it.

A **node** is one random variable: its name, the values it can take, the
names of its parents, and its **conditional probability table**. The
table has one row per combination of the parents' values, in the order
the parents are listed, and each row gives a probability for every value
of the node. A node with no parents has a single row, keyed by `()`.

An **assignment** gives a value to some or all of the variables, stored
as a dictionary from variable name to value.

    >>> rain = Node("rain", ["yes", "no"], [], {(): {"yes": 0.2, "no": 0.8}})
    >>> umbrella = Node(
    ...     "umbrella",
    ...     ["yes", "no"],
    ...     ["rain"],
    ...     {
    ...         ("yes",): {"yes": 0.9, "no": 0.1},
    ...         ("no",): {"yes": 0.3, "no": 0.7},
    ...     },
    ... )
    >>> network = Network([rain, umbrella])
    >>> umbrella.table[("no",)]["yes"]
    0.3
    >>> list(assignments(network, {"rain": "no"}))
    [{'rain': 'no', 'umbrella': 'yes'}, {'rain': 'no', 'umbrella': 'no'}]
"""

import itertools
import math
from collections.abc import Iterator
from dataclasses import dataclass

Assignment = dict[str, str]


@dataclass
class Node:
    """One random variable and its conditional probability table.

    Attributes:
        name: The variable's name, used as its key in an assignment.
        values: Every value the variable can take.
        parents: The names of the variables this one depends on directly.
        table: One row per combination of the parents' values, keyed by a
            tuple of those values in the order of `parents`. Each row maps
            every value of this variable to its probability.
    """

    name: str
    values: list[str]
    parents: list[str]
    table: dict[tuple[str, ...], dict[str, float]]

    def __post_init__(self) -> None:
        """Check that every row is a complete probability distribution."""
        for row, distribution in self.table.items():
            if len(row) != len(self.parents):
                raise ValueError(
                    f"{self.name}: row {row} does not match parents"
                    f" {self.parents}"
                )
            if set(distribution) != set(self.values):
                raise ValueError(
                    f"{self.name}: row {row} does not cover {self.values}"
                )
            if not math.isclose(sum(distribution.values()), 1.0):
                raise ValueError(f"{self.name}: row {row} does not sum to 1")


class Network:
    """A Bayesian network, its nodes listed with parents before children."""

    def __init__(self, nodes: list[Node]) -> None:
        """Store the nodes and check that their order is usable.

        Args:
            nodes: Every node in the network, each one after its parents.
        """
        seen: set[str] = set()
        for node in nodes:
            missing = [parent for parent in node.parents if parent not in seen]
            if missing:
                raise ValueError(
                    f"{node.name} is listed before its parents {missing}"
                )
            seen.add(node.name)
        self.nodes = nodes

    def node(self, name: str) -> Node:
        """Return the node with this name.

        Args:
            name: A variable's name.

        Returns:
            The node for that variable.
        """
        for node in self.nodes:
            if node.name == name:
                return node
        raise KeyError(name)

    def names(self) -> list[str]:
        """Return every variable's name, parents before children."""
        return [node.name for node in self.nodes]


def assignments(network: Network, fixed: Assignment) -> Iterator[Assignment]:
    """Yield every complete assignment that agrees with `fixed`.

    A variable named in `fixed` keeps that value. Every other variable takes
    each of its values in turn, so this yields one assignment per
    combination of the variables `fixed` leaves out.

    Args:
        network: The network whose variables to assign.
        fixed: Values that every yielded assignment keeps. Not changed.

    Yields:
        One dictionary per combination, giving every variable a value.
    """
    free = [node for node in network.nodes if node.name not in fixed]
    for values in itertools.product(*(node.values for node in free)):
        assignment = dict(fixed)
        for node, value in zip(free, values):
            assignment[node.name] = value
        yield {name: assignment[name] for name in network.names()}
