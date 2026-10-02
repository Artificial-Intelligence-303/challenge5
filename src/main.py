"""Ask the vehicle's network questions and print the answers.

Provided. Run it once `joint_probability` and `query` are written.

    uv run python src/main.py           the questions, and both action tables
    uv run python src/main.py tables    every conditional probability table
    uv run python src/main.py ask pedestrians detected=no visibility=poor
                                        one question of your own
"""

import sys

from bayesnet import Assignment, Network
from vehicle import NETWORK, OCCUPANTS_FIRST, build_network, query

QUESTIONS: list[tuple[str, Assignment]] = [
    ("pedestrians", {}),
    ("pedestrians", {"action": "occupants"}),
    ("pedestrians", {"detected": "yes"}),
    ("pedestrians", {"detected": "no"}),
    ("pedestrians", {"detected": "no", "visibility": "good"}),
    ("pedestrians", {"detected": "no", "visibility": "poor"}),
    ("detected", {"pedestrian_hurt": "yes"}),
]


def describe(variable: str, evidence: Assignment) -> str:
    """Write a question as text, such as `pedestrians | detected = no`.

    Args:
        variable: The variable asked about.
        evidence: The values observed.

    Returns:
        The question, with the evidence after a bar.
    """
    if not evidence:
        return f"{variable}, nothing observed"
    given = ", ".join(f"{name} = {value}" for name, value in evidence.items())
    return f"{variable} | {given}"


def answer(network: Network, variable: str, evidence: Assignment) -> str:
    """Ask one question and lay out every value's probability.

    Args:
        network: The network to ask.
        variable: The variable asked about.
        evidence: The values observed.

    Returns:
        Each value followed by its probability, to three places.
    """
    distribution = query(network, variable, evidence)
    return "   ".join(
        f"{value} {probability:.3f}"
        for value, probability in distribution.items()
    )


def show_questions() -> None:
    """Print every question's answer, then the two action tables compared."""
    print("\nThe car's network, answered by enumeration\n")
    for variable, evidence in QUESTIONS:
        line = describe(variable, evidence)
        print(f"  {line:<52}{answer(NETWORK, variable, evidence)}")

    tables = [NETWORK, build_network(OCCUPANTS_FIRST)]
    print(f"\nTwo action tables{'':<36}ACTION   OCCUPANTS_FIRST\n")
    rows = [
        ("P(pedestrian_hurt = yes)", "pedestrian_hurt", {}, "yes"),
        ("P(occupant_hurt = yes)", "occupant_hurt", {}, "yes"),
        (
            "P(pedestrians = present | action = occupants)",
            "pedestrians",
            {"action": "occupants"},
            "present",
        ),
    ]
    for label, variable, evidence, value in rows:
        cells = [
            f"{query(network, variable, evidence)[value]:.3f}"
            for network in tables
        ]
        print(f"  {label:<51}{cells[0]:>6}{cells[1]:>18}")


def show_tables() -> None:
    """Print every node's conditional probability table."""
    for node in NETWORK.nodes:
        parents = ", ".join(node.parents) if node.parents else "no parents"
        print(f"\n{node.name}  ({parents})")
        for row, distribution in node.table.items():
            given = ", ".join(row) if row else ""
            cells = "   ".join(
                f"{value} {probability:.2f}"
                for value, probability in distribution.items()
            )
            print(f"  {given:<22}{cells}")


def ask(arguments: list[str]) -> None:
    """Answer one question typed on the command line.

    Args:
        arguments: The variable's name, then any number of `name=value`.
    """
    variable, *observed = arguments
    evidence = dict(item.split("=", 1) for item in observed)
    if variable in evidence:
        print(f"\n{variable} is observed, so there is nothing to ask.")
        return
    print(f"\n  {describe(variable, evidence)}")
    print(f"  {answer(NETWORK, variable, evidence)}")


if __name__ == "__main__":
    if sys.argv[1:] == ["tables"]:
        show_tables()
    elif sys.argv[1:2] == ["ask"] and len(sys.argv) > 2:
        ask(sys.argv[2:])
    else:
        show_questions()
