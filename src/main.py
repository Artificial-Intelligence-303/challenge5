"""Ask what the car's sensors tell it and print the answers.

Provided. Run it once `joint`, `evidence_probability` and `posterior` are
written.

    uv run python src/main.py           every question, answered
    uv run python src/main.py tables    every prior and likelihood table
"""

import sys

from vehicle import (
    ACTION,
    AHEAD,
    CAMERA,
    DOWNTOWN,
    HIGHWAY,
    LABEL,
    LIDAR,
    OCCUPANTS_FIRST,
    Distribution,
    Likelihood,
    evidence_probability,
    joint,
    naive_bayes,
    posterior,
)

ROADS = {"downtown": DOWNTOWN, "highway": HIGHWAY}


def row(label: str, cells: list[float]) -> str:
    """Lay out one labelled row of numbers.

    Args:
        label: The row's label.
        cells: The numbers, printed to three places.

    Returns:
        The row as text.
    """
    return f"  {label:<44}" + "".join(f"{cell:>11.3f}" for cell in cells)


def show_joint() -> None:
    """Print the camera's joint table downtown, with its column totals."""
    table = joint(DOWNTOWN, CAMERA)
    print("\nThe camera downtown, as a joint table\n")
    print(f"  {'':<44}{'yes':>11}{'no':>11}")
    for hypothesis in DOWNTOWN:
        cells = [table[(hypothesis, report)] for report in ("yes", "no")]
        print(row(f"pedestrian {hypothesis}", cells))
    totals = [
        evidence_probability(DOWNTOWN, CAMERA, report)
        for report in ("yes", "no")
    ]
    print(row("P(camera = ...)", totals))


def show_roads() -> None:
    """Print what each report means downtown and on the highway."""
    print(f"\nWhat a report means{'':<29}downtown    highway\n")
    print(row("P(present)", [prior["present"] for prior in ROADS.values()]))
    for name, sensor in (("camera", CAMERA), ("lidar", LIDAR)):
        for report in ("yes", "no"):
            cells = [
                posterior(prior, sensor, report)["present"]
                for prior in ROADS.values()
            ]
            print(row(f"P(present | {name} = {report})", cells))

    print("\nA preview of naive Bayes: both sensors at once\n")
    for camera, lidar in (("yes", "yes"), ("yes", "no"), ("no", "yes")):
        cells = [
            naive_bayes(prior, [CAMERA, LIDAR], [camera, lidar])["present"]
            for prior in ROADS.values()
        ]
        print(row(f"P(present | camera = {camera}, lidar = {lidar})", cells))


def show_label() -> None:
    """Print what is ahead given each label, three hypotheses at once."""
    print("\nWhat is ahead, from the camera's label\n")
    print(f"  {'':<44}" + "".join(f"{name:>11}" for name in AHEAD))
    print(row("prior", list(AHEAD.values())))
    for label in ("person", "bicycle", "clear"):
        belief = posterior(AHEAD, LABEL, label)
        print(row(f"label = {label}", list(belief.values())))


def show_action() -> None:
    """Print what the car's action reveals, under both action tables."""
    tables = [ACTION, OCCUPANTS_FIRST]
    print(f"\nThe action as evidence, downtown{'':<17}ACTION  OCC_FIRST\n")
    print(
        row(
            "P(action = occupants)",
            [evidence_probability(DOWNTOWN, t, "occupants") for t in tables],
        )
    )
    print(
        row(
            "P(present | action = occupants)",
            [posterior(DOWNTOWN, t, "occupants")["present"] for t in tables],
        )
    )


def show_tables() -> None:
    """Print every prior and every likelihood table."""
    priors: dict[str, Distribution] = {
        "DOWNTOWN": DOWNTOWN,
        "HIGHWAY": HIGHWAY,
        "AHEAD": AHEAD,
    }
    for name, prior in priors.items():
        cells = "   ".join(f"{h} {p:.2f}" for h, p in prior.items())
        print(f"\n{name}\n  {cells}")
    tables: dict[str, Likelihood] = {
        "CAMERA": CAMERA,
        "LIDAR": LIDAR,
        "LABEL": LABEL,
        "ACTION": ACTION,
        "OCCUPANTS_FIRST": OCCUPANTS_FIRST,
    }
    for name, table in tables.items():
        print(f"\n{name}")
        for hypothesis, reports in table.items():
            cells = "   ".join(f"{e} {p:.2f}" for e, p in reports.items())
            print(f"  given {hypothesis:<12}{cells}")


if __name__ == "__main__":
    if sys.argv[1:] == ["tables"]:
        show_tables()
    else:
        show_joint()
        show_roads()
        show_label()
        show_action()
