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


def row(label: str, cells: list[float]) -> str:
    """Lay out one labelled row of numbers.

    Args:
        label: The row's label.
        cells: The numbers, printed to three places.

    Returns:
        The row as text.
    """
    line = f"  {label:<44}"
    for cell in cells:
        line += f"{cell:>11.3f}"
    return line


def show_joint() -> None:
    """Print the camera's joint table downtown, with its column totals."""
    table = joint(DOWNTOWN, CAMERA)
    print("\nThe camera downtown, as a joint table\n")
    print(f"  {'':<44}{'yes':>11}{'no':>11}")
    for hypothesis in DOWNTOWN:
        cells = [table[(hypothesis, "yes")], table[(hypothesis, "no")]]
        print(row(f"pedestrian {hypothesis}", cells))
    yes = evidence_probability(DOWNTOWN, CAMERA, "yes")
    no = evidence_probability(DOWNTOWN, CAMERA, "no")
    print(row("P(camera = ...)", [yes, no]))


def show_roads() -> None:
    """Print what each report means downtown and on the highway."""
    print(f"\nWhat a report means{'':<29}downtown    highway\n")
    print(row("P(present)", [DOWNTOWN["present"], HIGHWAY["present"]]))
    for name, sensor in [("camera", CAMERA), ("lidar", LIDAR)]:
        for report in ["yes", "no"]:
            cells = []
            for prior in [DOWNTOWN, HIGHWAY]:
                cells.append(posterior(prior, sensor, report)["present"])
            print(row(f"P(present | {name} = {report})", cells))

    print("\nA preview of the naive Bayes algorithm: both sensors at once\n")
    for camera, lidar in [("yes", "yes"), ("yes", "no"), ("no", "yes")]:
        cells = []
        for prior in [DOWNTOWN, HIGHWAY]:
            belief = naive_bayes(prior, [CAMERA, LIDAR], [camera, lidar])
            cells.append(belief["present"])
        print(row(f"P(present | camera = {camera}, lidar = {lidar})", cells))


def show_label() -> None:
    """Print what is ahead given each label, three hypotheses at once."""
    print("\nWhat is ahead, from the camera's label\n")
    header = f"  {'':<44}"
    for name in AHEAD:
        header += f"{name:>11}"
    print(header)
    print(row("prior", list(AHEAD.values())))
    for label in ("person", "bicycle", "clear"):
        belief = posterior(AHEAD, LABEL, label)
        print(row(f"label = {label}", list(belief.values())))


def show_action() -> None:
    """Print what the car's action reveals, under both action tables."""
    print(f"\nThe action as evidence, downtown{'':<17}ACTION  OCC_FIRST\n")
    chances = []
    beliefs = []
    for table in [ACTION, OCCUPANTS_FIRST]:
        chances.append(evidence_probability(DOWNTOWN, table, "occupants"))
        beliefs.append(posterior(DOWNTOWN, table, "occupants")["present"])
    print(row("P(action = occupants)", chances))
    print(row("P(present | action = occupants)", beliefs))


def show_tables() -> None:
    """Print every prior and every likelihood table."""
    priors: dict[str, Distribution] = {
        "DOWNTOWN": DOWNTOWN,
        "HIGHWAY": HIGHWAY,
        "AHEAD": AHEAD,
    }
    for name, prior in priors.items():
        line = ""
        for hypothesis, probability in prior.items():
            line += f"{hypothesis} {probability:.2f}   "
        print(f"\n{name}\n  {line}")
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
            line = ""
            for report, probability in reports.items():
                line += f"{report} {probability:.2f}   "
            print(f"  given {hypothesis:<12}{line}")


if __name__ == "__main__":
    if sys.argv[1:] == ["tables"]:
        show_tables()
    else:
        show_joint()
        show_roads()
        show_label()
        show_action()
