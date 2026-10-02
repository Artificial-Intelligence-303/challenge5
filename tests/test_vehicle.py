"""Checks for Challenge 5.

Readable on purpose. If a check fails, read the test: it says in words what
your code was supposed to do.

Most checks use the vehicle's network. A few use a second, two-variable
network, the AI-writing detector from Week 7, so that a function written
to fit the vehicle alone does not pass.
"""

import math

from bayesnet import Network, Node, assignments
from vehicle import (
    NETWORK,
    OCCUPANTS_FIRST,
    build_network,
    joint_probability,
    query,
)

# Week 7's detector: 10% of essays are AI-written, a flag catches 90% of
# those and 5% of essays a person wrote.
AUTHOR = Node("author", ["ai", "human"], [], {(): {"ai": 0.1, "human": 0.9}})
FLAG = Node(
    "flag",
    ["yes", "no"],
    ["author"],
    {
        ("ai",): {"yes": 0.9, "no": 0.1},
        ("human",): {"yes": 0.05, "no": 0.95},
    },
)
DETECTOR = Network([AUTHOR, FLAG])


def close(first: float, second: float) -> bool:
    """Return whether two probabilities agree to six decimal places."""
    return math.isclose(first, second, abs_tol=1e-6)


def test_joint_is_one_entry_from_each_table_multiplied() -> None:
    """Six variables, six table entries, one product."""
    assignment = {
        "pedestrians": "present",
        "visibility": "good",
        "detected": "yes",
        "action": "others",
        "pedestrian_hurt": "no",
        "occupant_hurt": "no",
    }
    expected = 0.7 * 0.7 * 0.95 * 0.5 * 0.9 * 0.7
    assert close(joint_probability(NETWORK, assignment), expected)


def test_joint_reads_the_row_for_the_parents_values() -> None:
    """In poor visibility, `detected` is read from its poor visibility row."""
    assignment = {
        "pedestrians": "present",
        "visibility": "poor",
        "detected": "no",
        "action": "occupants",
        "pedestrian_hurt": "yes",
        "occupant_hurt": "no",
    }
    expected = 0.7 * 0.3 * 0.4 * 0.8 * 0.5 * 0.9
    assert close(joint_probability(NETWORK, assignment), expected)


def test_joint_is_zero_where_a_table_says_impossible() -> None:
    """Nobody on the road means no pedestrian is hurt."""
    assignment = {
        "pedestrians": "absent",
        "visibility": "good",
        "detected": "no",
        "action": "occupants",
        "pedestrian_hurt": "yes",
        "occupant_hurt": "no",
    }
    assert joint_probability(NETWORK, assignment) == 0


def test_joint_sums_to_one_over_every_assignment() -> None:
    """Every one of the 64 complete assignments, added up, is certain."""
    total = sum(
        joint_probability(NETWORK, assignment)
        for assignment in assignments(NETWORK, {})
    )
    assert close(total, 1.0)


def test_joint_works_on_a_second_network() -> None:
    """The detector: an AI-written essay that is flagged."""
    assignment = {"author": "ai", "flag": "yes"}
    assert close(joint_probability(DETECTOR, assignment), 0.1 * 0.9)


def test_query_gives_a_value_for_every_value_of_the_variable() -> None:
    """Asking about `action` answers for both actions."""
    distribution = query(NETWORK, "action", {"detected": "yes"})
    assert set(distribution) == {"occupants", "others"}


def test_query_with_no_evidence_gives_a_parentless_node_its_table() -> None:
    """With nothing observed, pedestrians are present 70% of the time."""
    distribution = query(NETWORK, "pedestrians", {})
    assert close(distribution["present"], 0.7)
    assert close(distribution["absent"], 0.3)


def test_query_sums_out_the_hidden_variables() -> None:
    """P(detected = yes) adds every route to a detection."""
    distribution = query(NETWORK, "detected", {})
    expected = 0.7 * (0.7 * 0.95 + 0.3 * 0.6) + 0.3 * (0.7 * 0.05 + 0.3 * 0.1)
    assert close(distribution["yes"], expected)


def test_query_answers_add_up_to_one() -> None:
    """A distribution, whatever the evidence: the step that uses alpha."""
    for variable, evidence in [
        ("pedestrians", {"action": "occupants"}),
        ("visibility", {"detected": "no"}),
        ("detected", {"pedestrian_hurt": "yes", "occupant_hurt": "no"}),
    ]:
        assert close(sum(query(NETWORK, variable, evidence).values()), 1.0)


def test_query_turns_a_flag_around_into_the_author() -> None:
    """Week 7's answer: a flagged essay is AI-written with probability 2/3."""
    distribution = query(DETECTOR, "author", {"flag": "yes"})
    assert close(distribution["ai"], 0.09 / 0.135)


def test_query_uses_two_pieces_of_evidence() -> None:
    """Nothing detected in poor visibility leaves pedestrians about even."""
    distribution = query(
        NETWORK, "pedestrians", {"detected": "no", "visibility": "poor"}
    )
    expected = 0.7 * 0.4 / (0.7 * 0.4 + 0.3 * 0.9)
    assert close(distribution["present"], expected)


def test_query_leaves_the_evidence_unchanged() -> None:
    """The caller's evidence dictionary is the same after the call."""
    evidence = {"detected": "no"}
    query(NETWORK, "pedestrians", evidence)
    assert evidence == {"detected": "no"}


def test_the_question_from_2023() -> None:
    """The car protected its occupants: were pedestrians there?"""
    distribution = query(NETWORK, "pedestrians", {"action": "occupants"})
    assert 0.62 < distribution["present"] < 0.621


def test_poor_visibility_makes_a_missed_detection_mean_less() -> None:
    """Nothing detected is weaker evidence when the sensors see poorly."""
    good = query(
        NETWORK, "pedestrians", {"detected": "no", "visibility": "good"}
    )
    either = query(NETWORK, "pedestrians", {"detected": "no"})
    poor = query(
        NETWORK, "pedestrians", {"detected": "no", "visibility": "poor"}
    )
    assert good["present"] < either["present"] < poor["present"]


def test_the_two_action_tables_trade_one_harm_for_another() -> None:
    """Protecting occupants always hurts fewer occupants, more pedestrians."""
    first = build_network(OCCUPANTS_FIRST)
    assert (
        query(first, "pedestrian_hurt", {})["yes"]
        > query(NETWORK, "pedestrian_hurt", {})["yes"]
    )
    assert (
        query(first, "occupant_hurt", {})["yes"]
        < query(NETWORK, "occupant_hurt", {})["yes"]
    )
