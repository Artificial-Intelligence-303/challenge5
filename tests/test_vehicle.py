"""Checks for Challenge 5.

Readable on purpose. If a check fails, read the test: it says in words what
your code was supposed to do.

Most checks use the car's sensors. A few use the numbers from class: the
AI-writing detector, the spam filter from the worksheet and the email
folders, so that a function written to fit the car alone does not pass.
"""

import copy
import math

from vehicle import (
    ACTION,
    CAMERA,
    DOWNTOWN,
    HIGHWAY,
    LIDAR,
    OCCUPANTS_FIRST,
    evidence_probability,
    joint,
    naive_bayes,
    posterior,
)

# The detector from class: 10% of essays are AI-written, a flag catches 90%
# of those and 5% of essays a person wrote.
AUTHOR = {"ai": 0.1, "human": 0.9}
FLAG = {
    "ai": {"flagged": 0.9, "not": 0.1},
    "human": {"flagged": 0.05, "not": 0.95},
}

# The worksheet's spam filter: 20% of email is spam, the filter flags 95%
# of spam and 2% of real email.
EMAIL = {"spam": 0.2, "real": 0.8}
FILTER = {
    "spam": {"flagged": 0.95, "not": 0.05},
    "real": {"flagged": 0.02, "not": 0.98},
}

# Three folders, and the share of each folder's email that contains "free"
# or "meeting".
FOLDER = {"personal": 0.5, "promotions": 0.3, "spam": 0.2}
FREE = {
    "personal": {"yes": 0.05, "no": 0.95},
    "promotions": {"yes": 0.6, "no": 0.4},
    "spam": {"yes": 0.8, "no": 0.2},
}
MEETING = {
    "personal": {"yes": 0.3, "no": 0.7},
    "promotions": {"yes": 0.3, "no": 0.7},
    "spam": {"yes": 0.3, "no": 0.7},
}


def close(first: float, second: float, places: int = 6) -> bool:
    """Return whether two probabilities agree to the given decimal places."""
    return math.isclose(first, second, abs_tol=10**-places)


def test_joint_has_an_entry_for_every_hypothesis_and_report() -> None:
    """Two hypotheses times two reports is four cells, keyed (h, e)."""
    table = joint(DOWNTOWN, CAMERA)
    assert set(table) == {
        ("present", "yes"),
        ("present", "no"),
        ("absent", "yes"),
        ("absent", "no"),
    }


def test_joint_is_the_prior_times_the_likelihood() -> None:
    """P(present and yes) = P(present) * P(yes | present) = 0.3 * 0.85."""
    table = joint(DOWNTOWN, CAMERA)
    assert close(table[("present", "yes")], 0.3 * 0.85)
    assert close(table[("absent", "yes")], 0.7 * 0.10)
    assert close(table[("absent", "no")], 0.7 * 0.90)


def test_joint_sums_to_one() -> None:
    """Every cell of the table together is everything that can happen."""
    assert close(sum(joint(DOWNTOWN, CAMERA).values()), 1)
    assert close(sum(joint(FOLDER, FREE).values()), 1)


def test_joint_builds_the_worksheet_table() -> None:
    """For 1,000 emails: 190 and 10 spam, 16 and 784 real."""
    table = joint(EMAIL, FILTER)
    assert close(1000 * table[("spam", "flagged")], 190)
    assert close(1000 * table[("spam", "not")], 10)
    assert close(1000 * table[("real", "flagged")], 16)
    assert close(1000 * table[("real", "not")], 784)


def test_evidence_probability_adds_over_the_hypotheses() -> None:
    """P(camera = yes) downtown = 0.3 * 0.85 + 0.7 * 0.10 = 0.325."""
    assert close(evidence_probability(DOWNTOWN, CAMERA, "yes"), 0.325)


def test_evidence_probability_matches_the_worksheet() -> None:
    """P(flagged) = 0.95 * 0.20 + 0.02 * 0.80 = 0.206."""
    assert close(evidence_probability(EMAIL, FILTER, "flagged"), 0.206)


def test_evidence_probabilities_add_up_to_one() -> None:
    """The sensor reports yes or no, so the two probabilities add to 1."""
    total = sum(
        evidence_probability(HIGHWAY, LIDAR, report) for report in ("yes", "no")
    )
    assert close(total, 1)


def test_posterior_turns_a_flag_around_into_the_author() -> None:
    """A flagged essay is AI-written with probability 90/135 = 2/3."""
    assert close(posterior(AUTHOR, FLAG, "flagged")["ai"], 2 / 3)


def test_posterior_reads_the_report_that_was_observed() -> None:
    """An email the filter did not flag is spam with probability 10/794."""
    assert close(posterior(EMAIL, FILTER, "not")["spam"], 10 / 794)


def test_posterior_is_not_the_likelihood() -> None:
    """The camera says yes to 85% of pedestrians, but a yes is not 0.85.

    Downtown, P(present | yes) = 0.255 / 0.325.
    """
    belief = posterior(DOWNTOWN, CAMERA, "yes")
    assert close(belief["present"], 0.255 / 0.325)
    assert close(belief["absent"], 0.070 / 0.325)


def test_posterior_depends_on_the_prior() -> None:
    """The same camera on the highway: 0.0085 / 0.1075, about 0.079."""
    belief = posterior(HIGHWAY, CAMERA, "yes")
    assert close(belief["present"], 0.0085 / 0.1075)


def test_posterior_works_for_three_hypotheses() -> None:
    """An email with "free": personal 0.068, promotions 0.493, spam 0.438."""
    belief = posterior(FOLDER, FREE, "yes")
    assert set(belief) == set(FOLDER)
    assert close(belief["personal"], 0.025 / 0.365)
    assert close(belief["promotions"], 0.180 / 0.365)
    assert close(belief["spam"], 0.160 / 0.365)


def test_posterior_answers_add_up_to_one() -> None:
    """Whatever was observed, the hypotheses' probabilities add to 1."""
    for prior, likelihood, report in [
        (DOWNTOWN, CAMERA, "no"),
        (HIGHWAY, LIDAR, "yes"),
        (FOLDER, FREE, "no"),
        (AUTHOR, FLAG, "not"),
    ]:
        assert close(sum(posterior(prior, likelihood, report).values()), 1)


def test_evidence_every_hypothesis_agrees_on_leaves_the_prior() -> None:
    """The word "meeting" is in 30% of every folder, so it changes nothing."""
    belief = posterior(FOLDER, MEETING, "yes")
    for folder, probability in FOLDER.items():
        assert close(belief[folder], probability)


def test_posterior_leaves_its_arguments_unchanged() -> None:
    """The prior and the table are read, never written."""
    prior = copy.deepcopy(DOWNTOWN)
    likelihood = copy.deepcopy(CAMERA)
    posterior(prior, likelihood, "yes")
    assert prior == DOWNTOWN
    assert likelihood == CAMERA


def test_naive_bayes_combines_both_sensors() -> None:
    """Camera and lidar both say yes downtown: 0.2295 / 0.233, about 0.985.

    `naive_bayes` is provided. It is right only if `posterior` is.
    """
    belief = naive_bayes(DOWNTOWN, [CAMERA, LIDAR], ["yes", "yes"])
    assert close(belief["present"], 0.2295 / 0.2330)


def test_the_action_as_evidence() -> None:
    """The car protected its occupants: was a pedestrian present?

    Under ACTION, 0.15 / 0.71. Under OCCUPANTS_FIRST the car does the same
    thing either way, so the answer is the prior, 0.3.
    """
    action = posterior(DOWNTOWN, ACTION, "occupants")
    assert close(action["present"], 0.15 / 0.71)
    first = posterior(DOWNTOWN, OCCUPANTS_FIRST, "occupants")
    assert close(first["present"], 0.3)
