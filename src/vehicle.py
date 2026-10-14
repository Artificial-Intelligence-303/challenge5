"""What an autonomous car's sensors tell it, by Bayes' rule.

**You implement three functions in this file**, `joint`,
`evidence_probability` and `posterior`. Everything else is provided.

Every question here has the same shape. There are hypotheses about what is
ahead of the car, each with a prior probability. A sensor reports a value,
and each hypothesis gives that value a likelihood. Bayes' rule turns the
prior and the likelihoods into a posterior: how probable each hypothesis is
once the report is in.

A prior is a dictionary from each hypothesis to its probability:

    {"present": 0.3, "absent": 0.7}

A likelihood table is a dictionary from each hypothesis to the probability
of every value the sensor can report, given that hypothesis:

    {"present": {"yes": 0.85, "no": 0.15},
     "absent": {"yes": 0.10, "no": 0.90}}

Read it as P(camera = yes | present) = 0.85. Every number in every table is
invented for this challenge.
"""

Distribution = dict[str, float]
Likelihood = dict[str, Distribution]

# Whether a pedestrian is in the car's path, on two roads.
DOWNTOWN: Distribution = {"present": 0.3, "absent": 0.7}
HIGHWAY: Distribution = {"present": 0.01, "absent": 0.99}

# Each sensor reports yes or no: did it detect a pedestrian?
CAMERA: Likelihood = {
    "present": {"yes": 0.85, "no": 0.15},
    "absent": {"yes": 0.10, "no": 0.90},
}
LIDAR: Likelihood = {
    "present": {"yes": 0.90, "no": 0.10},
    "absent": {"yes": 0.05, "no": 0.95},
}

# Three hypotheses: what is ahead, and the label the camera gives it.
AHEAD: Distribution = {"pedestrian": 0.2, "cyclist": 0.1, "nothing": 0.7}
LABEL: Likelihood = {
    "pedestrian": {"person": 0.80, "bicycle": 0.05, "clear": 0.15},
    "cyclist": {"person": 0.35, "bicycle": 0.55, "clear": 0.10},
    "nothing": {"person": 0.05, "bicycle": 0.02, "clear": 0.93},
}

# The car faces an accident it cannot avoid. It takes either the action
# that protects its own occupants or the one that protects others. The
# action table from this course's 2023 version of this problem: the car
# protects its occupants half the time when a pedestrian is present, and
# 0.8 of the time when not.
ACTION: Likelihood = {
    "present": {"occupants": 0.5, "others": 0.5},
    "absent": {"occupants": 0.8, "others": 0.2},
}
# A second action table, for comparison: the car protects its occupants
# every time.
OCCUPANTS_FIRST: Likelihood = {
    "present": {"occupants": 1.0, "others": 0.0},
    "absent": {"occupants": 1.0, "others": 0.0},
}


def joint(
    prior: Distribution, likelihood: Likelihood
) -> dict[tuple[str, str], float]:
    """Return the joint probability of every hypothesis and every report.

    The product rule: P(h and e) = P(h) * P(e | h). This is the table of
    1,000 cases from class, divided by 1,000.

    Args:
        prior: The probability of each hypothesis.
        likelihood: For each hypothesis, the probability of each report.

    Returns:
        A dictionary from `(hypothesis, report)` to its joint probability,
        with an entry for every pair.
    """
    # TODO: implement this
    raise NotImplementedError


def evidence_probability(
    prior: Distribution, likelihood: Likelihood, observed: str
) -> float:
    """Return the probability that the sensor reports `observed`.

    The law of total probability: P(e) is the sum over the hypotheses of
    P(h) * P(e | h).

    Args:
        prior: The probability of each hypothesis.
        likelihood: For each hypothesis, the probability of each report.
        observed: The report.

    Returns:
        P(observed).
    """
    # TODO: implement this
    raise NotImplementedError


def posterior(
    prior: Distribution, likelihood: Likelihood, observed: str
) -> Distribution:
    """Return the probability of each hypothesis once `observed` is seen.

    Bayes' rule: P(h | e) = P(e | h) * P(h) / P(e), for every hypothesis.
    There may be two hypotheses or more.

    Args:
        prior: The probability of each hypothesis. Not changed.
        likelihood: For each hypothesis, the probability of each report.
            Not changed.
        observed: The report. Its probability is never 0 here.

    Returns:
        A probability for every hypothesis in `prior`, summing to 1.
    """
    # TODO: implement this
    raise NotImplementedError


def naive_bayes(
    prior: Distribution, likelihoods: list[Likelihood], observations: list[str]
) -> Distribution:
    """Return the posterior after several sensors' reports.

    Provided: a preview of naive Bayes, which comes back in Challenge 6.
    Each report updates the belief by Bayes' rule, and the posterior after
    one report is the prior for the next. That is only right if the sensors
    make their mistakes independently once the hypothesis is known, which
    is the assumption that makes naive Bayes naive.

    Args:
        prior: The probability of each hypothesis.
        likelihoods: One likelihood table per sensor.
        observations: What each sensor reported, in the same order.

    Returns:
        A probability for every hypothesis, summing to 1.
    """
    belief = prior
    for likelihood, observed in zip(likelihoods, observations, strict=True):
        belief = posterior(belief, likelihood, observed)
    return belief
