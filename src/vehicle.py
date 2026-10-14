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

From a formula to a dictionary lookup:

    P(h)        prior[hypothesis]              DOWNTOWN["present"] is 0.3
    P(e | h)    likelihood[hypothesis][report] LIDAR["present"]["yes"] is 0.9

Every docstring below works through the same example by hand. The car is
downtown, so the prior is `DOWNTOWN`, and the sensor is the lidar, so the
likelihood is `LIDAR`. Check each function against that example as you
write it, before moving on, because `posterior` calls
`evidence_probability`. For example, in `uv run python`:

    >>> import sys; sys.path.insert(0, "src")
    >>> from vehicle import DOWNTOWN, LIDAR, evidence_probability
    >>> evidence_probability(DOWNTOWN, LIDAR, "yes")
    0.305
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
# action table: the car protects its occupants half the time when a
# pedestrian is present, and 0.8 of the time when not.
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

    The product rule, for every hypothesis h and every report e:

        P(h and e) = P(h) * P(e | h)

    This is the table of 1,000 cases from class, divided by 1,000: one cell
    for each hypothesis and report.

    In code: start with an empty dictionary. Loop over the hypotheses in
    `likelihood`, and inside that over the reports in
    `likelihood[hypothesis]`. Store `prior[hypothesis]` times
    `likelihood[hypothesis][report]` under the key `(hypothesis, report)`.

    Worked example, downtown with the lidar:

        ("present", "yes")   0.3 * 0.90 = 0.270
        ("present", "no")    0.3 * 0.10 = 0.030
        ("absent", "yes")    0.7 * 0.05 = 0.035
        ("absent", "no")     0.7 * 0.95 = 0.665

    The four cells add up to 1.

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

    The law of total probability, with a sum over every hypothesis h:

        P(e) = sum of P(h) * P(e | h)

    It is the column total of the joint table: every way the report can
    happen, a real pedestrian or a false alarm.

    In code: start a total at 0, loop over the hypotheses in `prior`, and
    add `prior[hypothesis] * likelihood[hypothesis][observed]` each time.

    Worked example, downtown, when the lidar says yes:

        P(yes) = 0.3 * 0.90 + 0.7 * 0.05 = 0.270 + 0.035 = 0.305

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

    Bayes' rule, for every hypothesis h, where e is the report:

        P(h | e) = P(e | h) * P(h) / P(e)

    The top is one cell of the joint table; the bottom is that cell's
    column total, from `evidence_probability`. There may be two hypotheses
    or more: the same line works for each.

    In code: call `evidence_probability` once for the bottom. Then start a
    new, empty dictionary, loop over the hypotheses in `prior`, and store
    each one's top divided by the bottom. Do not change `prior`.

    Worked example, downtown, when the lidar says yes:

        P(present | yes) = 0.90 * 0.3 / 0.305 = 0.270 / 0.305 = 0.885
        P(absent | yes)  = 0.05 * 0.7 / 0.305 = 0.035 / 0.305 = 0.115

    The answers add up to 1. Note that 0.885 is not 0.90: P(present | yes)
    and P(yes | present) are different numbers.

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

    Provided: a preview of the naive Bayes algorithm, which comes back in
    class. Each report updates the belief by Bayes' rule, and the posterior
    after one report is the prior for the next. That is only right if the
    sensors make their mistakes independently once the hypothesis is known,
    which is the assumption that makes naive Bayes naive.

    Example: the camera and the lidar both say yes downtown. The camera's
    yes takes P(present) from 0.3 to 0.785; the lidar's yes then takes it
    from 0.785 to 0.985.

    Args:
        prior: The probability of each hypothesis.
        likelihoods: One likelihood table per sensor.
        observations: What each sensor reported, in the same order.

    Returns:
        A probability for every hypothesis, summing to 1.
    """
    belief = prior
    for sensor in range(len(likelihoods)):
        belief = posterior(belief, likelihoods[sensor], observations[sensor])
    return belief
