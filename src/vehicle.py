"""An autonomous vehicle's Bayesian network, queried by enumeration.

**You implement two functions in this file**, `joint_probability` and
`query`. Everything else is provided. Add anything else you use from
`bayesnet` to the import below.

The network describes one moment: an accident the car cannot avoid. It
has six variables. Whether pedestrians are present, and how good the
visibility is, decide whether the car's sensors detect pedestrians. The
car acts on what it detects: it takes the action that protects its own
occupants, or the one that protects others. Who gets hurt depends on that
action, and on who was there.
"""

from bayesnet import Assignment, Network, Node

PEDESTRIANS = Node(
    "pedestrians",
    ["present", "absent"],
    [],
    {(): {"present": 0.7, "absent": 0.3}},
)

VISIBILITY = Node(
    "visibility",
    ["good", "poor"],
    [],
    {(): {"good": 0.7, "poor": 0.3}},
)

DETECTED = Node(
    "detected",
    ["yes", "no"],
    ["pedestrians", "visibility"],
    {
        ("present", "good"): {"yes": 0.95, "no": 0.05},
        ("present", "poor"): {"yes": 0.6, "no": 0.4},
        ("absent", "good"): {"yes": 0.05, "no": 0.95},
        ("absent", "poor"): {"yes": 0.1, "no": 0.9},
    },
)

# The action table from this course's 2023 version of this problem: half
# the time when there are pedestrians, 0.8 when there are none. The car can
# only act on what it detects, so here the table depends on `detected`.
ACTION = Node(
    "action",
    ["occupants", "others"],
    ["detected"],
    {
        ("yes",): {"occupants": 0.5, "others": 0.5},
        ("no",): {"occupants": 0.8, "others": 0.2},
    },
)

# A second action table, for comparison: the car protects its occupants
# whatever it detects.
OCCUPANTS_FIRST = Node(
    "action",
    ["occupants", "others"],
    ["detected"],
    {
        ("yes",): {"occupants": 1.0, "others": 0.0},
        ("no",): {"occupants": 1.0, "others": 0.0},
    },
)

PEDESTRIAN_HURT = Node(
    "pedestrian_hurt",
    ["yes", "no"],
    ["pedestrians", "action"],
    {
        ("present", "occupants"): {"yes": 0.5, "no": 0.5},
        ("present", "others"): {"yes": 0.1, "no": 0.9},
        ("absent", "occupants"): {"yes": 0.0, "no": 1.0},
        ("absent", "others"): {"yes": 0.0, "no": 1.0},
    },
)

OCCUPANT_HURT = Node(
    "occupant_hurt",
    ["yes", "no"],
    ["action"],
    {
        ("occupants",): {"yes": 0.1, "no": 0.9},
        ("others",): {"yes": 0.3, "no": 0.7},
    },
)


def build_network(action: Node) -> Network:
    """Return the vehicle's network with the given action table.

    Args:
        action: `ACTION` or `OCCUPANTS_FIRST`.

    Returns:
        The six variables, parents before children.
    """
    return Network(
        [
            PEDESTRIANS,
            VISIBILITY,
            DETECTED,
            action,
            PEDESTRIAN_HURT,
            OCCUPANT_HURT,
        ]
    )


NETWORK = build_network(ACTION)


def joint_probability(network: Network, assignment: Assignment) -> float:
    """Return the probability of one complete assignment.

    A Bayesian network's joint probability is one entry from each node's
    table, multiplied together: the row for the values its parents take in
    this assignment, and the column for the value the node itself takes.

    Args:
        network: The network the assignment is over.
        assignment: A value for every variable in the network.

    Returns:
        The probability that every variable takes the value given.
    """
    # TODO: implement this
    raise NotImplementedError

def query(
    network: Network, variable: str, evidence: Assignment
) -> dict[str, float]:
    """Return the distribution of `variable` given the evidence.

    Inference by enumeration. For each value of `variable`, add up the joint
    probability of every complete assignment that agrees with the evidence
    and gives `variable` that value. Then scale the sums so they add to 1.

    Args:
        network: The network to ask.
        variable: The name of the variable asked about.
        evidence: The values observed, never including `variable` itself.
            Not changed.

    Returns:
        A probability for every value of `variable`, summing to 1.
    """
    # TODO: implement this
    raise NotImplementedError
