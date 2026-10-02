# Challenge 5: An Autonomous Vehicle's Dilemma

|                    |                                             |
| :----------------- | :------------------------------------------ |
| `Tuesday 13 Oct`   | Released, in lab                            |
| `Tuesday 20 Oct`   | Due at 2:30pm, the start of lab             |
| `Tuesday 20 Oct`   | Verbal checks, during that same lab session |
| Points             | 3                                           |

An autonomous car faces an accident it cannot avoid. It takes either the
action that protects its own occupants or the one that protects others,
and it chooses based on what its sensors detect, which is not always
what is there. A Bayesian network describes that moment, and inference by
enumeration answers questions about it: given what we observed, how likely
is each thing we did not?

You write the two functions that answer the questions. The network and the
questions are provided.

## Course learning outcomes

1. **Outcome 3.** Design, implement, and assess an intelligent system. You
   implement exact inference for a Bayesian network and assess what the
   car's network concludes from each observation.
2. **Outcome 5.** Evaluate intelligent systems while considering their
   social, political, and ethical implications. You compare two action
   tables by who each one hurts, and argue who should choose between them.

## The network

Six variables. Each one's table, in `src/vehicle.py`, gives its
probabilities for every combination of its parents' values.

| Variable          | Values                | Parents                     |
| :---------------- | :-------------------- | :-------------------------- |
| `pedestrians`     | `present`, `absent`   | none                        |
| `visibility`      | `good`, `poor`        | none                        |
| `detected`        | `yes`, `no`           | `pedestrians`, `visibility` |
| `action`          | `occupants`, `others` | `detected`                  |
| `pedestrian_hurt` | `yes`, `no`           | `pedestrians`, `action`     |
| `occupant_hurt`   | `yes`, `no`           | `action`                    |

`action = occupants` is the action that protects the car's occupants.
`ACTION` is the action table from this course's 2023 version of this
problem: the car protects its occupants with probability 0.5 when it
detects pedestrians and 0.8 when it does not. `OCCUPANTS_FIRST` protects
them every time.

Every number in every table is invented for this challenge.

## What you implement

Two functions in `src/vehicle.py`. Everything else is provided.

### 1. `joint_probability(network, assignment)`

Return the probability of one complete assignment: one entry from each
node's table, multiplied together.

### 2. `query(network, variable, evidence)`

Return the distribution of one variable given the evidence, as a
dictionary from each of its values to a probability. Use inference by
enumeration: sum the joint over the hidden variables, then normalize.

`src/bayesnet.py` holds `Node`, `Network` and `assignments`. Read its
docstring first; the example at the top shows every piece you need.

## Running it

```
uv run python src/main.py              the questions, and both action tables
uv run python src/main.py tables       every conditional probability table
uv run python src/main.py ask pedestrians detected=no visibility=poor
uv run pytest                          run the tests
uv run ruff check src tests            check your style
uv run ruff format src tests           fix most style problems automatically
uv run gatorgrade                      run the checks the way they are graded
```

`ask` answers any question you type: the variable first, then what was
observed as `name=value`.

## Style

Same as last time: the [Google Python Style
Guide](https://google.github.io/styleguide/pyguide.html), checked by
`uv run ruff check src tests`.

## Evaluation

This challenge is worth **3 points**.

| Component                 | Value   |
| :------------------------ | :------ |
| Programming               | 1       |
| Written reflection        | 1       |
| Verbal check              | 1       |
| **Total**                 | **3**   |

### Programming, 1 point

Run by `gatorgrade`, awarded as the fraction of the **code** checks that
pass. These are:

- `joint_probability` multiplies the right entry from each table, and the
  joint sums to 1, on the car's network and on a second network.
- `query` sums out the hidden variables, uses every piece of evidence,
  returns probabilities that add up to 1, and leaves the evidence it was
  given unchanged.
- The car's network gives the answers it should to the questions in
  `main.py`.
- Style passes `ruff` and type annotations check out under `mypy`.
- No `TODO` markers are left in `src/vehicle.py`.

`gatorgrade` also runs two checks on `docs/summary.md`. Those belong to the
reflection below, not to this point.

Autograder results are preliminary. The final grade is determined by the
instructor.

### Written reflection, 1 point

Complete `docs/summary.md`. Two questions and the disclosure, **200 words
minimum across all of them**. You are graded on whether the reasoning is
correct and specific to what your run printed, not on length.

### Verbal check, 1 point

During lab you answer two or three questions about your own code, in the
context of this assignment. Nothing to prepare beyond understanding what
you wrote.

## AI use on this challenge

Full policy: **[AI in this course](https://areweagentsyet.com/ai/)**. AI
tools are allowed here, you disclose them, and you have to be able to
explain what you submit.

> ### The skill this time: check a number a second way
>
> A probability can be checked. Before you trust one, from a tool or from
> your own code, compute it a second way: by hand from two table rows, or
> by asking whether the answers add up to 1.

Concretely, for this assignment:

- **Write `joint_probability` and `query` yourself.** They are this week's
  idea.
- **Check one answer by hand.** P(pedestrians | detected = no, visibility =
  poor) needs two rows of the `detected` table and Bayes' rule. Compare it
  with `ask`.
- **A tool is useful** for the Python around the inference: how to copy a
  dictionary, or what `tuple(...)` builds from a list.

## How this assignment was built

You are asked below to disclose your AI use, so here is mine.

I decided the shape: the 2023 autonomous vehicle question rebuilt as a
network where the car acts on what it detects, two action tables to
compare, and two functions for you to write.

I used Claude to draft `bayesnet.py`, the tests and `main.py`, and to
compute every probability the tests expect. I ran the reference solution
and the unfinished starter against every check, and checked the checks
against wrong versions of both functions, not only against empty ones.

## Disclosing your AI use

At the end of `docs/summary.md` there is a disclosure section. Answer three
things, a sentence or two each:

1. **Which tool.** Name it. If you used more than one, name each.
2. **What you used it for.** Which part of the assignment.
3. **What you did with what it gave you.** Accepted it, edited it, checked
   it against something, or threw it out.

**If you did not use any tool, write that.** That is a complete and
perfectly good answer.

Disclosing costs you nothing. There is no version of this where naming a
tool loses you points.
