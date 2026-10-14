# Challenge 5: An Autonomous Vehicle's Dilemma

|                    |                                             |
| :----------------- | :------------------------------------------ |
| `Thursday 15 Oct`  | Released                                    |
| `Thursday 22 Oct`  | Due at 11:00am, the start of class          |
| `Tuesday 27 Oct`   | Verbal checks, in lab                       |
| Points             | 3                                           |

An autonomous car has to decide what is ahead of it from what its sensors
report, and a sensor is sometimes wrong. A camera that detects 85% of
pedestrians does not make a detection 85% likely to be a pedestrian: that
depends on how often a pedestrian is there at all. Bayes' rule turns the
car's prior and its sensor's likelihoods into what it should believe.

You write the three functions that do it. The car's tables and the
questions are provided.

## Course learning outcomes

1. **Outcome 3.** Design, implement, and assess an intelligent system. You
   implement Bayes' rule for any number of hypotheses and assess what one
   sensor report, and then two, tell the car.
2. **Outcome 5.** Evaluate intelligent systems while considering their
   social, political, and ethical implications. You compare two action
   tables for an accident the car cannot avoid, and argue who should choose
   between them.

## The tables

Everything is in `src/vehicle.py`. A **prior** gives each hypothesis its
probability. A **likelihood table** gives, for each hypothesis, the
probability of every value a sensor can report.

| Name              | What it holds                                          |
| :---------------- | :----------------------------------------------------- |
| `DOWNTOWN`        | a prior: a pedestrian is present 30% of the time       |
| `HIGHWAY`         | a prior: a pedestrian is present 1% of the time        |
| `CAMERA`          | says yes to 85% of pedestrians and 10% of empty roads  |
| `LIDAR`           | says yes to 90% of pedestrians and 5% of empty roads   |
| `AHEAD`, `LABEL`  | pedestrian, cyclist or nothing, and the camera's label |
| `ACTION`          | how often the car protects its occupants               |
| `OCCUPANTS_FIRST` | a car that protects its occupants every time           |

`ACTION` is the action table: in an accident it cannot avoid, the car
protects its occupants with probability 0.5 when a pedestrian is present
and 0.8 when not.

Every number in every table is invented for this challenge.

## What you implement

Three functions in `src/vehicle.py`. Everything else is provided.

### 1. `joint(prior, likelihood)`

Return the joint probability of every hypothesis and every report, keyed
`(hypothesis, report)`. The product rule: P(h and e) = P(h) P(e | h).

### 2. `evidence_probability(prior, likelihood, observed)`

Return P(observed). The law of total probability: add P(h) P(e | h) over
the hypotheses.

### 3. `posterior(prior, likelihood, observed)`

Return P(h | observed) for every hypothesis, by Bayes' rule. It must work
for any number of hypotheses.

### Provided: `naive_bayes`

`naive_bayes` combines several sensors' reports by calling your
`posterior` once per report, so the posterior after one report is the prior
for the next. That is the **naive Bayes algorithm**: it assumes the sensors make their
mistakes independently once the hypothesis is known. Read it; it is three
lines, and it is right only if your `posterior` is.

## From the formula to code

Every function is one formula from class. Each formula is a product, a sum,
or a division, and each turns into a short loop over a dictionary.

| In the formula | In Python |
| :------------- | :-------- |
| P(h) | `prior[hypothesis]` |
| P(e \| h) | `likelihood[hypothesis][report]` |
| P(h and e) = P(h) P(e \| h) | `prior[hypothesis] * likelihood[hypothesis][report]` |
| P(e) = sum over h of P(h) P(e \| h) | a total that starts at 0, with one `+=` per hypothesis |
| P(h \| e) = P(e \| h) P(h) / P(e) | one division per hypothesis, by the same P(e) |

It is the table of 1,000 cases from class, without the 1,000. `joint`
fills in the cells; `evidence_probability` is a column total; `posterior`
divides each cell in a column by that column's total.

Each function's docstring works the same example by hand: the car is
downtown (prior `DOWNTOWN`) and the sensor is the lidar (likelihood `LIDAR`).
Write the functions in order and check each against its example before the
next, because `posterior` calls `evidence_probability`:

```
uv run python
>>> import sys; sys.path.insert(0, "src")
>>> from vehicle import DOWNTOWN, LIDAR, evidence_probability, posterior
>>> evidence_probability(DOWNTOWN, LIDAR, "yes")
0.305
>>> posterior(DOWNTOWN, LIDAR, "yes")
{'present': 0.885..., 'absent': 0.114...}
```

## Running it

```
uv run python src/main.py              every question, answered
uv run python src/main.py tables       every prior and likelihood table
uv run pytest                          run the tests
uv run ruff check src tests            check your style
uv run ruff format src tests           fix most style problems automatically
uv run gatorgrade                      run the checks the way they are graded
```

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

- `joint` multiplies each prior by its likelihood, has a cell for every
  pair, and sums to 1, on the car and on the spam filter from class.
- `evidence_probability` adds over the hypotheses, weighted by the prior.
- `posterior` turns a likelihood around into a posterior, reads the report
  that was observed, works for three hypotheses, adds up to 1, and leaves
  its arguments unchanged.
- `naive_bayes`, built on your `posterior`, and the action table give the
  answers they should.
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

- **Write `joint`, `evidence_probability` and `posterior` yourself.** They
  are this week's idea.
- **Check one answer by hand.** P(present | camera = yes) on the highway
  needs two cells of the camera's table and the prior. Work it out on paper
  and compare it with `main.py`.
- **A tool is useful** for the Python around the probability: how to add
  entries to a dictionary in a loop, or how to use a tuple as a key.

## How this assignment was built

You are asked below to disclose your AI use, so here is mine.

I came up with the autonomous vehicle question and decided its shape: the
car's sensors, the probability we worked by hand in class, two action tables
to compare, and three functions for you to write.

I used Claude to draft the tables, the tests and `main.py`, and to compute
every probability the tests expect. I ran the reference solution
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
