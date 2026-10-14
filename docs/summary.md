# Challenge 5 Reflection

Answer both questions. **200 words minimum across the document.**
You are graded on whether the reasoning is correct and specific to what
your own run printed. Write each answer in place of the placeholder under
its question.

Run `uv run python src/main.py` before you start writing. Everything these
questions ask about is in its output.

---

> **1. One detection, two roads.**
>
> The camera detects 85% of pedestrians. From your run, give
> P(present | camera = yes) downtown and on the highway. Explain, using the
> prior, why the same detection means so much less on the highway.
>
> Then give P(present | camera = yes, lidar = yes) on the highway.
> `naive_bayes` combines the two sensors as if they made their mistakes
> independently of each other. Name one situation on a road where the
> camera and the lidar would not.

TODO

---

> **2. What the action reveals, and who chooses it.**
>
> From your run, give P(action = occupants) and
> P(present | action = occupants) under `ACTION` and under
> `OCCUPANTS_FIRST`. Explain, using the likelihoods in each table, why the
> `OCCUPANTS_FIRST` answer equals the prior.
>
> Someone has to choose the numbers in the action table: the manufacturer,
> a regulator, the buyer of the car, or someone else. Say who should, and
> argue for it. Then name one thing these tables leave out that would
> change your answer.

TODO

---

> **AI tool disclosure**
>
> State what AI tools you used on this challenge, if any, and what they
> did. "None" is a complete answer if it is true. Be specific: "I used X to
> explain what a likelihood is" and "I used X to write `posterior`" are
> very different disclosures.

TODO
