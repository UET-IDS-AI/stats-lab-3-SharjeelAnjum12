[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/YGrPCab7)
# Prob and Stats Lab – Discrete Probability Distributions

Follow the instructions in each function carefully.
DO NOT change function names.
Use random_state=42 where required.

IMPORTANT:
Whenever absolute error is requested, compute:

    absolute_error = abs(empirical_value − theoretical_value)

--------------------------------------------------------

QUESTION 1 – Card Experiment (Without Replacement)

Two cards are drawn WITHOUT replacement from a 52-card deck.

Let:
A = first card is an Ace
B = second card is an Ace

Compute:
- P(A)
- P(B)
- P(B | A)
- P(A ∩ B)

Check independence:
P(A ∩ B) ?= P(A)P(B)

Simulate 200,000 experiments and estimate:
- empirical P(A)
- empirical P(B | A)

Compute absolute error between:
- theoretical P(B | A)
- empirical P(B | A)

--------------------------------------------------------

QUESTION 2 – Bernoulli (Light Bulb)

A light bulb is defective with probability p = 0.05.

Compute:
- Theoretical P(X = 1)
- Theoretical P(X = 0)
- Empirical P(X = 1)
- Absolute error

--------------------------------------------------------

QUESTION 3 – Binomial (10 Bulbs)

Inspect 10 bulbs independently.

Compute:
- Theoretical P(X = 0)
- Theoretical P(X = 2)
- Theoretical P(X ≥ 1)
- Empirical P(X ≥ 1)
- Absolute error

--------------------------------------------------------

QUESTION 4 – Geometric (Die Until 6)

Roll a fair die repeatedly until a 6 appears.

Compute:
- Theoretical P(X = 1)
- Theoretical P(X = 3)
- Theoretical P(X > 4)
- Empirical P(X > 4)
- Absolute error

--------------------------------------------------------

QUESTION 5 – Poisson (Customers per Hour)

Customers arrive at rate λ = 12 per hour.

Compute:
- Theoretical P(X = 0)
- Theoretical P(X = 15)
- Theoretical P(X ≥ 18)
- Empirical P(X ≥ 18)
- Absolute error
