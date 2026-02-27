"""
Prob and Stats Lab – Discrete Probability Distributions

Follow the instructions in each function carefully.
DO NOT change function names.
Use random_state=42 where required.
"""

import numpy as np
import math
from math import exp, factorial, comb

# =========================================================
# QUESTION 1 – Card Experiment
# =========================================================

def card_experiment():
    """
    STEP 1: Consider a standard 52-card deck.
            Assume 4 Aces.

    STEP 2: Compute analytically:
            - P(A)
            - P(B)
            - P(B | A)
            - P(A ∩ B)

    STEP 3: Check independence:
            P(A ∩ B) ?= P(A)P(B)

    STEP 4: Simulate 200,000 experiments
            WITHOUT replacement.
            Use random_state=42.

            Estimate:
            - empirical P(A)
            - empirical P(B | A)

    STEP 5: Compute absolute error BETWEEN:
            theoretical P(B | A)
            empirical P(B | A)

    RETURN:
        P_A,
        P_B,
        P_B_given_A,
        P_AB,
        empirical_P_A,
        empirical_P_B_given_A,
        absolute_error
    """

    # Theoretical Probabilities
    P_A = 4 / 52
    P_B_given_A = 3 / 51
    P_AB = P_A * P_B_given_A
    P_B = P_AB + (48 / 52) * (4 / 51)  # P(B) = P(B|A)*P(A) + P(B|not A)*P(not A)

    # Simulation (vectorized, without replacement)
    n_simulations = 200_000
    rng = np.random.RandomState(42)

    # Simulate first and second card indices
    first_card = rng.randint(0, 52, size=n_simulations)
    second_card = rng.randint(0, 51, size=n_simulations)
    second_card[second_card >= first_card] += 1  

    # Detect Aces (indices 0-3)
    first_is_ace = first_card < 4
    second_is_ace = second_card < 4

    # Count occurrences
    count_A = np.sum(first_is_ace)
    count_B_given_A = np.sum(first_is_ace & second_is_ace)

    # Empirical probabilities
    empirical_P_A = count_A / n_simulations
    empirical_P_B_given_A = count_B_given_A / count_A if count_A > 0 else 0

    # Absolute Error
    absolute_error = abs(empirical_P_B_given_A - P_B_given_A)

    return P_A, P_B, P_B_given_A, P_AB, empirical_P_A, empirical_P_B_given_A, absolute_error


# =========================================================
# QUESTION 2 – Bernoulli
# =========================================================

def bernoulli_lightbulb(p=0.05):
    """
    STEP 1: Define Bernoulli(p) PMF:
            p_X(x) = p^x (1-p)^(1-x)

    STEP 2: Compute theoretical:
            - P(X = 1)
            - P(X = 0)

    STEP 3: Simulate 100,000 bulbs
            using random_state=42.

    STEP 4: Compute empirical:
            - empirical P(X = 1)

    STEP 5: Compute absolute error BETWEEN:
            theoretical P(X = 1)
            empirical P(X = 1)

    RETURN:
        theoretical_P_X_1,
        theoretical_P_X_0,
        empirical_P_X_1,
        absolute_error
    """

    # Theoretical Probabilities
    theoretical_P_X_1 = p
    theoretical_P_X_0 = 1 - p

    # Simulation
    n_simulations = 100_000
    rng = np.random.RandomState(42)

    # Simulate bulbs: 1 = defective, 0 = working
    bulbs = rng.binomial(n=1, p=p, size=n_simulations)

    # Empirical P(X = 1)
    empirical_P_X_1 = np.mean(bulbs)

    # Absolute Error
    absolute_error = abs(empirical_P_X_1 - theoretical_P_X_1)

    return theoretical_P_X_1, theoretical_P_X_0, empirical_P_X_1, absolute_error


# =========================================================
# QUESTION 3 – Binomial
# =========================================================

def binomial_bulbs(n=10, p=0.05):
    """
    STEP 1: Define Binomial(n,p) PMF:
            P(X=k) = C(n,k)p^k(1-p)^(n-k)

    STEP 2: Compute theoretical:
            - P(X = 0)
            - P(X = 2)
            - P(X ≥ 1)

    STEP 3: Simulate 100,000 inspections
            using random_state=42.

    STEP 4: Compute empirical:
            - empirical P(X ≥ 1)

    STEP 5: Compute absolute error BETWEEN:
            theoretical P(X ≥ 1)
            empirical P(X ≥ 1)

    RETURN:
        theoretical_P_0,
        theoretical_P_2,
        theoretical_P_ge_1,
        empirical_P_ge_1,
        absolute_error
    """

    # Theoretical Probabilities
    theoretical_P_0 = (1 - p) ** n
    theoretical_P_2 = comb(n, 2) * (p ** 2) * ((1 - p) ** (n - 2))
    theoretical_P_ge_1 = 1 - theoretical_P_0

    # Simulation
    n_simulations = 100_000
    rng = np.random.RandomState(42)

    # Simulate n bulbs per experiment (1 = defective, 0 = working)
    bulbs = rng.binomial(n=1, p=p, size=(n_simulations, n))
    X = bulbs.sum(axis=1)  # number of defective bulbs per experiment

    # Empirical P(X ≥ 1)
    empirical_P_ge_1 = np.mean(X >= 1)

    # Absolute Error
    absolute_error = abs(empirical_P_ge_1 - theoretical_P_ge_1)

    return theoretical_P_0, theoretical_P_2, theoretical_P_ge_1, empirical_P_ge_1, absolute_error


# =========================================================
# QUESTION 4 – Geometric
# =========================================================

def geometric_die():
    """
    STEP 1: Let p = 1/6.

    STEP 2: Define Geometric PMF:
            P(X=k) = (5/6)^(k-1)*(1/6)

    STEP 3: Compute theoretical:
            - P(X = 1)
            - P(X = 3)
            - P(X > 4)

    STEP 4: Simulate 200,000 experiments
            using random_state=42.

    STEP 5: Compute empirical:
            - empirical P(X > 4)

    STEP 6: Compute absolute error BETWEEN:
            theoretical P(X > 4)
            empirical P(X > 4)

    RETURN:
        theoretical_P_1,
        theoretical_P_3,
        theoretical_P_gt_4,
        empirical_P_gt_4,
        absolute_error
    """

    # Parameters
    p = 1/6  # probability of rolling a 6

    # Theoretical Probabilities
    theoretical_P_1 = p
    theoretical_P_3 = ((1 - p) ** 2) * p
    theoretical_P_gt_4 = (1 - p) ** 4

    # Simulation
    n_simulations = 200_000
    rng = np.random.RandomState(42)

    # Simulate geometric distribution: number of rolls until first 6
    # np.random.geometric returns k = # of trials until first success
    rolls = rng.geometric(p=p, size=n_simulations)

    # Empirical P(X > 4)
    empirical_P_gt_4 = np.mean(rolls > 4)

    # Absolute Error
    absolute_error = abs(empirical_P_gt_4 - theoretical_P_gt_4)

    return theoretical_P_1, theoretical_P_3, theoretical_P_gt_4, empirical_P_gt_4, absolute_error


# =========================================================
# QUESTION 5 – Poisson
# =========================================================

def poisson_customers(lam=12):
    """
    STEP 1: Define Poisson PMF:
            P(X=k) = e^(-λ) λ^k / k!

    STEP 2: Compute theoretical:
            - P(X = 0)
            - P(X = 15)
            - P(X ≥ 18)

    STEP 3: Simulate 100,000 hours
            using random_state=42.

    STEP 4: Compute empirical:
            - empirical P(X ≥ 18)

    STEP 5: Compute absolute error BETWEEN:
            theoretical P(X ≥ 18)
            empirical P(X ≥ 18)

    RETURN:
        theoretical_P_0,
        theoretical_P_15,
        theoretical_P_ge_18,
        empirical_P_ge_18,
        absolute_error
    """

    # Theoretical Probabilities
    # Poisson PMF: P(X=k) = e^(-λ) λ^k / k!
    theoretical_P_0 = exp(-lam) * lam**0 / factorial(0)
    theoretical_P_15 = exp(-lam) * lam**15 / factorial(15)
    # P(X ≥ 18) = 1 - sum_{k=0}^{17} P(X=k)
    P_leq_17 = sum(exp(-lam) * lam**k / factorial(k) for k in range(18))
    theoretical_P_ge_18 = 1 - P_leq_17

    # Simulation
    n_simulations = 100_000
    rng = np.random.RandomState(42)

    # Simulate Poisson arrivals per hour
    arrivals = rng.poisson(lam=lam, size=n_simulations)

    # Empirical P(X ≥ 18)
    empirical_P_ge_18 = np.mean(arrivals >= 18)

    # Absolute Error
    absolute_error = abs(empirical_P_ge_18 - theoretical_P_ge_18)

    return theoretical_P_0, theoretical_P_15, theoretical_P_ge_18, empirical_P_ge_18, absolute_error
