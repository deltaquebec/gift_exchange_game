#!/usr/bin/env python3
"""
Compute the number of distinct game trajectories

Note our working defintion here: a trajectory is a complete sequence of actions (steals, opens) and gift choices
from the initial state to the final state. It is a full game.
"""

from math import factorial, e


def A000522(k: int) -> int:
    """
    OEIS A000522: Number of ways to arrange k items with possible gaps
    
    A(k) = sum_{j=0}^{k} k!/j!
    
    counts action patterns in round k+1 (with k potential victims)
    """
    return sum(factorial(k) // factorial(j) for j in range(k + 1))


def trajectory_count(n: int, include_final_swap: bool = False) -> int:
    """
    Count distinct game trajectories for n players under standard rules.
    
    Standard rules: ell_round = 1, ell_total = infinity
    
    Formula: T(n) = n! × prod_{k=0}^{n-1} A000522(k)
        n: number of players
        include_final_swap: if True, multiply by n for final swap options
    """
    # product of A(k) for k = 1 to n, where A(k) = A000522(k-1)
    action_product = 1
    for k in range(n):
        action_product *= A000522(k)
    
    # multiply by n! for gift selection choices
    result = factorial(n) * action_product
    
    # include final swap (n choices: keep or swap with n-1 others) TODO: toggle this feature for game variations
    if include_final_swap:
        result *= n
    
    return result


def trajectory_count_finite_limit(n: int, ell_total: int) -> int:
    """
    Count trajectories with finite lifetime steal limit using DP
    
    Algorithm: track multisets of steal counts across rounds
    State: tuple of sorted steal counts (representing multiset)
        n: number of players
        ell_total: maximum times any gift can be stolen (>= 1)
    """
    from collections import defaultdict
    
    def multiset_to_tuple(counts: list) -> tuple:
        """Convert list of counts to canonical multiset representation"""
        return tuple(sorted(counts))
    
    def count_chains(counts: tuple, targets: tuple) -> dict:
        """
        Recursively enumerate all valid stealing chains
        """
        result = defaultdict(int)
        
        for i, g in enumerate(targets):
            # steal this gift: increment its count
            new_counts = list(counts)
            # find and increment one instance of g
            idx = new_counts.index(g)
            new_counts[idx] = g + 1
            new_counts_tuple = multiset_to_tuple(new_counts)
            
            # NOTE: chain could end here (victim opens)
            result[new_counts_tuple] += 1
            
            # NOTE: chain continues: remove g from available targets (chain-lock)
            remaining = targets[:i] + targets[i+1:]
            if remaining:
                sub_chains = count_chains(new_counts_tuple, remaining)
                for final_counts, ways in sub_chains.items():
                    result[final_counts] += ways
        
        return dict(result)
    
    # initialize: after round 1, one gift with 0 steals
    # observe: S maps multiset (as sorted tuple) -> number of ways to reach it
    S = {(0,): 1}
    
    for r in range(2, n + 1):
        S_next = defaultdict(int)
        
        for counts, ways in S.items():
            # open immediately (no stealing)
            new_counts = multiset_to_tuple(list(counts) + [0])
            S_next[new_counts] += ways
            
            # initiate a stealing chain
            # valid targets are gifts with count < ell_total
            targets = tuple(c for c in counts if c < ell_total)
            
            if targets:
                chain_outcomes = count_chains(counts, targets)
                for final_counts, chain_ways in chain_outcomes.items():
                    # chain ends with victim opening: add a 0
                    new_counts = multiset_to_tuple(list(final_counts) + [0])
                    S_next[new_counts] += ways * chain_ways
        
        S = dict(S_next)
    
    # total action patterns × n! (for gift identity choices)
    total_patterns = sum(S.values())
    return factorial(n) * total_patterns


def trajectory_count_one_and_done(n: int) -> int:
    """
    Count trajectories when each gift can only be stolen once ever (ell_total=1)
    
    NOTE: special case of trajectory_count_finite_limit but we keep it
    for completeness
    """
    return trajectory_count_finite_limit(n, ell_total=1)


def verify_small_cases():
    """Verify formulas against manual enumeration for small n"""
    
    print("verification of trajectory count formula")
    
    # n = 2
    # round 1: P1 opens (2 choices: G1 or G2)
    # round 2: P2 opens (1 choice) OR P2 steals from P1, P1 opens (1 choice)
    #          -> 2 action patterns × 1 gift choice = 2 ways
    # total: 2 × 2 = 4
    # TODO: check this!!
    print(f"n=2: formula gives {trajectory_count(2)}, expected 4")
    
    # n = 3
    # round 1: 1 action × 3 gifts = 3
    # round 2: 2 actions × 2 gifts = 4
    # round 3: 5 actions × 1 gift = 5
    # total: 3 × 4 × 5 = 60
    # TODO: check this!!
    print(f"n=3: formula gives {trajectory_count(3)}, expected 60")
    
    # Verify A(k) sequence
    print("\nA000522 sequence (recall: action patterns per round):")
    for k in range(1, 9):
        print(f"  A({k}) = A000522({k-1}) = {A000522(k-1)}")
    
    print("\ntrajectory counts T(n):")
    for n in range(1, 11):
        T = trajectory_count(n)
        T_swap = trajectory_count(n, include_final_swap=True)
        print(f"  n={n:2d}: T(n) = {T:>25,d}   with swap: {T_swap:>28,d}")
    
    print("\ncomparison: T(n) vs n! vs n^n")
    for n in range(1, 9):
        T = trajectory_count(n)
        fact = factorial(n)
        power = n ** n
        ratio = T / fact
        print(f"  n={n}: T(n)/n! = {ratio:,.1f}x")


def compare_total_limits(n: int = 6):
    """Compare trajectory counts for different ell_total values"""
    
    print(f"\neffect of lifetime steal limit (n={n})")
    
    T_inf = trajectory_count(n)
    T_three = trajectory_count_finite_limit(n, ell_total=3)
    T_one = trajectory_count_finite_limit(n, ell_total=1)
    
    print(f"  ell_total = infinity:  T({n}) = {T_inf:,d}")
    print(f"  ell_total = 3:  T({n}) = {T_three:,d}")
    print(f"  ell_total = 1:  T({n}) = {T_one:,d}")
    print(f"  ratio (infinity/1): {T_inf / T_one:.2f}x")
    print(f"  ratio (infinity/3): {T_inf / T_three:.2f}x")


def main():
    verify_small_cases()
    compare_total_limits(6)
    compare_total_limits(8)
    
    print("SUMMARY: trajectory Count Formula")
    print("""
For n players under standard gift exchange rules:

    T(n) = n! × ∏_{k=0}^{n-1} A(k)

where A(k) = Σ_{j=0}^{k} k!/j!  (OEIS A000522)

asymptotically: A(k) ~ k! × e, so T(n) grows faster than (n!)^2.

counts distinct game trajectories, where a trajectory specifies every action (steal target or open) and gift choice.
""")


if __name__ == "__main__":
    main()
