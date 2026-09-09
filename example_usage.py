"""
Demonstration of Bayesian Network Exact Variable Elimination Skill
"""

from client import BayesianNetwork

def main():
    print("=== Bayesian Network Exact Inference (Alarm Example) ===")
    bn = BayesianNetwork()

    # Burglary B: P(B=True) = 0.01, P(B=False) = 0.99
    bn.add_factor(["B"], {(True,): 0.01, (False,): 0.99})

    # Earthquake E: P(E=True) = 0.02, P(E=False) = 0.98
    bn.add_factor(["E"], {(True,): 0.02, (False,): 0.98})

    # Alarm A given B, E: P(A | B, E)
    # (B, E, A)
    alarm_table = {
        (True, True, True): 0.95, (True, True, False): 0.05,
        (True, False, True): 0.94, (True, False, False): 0.06,
        (False, True, True): 0.29, (False, True, False): 0.71,
        (False, False, True): 0.001, (False, False, False): 0.999,
    }
    bn.add_factor(["B", "E", "A"], alarm_table)

    print("Query: Probability of Burglary given Alarm sounded (A=True):")
    posterior = bn.query_marginal(target_var="B", evidence={"A": True}, elimination_order=["E"])
    print(f"P(B | A=True): {posterior}")

    assert posterior[True] > 0.01  # Posterior burglary probability rises given alarm
    print("\nBayesian Network Exact Variable Elimination Verification PASS!")

if __name__ == "__main__":
    main()
