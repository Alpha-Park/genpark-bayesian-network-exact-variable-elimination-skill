"""
Bayesian Network Exact Variable Elimination Skill Client
Pure Python Standard Library implementation of Exact Inference in Bayesian Networks (Pearl / Koller & Friedman).
Represents discrete conditional probability tables as multidimensional factor matrices,
supporting factor reduction (conditioning on evidence), pointwise multiplication, and marginalization.
"""

from typing import List, Dict, Any, Tuple, Optional, Set
import itertools


class Factor:
    def __init__(self, variables: List[str], values: Dict[Tuple[Any, ...], float]):
        self.variables = variables
        self.values = values  # tuple of var values -> probability

    def reduce_evidence(self, evidence: Dict[str, Any]) -> "Factor":
        """Filter factor entries that match observed evidence."""
        new_values = {}
        for row, prob in self.values.items():
            match = True
            for var_idx, var_name in enumerate(self.variables):
                if var_name in evidence and evidence[var_name] != row[var_idx]:
                    match = False
                    break
            if match:
                new_values[row] = prob
        return Factor(list(self.variables), new_values)

    def multiply(self, other: "Factor") -> "Factor":
        """Pointwise multiplication of two factors over common and disjoint variables."""
        merged_vars = list(self.variables)
        for v in other.variables:
            if v not in merged_vars:
                merged_vars.append(v)

        # Build index lookups
        self_indices = [merged_vars.index(v) for v in self.variables]
        other_indices = [merged_vars.index(v) for v in other.variables]

        new_values = {}
        # Collect distinct values per merged variable
        domain_per_var: Dict[str, Set[Any]] = {v: set() for v in merged_vars}
        for row in self.values:
            for idx, val in enumerate(row):
                domain_per_var[self.variables[idx]].add(val)
        for row in other.values:
            for idx, val in enumerate(row):
                domain_per_var[other.variables[idx]].add(val)

        all_assignments = itertools.product(*[domain_per_var[v] for v in merged_vars])
        for assignment in all_assignments:
            self_key = tuple(assignment[i] for i in self_indices)
            other_key = tuple(assignment[i] for i in other_indices)
            p1 = self.values.get(self_key, 0.0)
            p2 = other.values.get(other_key, 0.0)
            if p1 > 0 and p2 > 0:
                new_values[assignment] = p1 * p2

        return Factor(merged_vars, new_values)

    def marginalize(self, var_to_eliminate: str) -> "Factor":
        """Sum out a variable from the factor."""
        if var_to_eliminate not in self.variables:
            return self

        elim_idx = self.variables.index(var_to_eliminate)
        remaining_vars = [v for v in self.variables if v != var_to_eliminate]
        new_values: Dict[Tuple[Any, ...], float] = {}

        for row, prob in self.values.items():
            rem_key = tuple(row[i] for i in range(len(row)) if i != elim_idx)
            new_values[rem_key] = new_values.get(rem_key, 0.0) + prob

        return Factor(remaining_vars, new_values)

    def normalize(self) -> "Factor":
        total = sum(self.values.values())
        if total > 0:
            norm_values = {k: v / total for k, v in self.values.items()}
            return Factor(list(self.variables), norm_values)
        return self


class BayesianNetwork:
    def __init__(self):
        self.factors: List[Factor] = []

    def add_factor(self, variables: List[str], table: Dict[Tuple[Any, ...], float]):
        self.factors.append(Factor(variables, table))

    def query_marginal(self, target_var: str, evidence: Dict[str, Any], elimination_order: List[str]) -> Dict[Any, float]:
        """Compute P(target_var | evidence) via Variable Elimination."""
        current_factors = [f.reduce_evidence(evidence) for f in self.factors]

        for var in elimination_order:
            if var == target_var or var in evidence:
                continue
            # Collect factors containing var
            relevant = [f for f in current_factors if var in f.variables]
            current_factors = [f for f in current_factors if var not in f.variables]

            if not relevant:
                continue

            # Multiply all relevant factors
            product_factor = relevant[0]
            for f in relevant[1:]:
                product_factor = product_factor.multiply(f)

            # Eliminate var
            summed_factor = product_factor.marginalize(var)
            current_factors.append(summed_factor)

        # Multiply remaining factors
        final_factor = current_factors[0]
        for f in current_factors[1:]:
            final_factor = final_factor.multiply(f)

        # Marginalize any leftover variables except target
        for v in list(final_factor.variables):
            if v != target_var:
                final_factor = final_factor.marginalize(v)

        final_factor = final_factor.normalize()
        return {k[0]: round(v, 4) for k, v in final_factor.values.items()}
