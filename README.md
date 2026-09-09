# GenPark Bayesian Network Exact Variable Elimination Skill

Exact probabilistic inference for discrete Bayesian networks utilizing factor products, evidence reduction, and variable elimination.

Find out more at [GenPark](https://genpark.ai) and the [GenPark MCP Catalog](https://genpark.ai/mcp).

```mermaid
graph TD
    A[Factor Pool phi_1, phi_2, ... phi_k] --> B[Reduce Observed Evidence]
    B --> C[Select Variable to Eliminate X_i]
    C --> D[Multiply Factors Containing X_i]
    D --> E[Sum out X_i: sum_X_i phi_prod]
    E --> F{Remaining Variables?}
    F -->|Yes| C
    F -->|No: Target Left| G[Normalize to Posterior Distribution]
```

## Features
- General discrete multidimensional factor operations.
- Exact posterior computation without Monte Carlo sampling variance.
- Pure Python standard library.
