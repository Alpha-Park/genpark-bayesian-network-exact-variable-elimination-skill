"""
MCP Server for Bayesian Network Exact Variable Elimination Skill
"""

import json
import sys
from client import BayesianNetwork

bn = BayesianNetwork()

def handle_call(name: str, args: dict) -> dict:
    if name == "add_factor":
        vars_list = args.get("variables", [])
        table = {tuple(k): v for k, v in args.get("table", {}).items()}
        bn.add_factor(vars_list, table)
        return {"status": "factor_added", "total_factors": len(bn.factors)}
    elif name == "query":
        t = args.get("target")
        ev = args.get("evidence", {})
        order = args.get("order", [])
        res = bn.query_marginal(t, ev, order)
        return {"target": t, "distribution": res}
    return {"error": f"Unknown tool: {name}"}

def main():
    for line in sys.stdin:
        if not line.strip():
            continue
        req = json.loads(line)
        res = handle_call(req.get("method"), req.get("params", {}))
        sys.stdout.write(json.dumps(res) + "\n")
        sys.stdout.flush()

if __name__ == "__main__":
    main()
