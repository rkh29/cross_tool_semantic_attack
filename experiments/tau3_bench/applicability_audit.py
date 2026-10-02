#!/usr/bin/env python3
"""Check whether the original ContractShift unit/scale attack fits τ³-bench retail.

This audit is intentionally read-only. It inspects the pinned benchmark source and
does not change the benchmark tools or policies.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import subprocess
from pathlib import Path

EXPECTED_COMMIT = "5bfa7e37b36656b37dc6d022156be6563c1007f3"
TOOL_SOURCE = Path("src/tau2/domains/retail/tools.py")
REQUIRED_SOURCE_METHOD = "get_order_details"
REQUIRED_TARGET_METHOD = "return_delivered_order_items"
REQUIRED_AMOUNT_ARGUMENT = "amount_yuan"


def method_arguments(source: str, class_name: str, method_name: str) -> list[str]:
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for member in node.body:
                if isinstance(member, (ast.FunctionDef, ast.AsyncFunctionDef)) and member.name == method_name:
                    return [argument.arg for argument in member.args.args if argument.arg != "self"]
    raise ValueError(f"Could not find {class_name}.{method_name}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tau3-root",
        type=Path,
        default=Path(os.environ.get("TAU3_BENCH_ROOT", "../tau3-bench-upstream")),
        help="Path to the pinned sierra-research/tau2-bench checkout",
    )
    args = parser.parse_args()
    # Keep junction/symlink spellings intact; Path.resolve() canonicalizes a
    # Windows junction to its target and can reintroduce non-ASCII CLI paths.
    root = args.tau3_root.absolute()
    source_path = root / TOOL_SOURCE
    if not source_path.is_file():
        parser.error(f"Retail tools source not found: {source_path}")

    commit = subprocess.run(
        ["git", "-c", "safe.directory=*", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    source = source_path.read_text(encoding="utf-8")
    read_args = method_arguments(source, "RetailTools", REQUIRED_SOURCE_METHOD)
    return_args = method_arguments(source, "RetailTools", REQUIRED_TARGET_METHOD)
    amount_supported = REQUIRED_AMOUNT_ARGUMENT in return_args

    result = {
        "audit": "contractshift_unit_scale_applicability",
        "benchmark": "sierra-research/tau2-bench (τ³-bench)",
        "benchmark_commit": commit,
        "expected_commit": EXPECTED_COMMIT,
        "pinned_source": commit == EXPECTED_COMMIT,
        "source_tool": {
            "name": REQUIRED_SOURCE_METHOD,
            "arguments": read_args,
        },
        "refund_tool": {
            "name": REQUIRED_TARGET_METHOD,
            "arguments": return_args,
            "accepts_refund_amount": amount_supported,
        },
        "original_attack_requirement": {
            "attacker_controlled_unit_label_on_source_data": {
                "required": True,
                "present": False,
            },
            "target_accepts_amount_yuan": {
                "required": True,
                "present": amount_supported,
            },
        },
        "result": "applicable" if amount_supported else "not_applicable",
        "reason": (
            "The original unit/scale attack requires the agent to pass a monetary amount to the refund tool. "
            "The pinned τ³-bench retail return tool accepts item IDs and a payment-method ID, but no refund amount. "
            "Adding an amount argument would modify the benchmark target."
            if not amount_supported
            else "The return tool accepts a monetary amount; inspect the original task and policy before proceeding."
        ),
        "benchmark_modified": False,
    }
    print(json.dumps(result, ensure_ascii=True, indent=2))
    return 0 if result["pinned_source"] and not amount_supported else 1


if __name__ == "__main__":
    raise SystemExit(main())
