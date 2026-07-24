from __future__ import annotations

import ast
from dataclasses import dataclass


class PolicyViolation(ValueError):
    """Raised when candidate source exceeds its research permissions."""


@dataclass(frozen=True)
class CandidatePolicy:
    allowed_imports: frozenset[str] = frozenset(
        {
            "__future__",
            "joblib",
            "lightgbm",
            "math",
            "numpy",
            "pandas",
            "scipy",
            "sklearn",
            "statistics",
            "torch",
            "warnings",
        }
    )
    allowed_local_imports: frozenset[str] = frozenset({"prepare", "metrics"})
    max_lines: int = 1600
    max_ast_nodes: int = 20_000
    forbidden_names: frozenset[str] = frozenset(
        {
            "open",
            "exec",
            "eval",
            "compile",
            "input",
            "breakpoint",
            "__import__",
            "globals",
            "locals",
            "vars",
            "getattr",
            "setattr",
            "delattr",
        }
    )


def validate_candidate_source(source: str, policy: CandidatePolicy | None = None) -> ast.Module:
    active = policy or CandidatePolicy()
    line_count = len(source.splitlines())
    if line_count > active.max_lines:
        raise PolicyViolation(f"Candidate has {line_count} lines; maximum is {active.max_lines}")
    try:
        tree = ast.parse(source)
    except SyntaxError as error:
        raise PolicyViolation(f"Candidate is not valid Python: {error}") from error
    nodes = list(ast.walk(tree))
    if len(nodes) > active.max_ast_nodes:
        raise PolicyViolation(
            f"Candidate has {len(nodes)} AST nodes; maximum is {active.max_ast_nodes}"
        )

    violations: list[str] = []
    for node in nodes:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            modules = (
                [alias.name for alias in node.names]
                if isinstance(node, ast.Import)
                else [node.module or ""]
            )
            for module in modules:
                root = module.split(".", 1)[0]
                if root not in active.allowed_imports | active.allowed_local_imports:
                    violations.append(f"import {module!r} is not allowed")
        elif isinstance(node, ast.Name) and node.id in active.forbidden_names:
            violations.append(f"name {node.id!r} is not allowed")
        elif isinstance(node, ast.Attribute) and (
            (node.attr.startswith("_") and node.attr not in {"__name__"})
            or node.attr in {"environ", "modules", "__dict__"}
        ):
            violations.append(f"attribute {node.attr!r} is not allowed")
    if violations:
        unique = "; ".join(dict.fromkeys(violations))
        raise PolicyViolation(unique)
    return tree
