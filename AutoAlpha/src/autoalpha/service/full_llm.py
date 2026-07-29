from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

from autoalpha.service.openai_client import CompatibleChatClient, ModelInvocationError

REVIEWER = "INDEPENDENT_REVIEWER"
FALSIFICATION_DESIGNER = "FALSIFICATION_DESIGNER"
ROOT_CAUSE_ANALYST = "ROOT_CAUSE_ANALYST"
FACTOR_LIBRARIAN = "FACTOR_LIBRARIAN"
PORTFOLIO_RESEARCHER = "PORTFOLIO_MECHANISM_RESEARCHER"
TCA_OBSERVER = "TCA_PAPER_OBSERVER"

RESEARCHER_DOMAIN = "RESEARCHER"
DATA_OFFICER_DOMAIN = "DATA_OFFICER"
RISK_OFFICER_DOMAIN = "RISK_OFFICER"
PORTFOLIO_MANAGER_DOMAIN = "PORTFOLIO_MANAGER"
AUDITOR_DOMAIN = "AUDITOR"
TRADER_DOMAIN = "TRADER"

FULL_LLM_ROLES = (
    REVIEWER,
    FALSIFICATION_DESIGNER,
    ROOT_CAUSE_ANALYST,
    FACTOR_LIBRARIAN,
    PORTFOLIO_RESEARCHER,
    TCA_OBSERVER,
)

RESEARCH_TEAM_DOMAINS = (
    RESEARCHER_DOMAIN,
    DATA_OFFICER_DOMAIN,
    RISK_OFFICER_DOMAIN,
    PORTFOLIO_MANAGER_DOMAIN,
    AUDITOR_DOMAIN,
    TRADER_DOMAIN,
)

ROLE_DOMAIN_MAP = {
    FALSIFICATION_DESIGNER: (RESEARCHER_DOMAIN, AUDITOR_DOMAIN),
    FACTOR_LIBRARIAN: (RESEARCHER_DOMAIN, DATA_OFFICER_DOMAIN),
    REVIEWER: (DATA_OFFICER_DOMAIN, AUDITOR_DOMAIN),
    PORTFOLIO_RESEARCHER: (PORTFOLIO_MANAGER_DOMAIN, RISK_OFFICER_DOMAIN),
    ROOT_CAUSE_ANALYST: (AUDITOR_DOMAIN, RISK_OFFICER_DOMAIN),
    TCA_OBSERVER: (TRADER_DOMAIN, RISK_OFFICER_DOMAIN),
}

DOMAIN_LABELS = {
    RESEARCHER_DOMAIN: "研究员",
    DATA_OFFICER_DOMAIN: "数据官",
    RISK_OFFICER_DOMAIN: "风控官",
    PORTFOLIO_MANAGER_DOMAIN: "组合经理",
    AUDITOR_DOMAIN: "审计员",
    TRADER_DOMAIN: "交易员",
}

DOMAIN_RESPONSIBILITIES = {
    RESEARCHER_DOMAIN: "提出机制假设、证伪计划和可解释分类",
    DATA_OFFICER_DOMAIN: "检查字段、时点、数据契约和可用性边界",
    RISK_OFFICER_DOMAIN: "挑战回撤、换手、容量、成本和集中度",
    PORTFOLIO_MANAGER_DOMAIN: "判断与现有组合的搭配价值和冗余风险",
    AUDITOR_DOMAIN: "记录污染、过拟合、失败根因和门禁证据",
    TRADER_DOMAIN: "把研究结论约束到真实调仓、成交和交割监控",
}

DOMAIN_HEADLINE_KEYS = {
    RESEARCHER_DOMAIN: (
        "canonical_mechanism",
        "mechanism_summary",
        "null_hypothesis",
    ),
    DATA_OFFICER_DOMAIN: ("leakage_risk", "execution_risk", "verdict"),
    RISK_OFFICER_DOMAIN: (
        "cost_risk",
        "capacity_risk",
        "primary_cause",
        "interaction_risks",
    ),
    PORTFOLIO_MANAGER_DOMAIN: (
        "preferred_action",
        "candidate_role",
        "complementarity_thesis",
    ),
    AUDITOR_DOMAIN: ("primary_cause", "verdict", "kill_conditions"),
    TRADER_DOMAIN: ("execution_diagnosis", "turnover_driver", "monitoring_actions"),
}

CANONICAL_MECHANISMS = {
    "PRICE_REVERSAL",
    "PRICE_MOMENTUM",
    "VOLUME_LIQUIDITY",
    "PRICE_VOLUME_INTERACTION",
    "VOLATILITY_RISK",
    "PATH_STABILITY",
    "EXECUTION_EFFICIENCY",
    "CROSS_SECTIONAL_RELATIVE_VALUE",
    "OTHER_INTERPRETABLE",
}

FALSIFICATION_TESTS = {
    "PARAMETER_NEIGHBORHOOD",
    "DELAY_SENSITIVITY",
    "COST_STRESS",
    "WALK_FORWARD_STABILITY",
    "REGIME_STABILITY",
    "TURNOVER_BREAKDOWN",
    "COVERAGE_STABILITY",
    "CORRELATION_REDUNDANCY",
}


@dataclass(frozen=True)
class RoleSpec:
    role: str
    stage: str
    required_keys: frozenset[str]
    prompt: str


@dataclass(frozen=True)
class RoleOutcome:
    role: str
    stage: str
    status: str
    artifact: dict[str, Any]
    usage: dict[str, int]
    prompt_hash: str | None
    response_hash: str | None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "stage": self.stage,
            "status": self.status,
            "artifact": self.artifact,
            "usage": self.usage,
            "prompt_hash": self.prompt_hash,
            "response_hash": self.response_hash,
            "error": self.error,
        }


_COMMON_POLICY = """
You are one bounded advisory role inside an institutional quantitative-research platform.
Return exactly one JSON object and no prose. You may explain, challenge, classify, or propose tests,
but you must not approve a factor, change a deterministic gate, choose portfolio weights, infer
hidden-test details, or request exact performance metrics. Treat all supplied evidence as public
research evidence. Do not invent unavailable data. Keep arrays concise and use stable uppercase
category identifiers where the contract requests them.
The data contract distinguishes RESEARCH_ELIGIBLE fields from staged and catalog-only inventory.
Only fields explicitly allowed in expressions may be treated as candidate inputs; staged products
may be discussed as future data work but never assumed present in the evaluated candidate.
"""

_SPECS = {
    REVIEWER: RoleSpec(
        REVIEWER,
        "PRE_EVALUATION",
        frozenset(
            {
                "verdict",
                "hypothesis_alignment",
                "leakage_risk",
                "execution_risk",
                "strengths",
                "concerns",
                "required_checks",
            }
        ),
        _COMMON_POLICY
        + """
Act as an independent pre-evaluation reviewer. Inspect only the hypothesis, typed expression,
declared data fields, timing convention, and non-performance library context. Determine whether the
formula actually expresses the hypothesis, identify possible look-ahead or execution ambiguity,
and name deterministic checks. verdict is advisory and must be one of CLEAR, CONCERNS, or
BLOCKING_CONCERN.
Risk fields must be LOW, MEDIUM, or HIGH. Do not predict returns.
""",
    ),
    FALSIFICATION_DESIGNER: RoleSpec(
        FALSIFICATION_DESIGNER,
        "PRE_EVALUATION",
        frozenset(
            {
                "null_hypothesis",
                "tests",
                "kill_conditions",
                "expected_failure_modes",
            }
        ),
        _COMMON_POLICY
        + f"""
Design a pre-registered falsification plan for the proposed economic mechanism. Every tests item
must contain test_type, rationale, and expected_if_valid. test_type must be one of
{sorted(FALSIFICATION_TESTS)}. Do not choose numeric thresholds or tune windows. Kill conditions
must be observable and must not depend on hidden-test results.
""",
    ),
    ROOT_CAUSE_ANALYST: RoleSpec(
        ROOT_CAUSE_ANALYST,
        "POST_EVALUATION",
        frozenset(
            {
                "primary_cause",
                "contributing_causes",
                "responsibility",
                "next_action",
                "retry_same_direction",
                "explanation",
            }
        ),
        _COMMON_POLICY
        + """
Diagnose categorical public gate failures. primary_cause and responsibility must distinguish
CANDIDATE, DATA, PROTOCOL, STATISTICS, PORTFOLIO, EXECUTION, MODEL, or SYSTEM. next_action must be
one of RETAIN_FOR_LIBRARY, NEW_MECHANISM, WAIT_FOR_DATA, REPAIR_PIPELINE,
CHANGE_PRODUCT_RESEARCH, or HUMAN_REVIEW. retry_same_direction is a boolean. Do not ask for
exact Sharpe or threshold distance.
""",
    ),
    FACTOR_LIBRARIAN: RoleSpec(
        FACTOR_LIBRARIAN,
        "PRE_EVALUATION",
        frozenset(
            {
                "canonical_mechanism",
                "mechanism_summary",
                "tags",
                "related_factors",
                "distinguishing_features",
            }
        ),
        _COMMON_POLICY
        + f"""
Act as a factor librarian. Normalize the proposal into exactly one canonical_mechanism from
{sorted(CANONICAL_MECHANISMS)}. Write a concise mechanism summary and stable lowercase tags.
related_factors may reference only factor ids supplied in library_context and each item must contain
factor_id, relation, confidence, and rationale. Never classify on performance.
""",
    ),
    PORTFOLIO_RESEARCHER: RoleSpec(
        PORTFOLIO_RESEARCHER,
        "PORTFOLIO_ADVISORY",
        frozenset(
            {
                "candidate_role",
                "complementarity_thesis",
                "interaction_risks",
                "suggested_tests",
                "preferred_action",
            }
        ),
        _COMMON_POLICY
        + """
Study economic complementarity between the candidate and the active factor mechanisms. Do not
choose weights. preferred_action is advisory and must be HOLD, ADD_TEST, REPLACE_TEST, or
REMOVE_TEST. Focus on redundancy, opposing regimes, turnover netting, and concentration. The
deterministic portfolio engine remains the sole decision maker.
""",
    ),
    TCA_OBSERVER: RoleSpec(
        TCA_OBSERVER,
        "POST_EVALUATION",
        frozenset(
            {
                "execution_diagnosis",
                "turnover_driver",
                "cost_risk",
                "capacity_risk",
                "monitoring_actions",
            }
        ),
        _COMMON_POLICY
        + """
Act as a transaction-cost and paper-trading observer. Interpret only categorical execution
evidence and declared timing assumptions. cost_risk and capacity_risk must be LOW, MEDIUM, HIGH,
or UNKNOWN. Recommend measurements and alerts, never orders, position sizes, or gate overrides.
""",
    ),
}


class FullLLMResearchTeam:
    def __init__(self, client: CompatibleChatClient) -> None:
        self.client = client

    async def pre_evaluation(
        self,
        *,
        candidate: dict[str, Any],
        library_context: list[dict[str, Any]],
        data_context: dict[str, Any],
        roles: tuple[str, ...] | None = None,
    ) -> dict[str, RoleOutcome]:
        selected = roles or (REVIEWER, FALSIFICATION_DESIGNER, FACTOR_LIBRARIAN)
        allowed = {REVIEWER, FALSIFICATION_DESIGNER, FACTOR_LIBRARIAN}
        if any(role not in allowed for role in selected):
            raise ValueError("Invalid pre-evaluation role selection")
        shared = {
            "candidate": _candidate_context(candidate),
            "library_context": _library_context(library_context),
            "data_context": _data_contract_context(data_context),
        }
        outcomes = await asyncio.gather(*(self._invoke(role, shared) for role in selected))
        return {outcome.role: outcome for outcome in outcomes}

    async def portfolio_advisory(
        self,
        *,
        candidate: dict[str, Any],
        librarian: dict[str, Any],
        active_members: list[dict[str, Any]],
        public_feedback: dict[str, Any],
    ) -> RoleOutcome:
        return await self._invoke(
            PORTFOLIO_RESEARCHER,
            {
                "candidate": _candidate_context(candidate),
                "candidate_knowledge": librarian,
                "active_portfolio": _active_portfolio_context(active_members),
                "public_feedback": public_feedback,
            },
        )

    async def post_evaluation(
        self,
        *,
        candidate: dict[str, Any],
        public_feedback: dict[str, Any],
        falsification_plan: dict[str, Any],
        falsification_results: list[dict[str, Any]],
        portfolio_advisory: dict[str, Any],
        roles: tuple[str, ...] | None = None,
    ) -> dict[str, RoleOutcome]:
        selected = roles or (ROOT_CAUSE_ANALYST, TCA_OBSERVER)
        if any(role not in {ROOT_CAUSE_ANALYST, TCA_OBSERVER} for role in selected):
            raise ValueError("Invalid post-evaluation role selection")
        shared = {
            "candidate": _candidate_context(candidate),
            "public_feedback": public_feedback,
            "falsification_plan": falsification_plan,
            "falsification_results": falsification_results,
            "portfolio_advisory": portfolio_advisory,
        }
        outcomes = await asyncio.gather(*(self._invoke(role, shared) for role in selected))
        return {outcome.role: outcome for outcome in outcomes}

    async def _invoke(self, role: str, context: dict[str, Any]) -> RoleOutcome:
        spec = _SPECS[role]
        try:
            generated = await self.client.analyze(
                role=role,
                system_prompt=spec.prompt,
                context=context,
                required_keys=spec.required_keys,
            )
            artifact = _normalize_artifact(role, generated.artifact, context)
            return RoleOutcome(
                role=role,
                stage=spec.stage,
                status="COMPLETED",
                artifact=artifact,
                usage=generated.usage,
                prompt_hash=generated.prompt_hash,
                response_hash=generated.response_hash,
            )
        except ModelInvocationError as error:
            return RoleOutcome(
                role=role,
                stage=spec.stage,
                status="FAILED",
                artifact={"advisory_available": False},
                usage=error.usage,
                prompt_hash=error.prompt_hash,
                response_hash=error.response_hash,
                error=f"{error.stage}: {type(error).__name__}",
            )
        except Exception as error:  # Role advice must never control iteration availability.
            return RoleOutcome(
                role=role,
                stage=spec.stage,
                status="FAILED",
                artifact={"advisory_available": False},
                usage={},
                prompt_hash=None,
                response_hash=None,
                error=f"role_normalization: {type(error).__name__}",
            )


def categorical_research_feedback(
    metrics: dict[str, Any],
    *,
    decision: str,
    candidate_eligible: bool,
    portfolio_action: str,
    portfolio_accepted: bool,
) -> dict[str, Any]:
    exploratory = sorted(str(item) for item in metrics.get("exploratory_gate_failures", []))
    portfolio = sorted(str(item) for item in metrics.get("portfolio_action_gate_failures", []))
    all_failures = set(exploratory) | set(portfolio)
    return {
        "decision_class": decision,
        "candidate_eligible": bool(candidate_eligible),
        "portfolio_action": portfolio_action,
        "portfolio_accepted": bool(portfolio_accepted),
        "failure_categories": sorted(all_failures),
        "evidence_bands": {
            "statistical_reliability": _band(
                all_failures,
                {"deflated_sharpe", "net_return_significance", "multiple_testing"},
            ),
            "cross_period_stability": _band(
                all_failures,
                {
                    "walk_forward_positive_fraction",
                    "walk_forward_worst_sharpe",
                    "annual_dispersion",
                    "parameter_positive_fraction",
                    "parameter_worst_sharpe",
                },
            ),
            "execution_efficiency": _band(
                all_failures,
                {"turnover", "cost_stress", "capacity", "coverage"},
            ),
            "portfolio_increment": _band(
                all_failures,
                {"incremental_net_ir", "incremental_annual_return", "portfolio_value"},
            ),
        },
        "feedback_policy": "CATEGORICAL_PUBLIC_ONLY_NO_EXACT_METRICS",
    }


def evaluate_falsification_plan(
    plan: dict[str, Any], metrics: dict[str, Any]
) -> list[dict[str, Any]]:
    failures = set(str(item) for item in metrics.get("exploratory_gate_failures", []))
    failures.update(str(item) for item in metrics.get("portfolio_action_gate_failures", []))
    mapping = {
        "PARAMETER_NEIGHBORHOOD": {"parameter_positive_fraction", "parameter_worst_sharpe"},
        "COST_STRESS": {"cost_stress"},
        "WALK_FORWARD_STABILITY": {
            "walk_forward_positive_fraction",
            "walk_forward_worst_sharpe",
        },
        "TURNOVER_BREAKDOWN": {"turnover"},
        "COVERAGE_STABILITY": {"coverage"},
        "CORRELATION_REDUNDANCY": {"factor_correlation"},
    }
    results = []
    for raw in plan.get("tests", [])[:8]:
        if not isinstance(raw, dict):
            continue
        test_type = str(raw.get("test_type", "")).upper()
        if test_type not in FALSIFICATION_TESTS:
            continue
        relevant = mapping.get(test_type)
        status = (
            "NOT_EVALUATED" if relevant is None else "FAILED" if failures & relevant else "PASSED"
        )
        results.append(
            {
                "test_type": test_type,
                "status": status,
                "failure_categories": sorted(failures & (relevant or set())),
            }
        )
    return results


def role_catalog() -> list[dict[str, Any]]:
    return [
        {
            "role": spec.role,
            "stage": spec.stage,
            "required_keys": sorted(spec.required_keys),
            "domains": list(ROLE_DOMAIN_MAP.get(spec.role, ())),
            "decision_authority": "ADVISORY_ONLY",
        }
        for spec in _SPECS.values()
    ]


def research_team_domain_catalog() -> list[dict[str, Any]]:
    return [
        {
            "domain": domain,
            "label": DOMAIN_LABELS[domain],
            "responsibility": DOMAIN_RESPONSIBILITIES[domain],
            "roles": [
                role for role, domains in ROLE_DOMAIN_MAP.items() if domain in domains
            ],
            "decision_authority": "ADVISORY_ONLY_STRUCTURED_EVIDENCE",
        }
        for domain in RESEARCH_TEAM_DOMAINS
    ]


def summarize_research_team_domains(
    artifacts: list[dict[str, Any]],
) -> dict[str, Any]:
    matrix = {
        item["domain"]: {
            **item,
            "completed": 0,
            "failed": 0,
            "artifact_count": 0,
            "latest_at": None,
            "latest_candidate_id": None,
            "latest_run_id": None,
            "latest_iteration": None,
            "latest_role": None,
            "latest_headline": None,
            "covered_roles": [],
        }
        for item in research_team_domain_catalog()
    }
    covered: dict[str, set[str]] = {domain: set() for domain in matrix}
    for artifact in artifacts:
        role = str(artifact.get("role") or "")
        for domain in ROLE_DOMAIN_MAP.get(role, ()):
            item = matrix[domain]
            item["artifact_count"] += 1
            if artifact.get("status") == "COMPLETED":
                item["completed"] += 1
            else:
                item["failed"] += 1
            covered[domain].add(role)
            created_at = artifact.get("created_at")
            if item["latest_at"] is None or str(created_at or "") > str(item["latest_at"] or ""):
                item["latest_at"] = created_at
                item["latest_candidate_id"] = artifact.get("candidate_id")
                item["latest_run_id"] = artifact.get("run_id")
                item["latest_iteration"] = artifact.get("iteration")
                item["latest_role"] = role
                item["latest_headline"] = _domain_headline(
                    domain,
                    artifact.get("artifact") if isinstance(artifact.get("artifact"), dict) else {},
                    artifact.get("error"),
                )
    for domain, roles in covered.items():
        matrix[domain]["covered_roles"] = sorted(roles)
        expected = set(matrix[domain]["roles"])
        matrix[domain]["coverage_ratio"] = (
            len(roles & expected) / len(expected) if expected else 0.0
        )
        matrix[domain]["status"] = (
            "ACTIVE"
            if matrix[domain]["completed"]
            else "FAILED_OPEN"
            if matrix[domain]["failed"]
            else "WAITING_FOR_ARTIFACTS"
        )
    return {
        "protocol": "LLM_RESEARCH_TEAM_DOMAIN_MATRIX_V1",
        "decision_authority": "ADVISORY_ONLY_DETERMINISTIC_GATES_REMAIN_FINAL",
        "domains": [matrix[domain] for domain in RESEARCH_TEAM_DOMAINS],
    }


def _domain_headline(domain: str, artifact: dict[str, Any], error: Any) -> str:
    if error:
        return str(error)[:180]
    for key in DOMAIN_HEADLINE_KEYS.get(domain, ()):
        value = artifact.get(key)
        if value in (None, "", []):
            continue
        if isinstance(value, list):
            return " · ".join(str(item) for item in value[:3])[:180]
        return str(value)[:180]
    return "结构化研究意见"


def _candidate_context(candidate: dict[str, Any]) -> dict[str, Any]:
    allowed = {
        "name",
        "family",
        "hypothesis",
        "change",
        "expected",
        "expected_direction",
        "expression",
    }
    return {key: candidate[key] for key in allowed if key in candidate}


def _library_context(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "factor_id": str(record.get("factor_id")),
            "name": str(record.get("name", "")),
            "family": str(record.get("family", "")),
            "expression": record.get("proposal", {}).get("expression"),
        }
        for record in records[:40]
    ]


def _data_contract_context(data_context: dict[str, Any]) -> dict[str, Any]:
    allowed = {
        "research_task_id",
        "first_trade_date",
        "last_trade_date",
        "available_factor_fields",
        "field_catalog",
        "data_products",
        "data_policy",
        "signal_timing",
        "extended_data_experiment",
        "price_research_ready",
        "source_integrity_passed",
        "institutional_pit_ready",
        "blockers",
    }
    return {key: data_context[key] for key in allowed if key in data_context}


def _active_portfolio_context(members: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "factor_count": len(members),
        "members": [
            {
                "factor_id": member.get("factor_id"),
                "name": member.get("name"),
                "family": member.get("family"),
                "hypothesis": member.get("hypothesis"),
            }
            for member in members[:12]
        ],
        "weights_withheld": True,
    }


def _normalize_artifact(
    role: str, artifact: dict[str, Any], context: dict[str, Any]
) -> dict[str, Any]:
    clean = _bounded_value(artifact)
    assert isinstance(clean, dict)
    if role == FACTOR_LIBRARIAN:
        mechanism = str(clean.get("canonical_mechanism", "OTHER_INTERPRETABLE")).upper()
        clean["canonical_mechanism"] = (
            mechanism if mechanism in CANONICAL_MECHANISMS else "OTHER_INTERPRETABLE"
        )
        clean["tags"] = sorted(
            {
                str(tag).strip().lower().replace(" ", "-")
                for tag in clean.get("tags", [])[:12]
                if str(tag).strip()
            }
        )
        allowed_ids = {str(item.get("factor_id")) for item in context.get("library_context", [])}
        related_factors = []
        for relation in clean.get("related_factors", [])[:12]:
            if not isinstance(relation, dict):
                continue
            if str(relation.get("factor_id")) not in allowed_ids:
                continue
            relation["confidence"] = _confidence(relation.get("confidence"))
            related_factors.append(relation)
        clean["related_factors"] = related_factors
    elif role == FALSIFICATION_DESIGNER:
        clean["tests"] = [
            test
            for test in clean.get("tests", [])[:8]
            if isinstance(test, dict)
            and str(test.get("test_type", "")).upper() in FALSIFICATION_TESTS
        ]
        for test in clean["tests"]:
            test["test_type"] = str(test["test_type"]).upper()
    return clean


def _bounded_value(value: Any, *, depth: int = 0) -> Any:
    if depth >= 6:
        return "[DEPTH_LIMIT]"
    if isinstance(value, dict):
        return {
            str(key)[:80]: _bounded_value(item, depth=depth + 1)
            for key, item in list(value.items())[:40]
        }
    if isinstance(value, list):
        return [_bounded_value(item, depth=depth + 1) for item in value[:20]]
    if isinstance(value, str):
        return value[:2000]
    if isinstance(value, bool | int | float) or value is None:
        return value
    return str(value)[:2000]


def _band(failures: set[str], related: set[str]) -> str:
    return "NEEDS_IMPROVEMENT" if failures & related else "NO_PUBLIC_FAILURE_RECORDED"


def _confidence(value: Any) -> float:
    if isinstance(value, str):
        label = value.strip().upper()
        if label in {"LOW", "MEDIUM", "HIGH"}:
            return {"LOW": 0.30, "MEDIUM": 0.60, "HIGH": 0.90}[label]
    try:
        return min(max(float(value), 0.0), 1.0)
    except (TypeError, ValueError):
        return 0.0
