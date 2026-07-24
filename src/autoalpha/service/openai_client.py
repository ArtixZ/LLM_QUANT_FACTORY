from __future__ import annotations

import asyncio
import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import httpx

from autoalpha.dsl.expression import Expression, FactorDefinition


@dataclass(frozen=True)
class GeneratedProposal:
    factor: FactorDefinition
    change: str
    expected: str
    raw: dict[str, Any]
    usage: dict[str, int]
    prompt_hash: str
    response_hash: str


@dataclass(frozen=True)
class GeneratedAnalysis:
    role: str
    artifact: dict[str, Any]
    usage: dict[str, int]
    prompt_hash: str
    response_hash: str


@dataclass(frozen=True)
class _ChatEnvelope:
    content: str
    usage: dict[str, int]
    prompt_hash: str
    response_hash: str
    status_code: int


class ModelInvocationError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        stage: str,
        prompt_hash: str,
        response_hash: str | None = None,
        usage: dict[str, int] | None = None,
        status_code: int | None = None,
        provider_error: dict[str, str] | None = None,
    ) -> None:
        super().__init__(message)
        self.stage = stage
        self.prompt_hash = prompt_hash
        self.response_hash = response_hash
        self.usage = usage or {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        self.status_code = status_code
        self.provider_error = provider_error

    def audit_payload(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "prompt_hash": self.prompt_hash,
            "response_hash": self.response_hash,
            "usage": self.usage,
            "status_code": self.status_code,
            "provider_error": self.provider_error,
        }


def _provider_error_details(response: httpx.Response | None) -> dict[str, str] | None:
    if response is None or not response.is_error:
        return None
    try:
        body = response.json()
    except ValueError:
        text = response.text.strip()
        return {"message": text[:500]} if text else None
    error = body.get("error") if isinstance(body, dict) else None
    source = error if isinstance(error, dict) else body if isinstance(body, dict) else {}
    details = {
        key: str(source[key])[:500]
        for key in ("code", "type", "param", "message")
        if source.get(key) is not None
    }
    return details or None


class CompatibleChatClient:
    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        timeout: float = 120,
        temperature: float = 0.7,
        max_tokens: int = 8192,
        transport_retries: int = 2,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.transport_retries = transport_retries
        self.transport = transport

    async def propose(
        self,
        memories: list[dict[str, Any]],
        iteration: int,
        *,
        data_context: dict[str, Any] | None = None,
    ) -> GeneratedProposal:
        messages: list[dict[str, str]] = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "iteration": iteration,
                        "data_context": data_context or {},
                        "research_memory": memories,
                    },
                    ensure_ascii=False,
                ),
            },
        ]
        return await self._generate(messages)

    async def repair(
        self,
        rejected: GeneratedProposal,
        reason: str,
        memories: list[dict[str, Any]],
        iteration: int,
        *,
        data_context: dict[str, Any] | None = None,
    ) -> GeneratedProposal:
        messages: list[dict[str, str]] = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "iteration": iteration,
                        "data_context": data_context or {},
                        "research_memory": memories,
                        "repair_request": {
                            "reason": reason,
                            "rejected_proposal": rejected.raw,
                            "instruction": (
                                "Return a materially distinct corrected proposal. Fix the exact "
                                "contract or semantic error and do not repeat the rejected tree."
                            ),
                        },
                    },
                    ensure_ascii=False,
                ),
            },
        ]
        return await self._generate(messages)

    async def analyze(
        self,
        *,
        role: str,
        system_prompt: str,
        context: dict[str, Any],
        required_keys: frozenset[str] | set[str],
    ) -> GeneratedAnalysis:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(context, ensure_ascii=False)},
        ]
        return await self._generate_analysis(
            role,
            messages,
            frozenset(required_keys),
        )

    async def _generate_analysis(
        self,
        role: str,
        messages: list[dict[str, str]],
        required_keys: frozenset[str],
        *,
        allow_contract_repair: bool = True,
    ) -> GeneratedAnalysis:
        envelope = await self._request(messages)
        try:
            artifact = _parse_json_object(envelope.content)
            missing = required_keys - artifact.keys()
            if missing:
                raise ValueError(f"Role artifact is missing fields: {sorted(missing)}")
        except (TypeError, ValueError, json.JSONDecodeError) as error:
            if allow_contract_repair:
                repaired = [
                    *messages,
                    {"role": "assistant", "content": envelope.content},
                    {
                        "role": "user",
                        "content": (
                            f"The {role} artifact violated its JSON contract: "
                            f"{type(error).__name__}: {error}. Return one corrected JSON object "
                            f"containing these required keys: {sorted(required_keys)}."
                        ),
                    },
                ]
                return await self._generate_analysis(
                    role,
                    repaired,
                    required_keys,
                    allow_contract_repair=False,
                )
            raise ModelInvocationError(
                f"Role artifact rejected: {type(error).__name__}: {error}",
                stage="role_contract",
                prompt_hash=envelope.prompt_hash,
                response_hash=envelope.response_hash,
                usage=envelope.usage,
                status_code=envelope.status_code,
            ) from error
        return GeneratedAnalysis(
            role=role,
            artifact=artifact,
            usage=envelope.usage,
            prompt_hash=envelope.prompt_hash,
            response_hash=envelope.response_hash,
        )

    async def _generate(
        self,
        messages: list[dict[str, str]],
        *,
        allow_contract_repair: bool = True,
    ) -> GeneratedProposal:
        envelope = await self._request(messages)
        try:
            raw = _parse_json(envelope.content)
            expression = Expression.from_dict(raw["expression"])
            factor = FactorDefinition(
                name=str(raw["name"]),
                family=str(raw["family"]),
                hypothesis=str(raw["hypothesis"]),
                expression=expression,
                expected_direction=_direction(raw.get("expected_direction", 1)),
            )
        except (KeyError, TypeError, ValueError) as error:
            if allow_contract_repair:
                corrected_messages = [
                    *messages,
                    {"role": "assistant", "content": envelope.content},
                    {
                        "role": "user",
                        "content": (
                            "The previous response violated the proposal contract: "
                            f"{type(error).__name__}: {error}. Return one corrected JSON object "
                            "only, using the exact DSL contracts in the system message."
                        ),
                    },
                ]
                return await self._generate(corrected_messages, allow_contract_repair=False)
            raise ModelInvocationError(
                f"Proposal contract rejected: {type(error).__name__}: {error}",
                stage="proposal_contract",
                prompt_hash=envelope.prompt_hash,
                response_hash=envelope.response_hash,
                usage=envelope.usage,
                status_code=envelope.status_code,
            ) from error
        return GeneratedProposal(
            factor=factor,
            change=str(raw["change"]),
            expected=str(raw["expected"]),
            raw=raw,
            usage=envelope.usage,
            prompt_hash=envelope.prompt_hash,
            response_hash=envelope.response_hash,
        )

    async def _request(self, messages: list[dict[str, str]]) -> _ChatEnvelope:
        prompt_hash = hashlib.sha256(
            json.dumps(messages, ensure_ascii=False, sort_keys=True).encode()
        ).hexdigest()
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "response_format": {"type": "json_object"},
            "max_tokens": self.max_tokens,
        }
        headers = {"Authorization": f"Bearer {self.api_key}"}
        response: httpx.Response | None = None
        last_error: httpx.HTTPError | None = None
        for attempt in range(self.transport_retries + 1):
            request_payload = dict(payload)
            try:
                async with httpx.AsyncClient(
                    timeout=self.timeout, transport=self.transport
                ) as client:
                    response = await client.post(
                        f"{self.base_url}/chat/completions",
                        headers=headers,
                        json=request_payload,
                    )
                    if response.status_code == 400:
                        request_payload.pop("response_format", None)
                        response = await client.post(
                            f"{self.base_url}/chat/completions",
                            headers=headers,
                            json=request_payload,
                        )
                    response.raise_for_status()
                break
            except httpx.HTTPError as error:
                last_error = error
                retryable = isinstance(
                    error,
                    (
                        httpx.ConnectError,
                        httpx.ReadTimeout,
                        httpx.RemoteProtocolError,
                    ),
                )
                if not retryable or attempt >= self.transport_retries:
                    break
                await asyncio.sleep(2.0**attempt)
        if response is None or last_error is not None and response.is_error:
            error = last_error or httpx.HTTPError("Provider request failed")
            status_code = (
                error.response.status_code if isinstance(error, httpx.HTTPStatusError) else None
            )
            provider_error = _provider_error_details(response)
            raise ModelInvocationError(
                f"Provider request failed after {self.transport_retries + 1} attempts: "
                f"{type(error).__name__}",
                stage="transport",
                prompt_hash=prompt_hash,
                status_code=status_code,
                provider_error=provider_error,
            ) from error
        try:
            body = response.json()
            content = body["choices"][0]["message"]["content"]
        except (KeyError, TypeError, ValueError) as error:
            raise ModelInvocationError(
                f"Invalid provider response envelope: {type(error).__name__}",
                stage="response_envelope",
                prompt_hash=prompt_hash,
                status_code=response.status_code,
            ) from error
        response_hash = hashlib.sha256(str(content).encode()).hexdigest()
        usage_raw = body.get("usage", {})
        usage = {
            "prompt_tokens": int(usage_raw.get("prompt_tokens", 0)),
            "completion_tokens": int(usage_raw.get("completion_tokens", 0)),
            "total_tokens": int(usage_raw.get("total_tokens", 0)),
        }
        return _ChatEnvelope(
            content=str(content),
            usage=usage,
            prompt_hash=prompt_hash,
            response_hash=response_hash,
            status_code=response.status_code,
        )


def _parse_json(content: str) -> dict[str, Any]:
    value = _parse_json_object(content)
    required = {"name", "family", "hypothesis", "change", "expected", "expression"}
    missing = required - value.keys()
    if missing:
        raise ValueError(f"Model proposal is missing fields: {sorted(missing)}")
    return value


def _parse_json_object(content: str) -> dict[str, Any]:
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", content.strip())
    value = json.loads(cleaned)
    if not isinstance(value, Mapping):
        raise TypeError("Model response must be a JSON object")
    return dict(value)


def _direction(value: Any) -> int:
    if isinstance(value, str):
        value = value.strip().lower()
    if value in (1, 1.0, "+1", "1", "positive", "long", "up", "higher", "bullish"):
        return 1
    if value in (-1, -1.0, "-1", "negative", "short", "down", "lower", "bearish"):
        return -1
    raise ValueError(f"Unsupported expected_direction: {value!r}")


_SYSTEM_PROMPT = """You are the Researcher in an institutional A-share factor platform.
Return exactly one JSON object and no prose. Propose one falsifiable cross-sectional factor, taking
the supplied persistent memory into account and avoiding prior failures and duplicates.

The user payload contains data_context.research_program. Treat its active portfolio as the
benchmark to beat, not as background prose. Follow the current research_mandate, gate_policy and
action_weight_grid. Target incremental portfolio value, low correlation and bounded turnover.
The primary objective and every promotion gate use the A-share long-only weekly execution proxy.
Long-short Alpha, IC and Rank IC are secondary diagnostics only and cannot rescue a candidate that
fails long-only return, drawdown, cost, turnover, stability or capacity requirements.
The adaptive_direction object is a frozen micro-campaign selected by a deterministic public-data
diagnostic. The proposal must directly serve its objective and success criteria, and the change
field must state that alignment. If target_mechanism is present, test exactly that economic
mechanism; changing only a lookback, rank wrapper, smoothing window, or sign does not qualify.
Do not switch objectives, blend several directions, request more
attempts, or continue a cooled direction. Every proposal consumes one of at most three attempts,
including repaired or failed proposals, so prefer one clear mechanism test over parameter tuning.
Explicitly avoid mechanisms and expression signatures that dominate recent failures. A parameter-
only variation of a prior tree is not novel. Prefer slower signals when recent candidates exceed the
turnover gate. The candidate may enter at a small weight, so a modest standalone factor is useful
when it is genuinely orthogonal to the active members.
Candidates are also screened for behavioral redundancy: a deterministic gate rejects any signal
whose realized cross-sectional correlation with an existing library factor is too high, even when
the expression tree differs. Memory entries expose this as single_factor.library_redundancy and a
signal_redundancy admission failure. When a redundancy failure or a high max_signal_correlation
appears in memory, do not reshape the same signal with different operators; switch to a different
economic mechanism, horizon class, or data product instead.

Public walk-forward feedback is adaptive research evidence, not untouched out-of-sample evidence.
Prefer mechanisms that remain positive across folds and parameter neighborhoods instead of chasing
the best aggregate Sharpe. Memories without an evaluation protocol are legacy diagnostics and may
only be used to avoid duplicates or known failures, never as promotion evidence. The holdout is a
sealed evaluator: do not infer its dates, metrics, regimes, thresholds, or likely failure details,
and never propose a parameter change in response to a categorical holdout or capital verdict.

Required keys: name, family, hypothesis, change, expected, expected_direction, expression.
expression is a typed tree: {"operator": ..., "arguments": [...], "parameters": {...}}.
Allowed fields are exactly data_context.available_factor_fields. The base fields are close,
adj_close, amount and vol. When present, daily_basic fields add valuation, capitalization,
turnover and volume-ratio observations; moneyflow fields add after-close order-flow observations.
Use data_context.field_catalog for each field's economic meaning, unit, source product and status.
data_context.data_products is a capability inventory: RESEARCH_ELIGIBLE products may be used,
STAGED_COVERAGE_INCOMPLETE products are visible roadmap only, and catalog/downloadable products
must not be referenced until their point-in-time integration is complete. If adaptive_direction is
EXPLORE_EXTENDED_DATA, the expression must use at least one required_factor_fields entry and should
test one coherent valuation, turnover, capitalization, or order-flow mechanism rather than mixing
unrelated new fields. A server-side validator enforces this alignment.
Treat every daily field as end-of-day information that can only drive the next session's order.
Allowed operators: field, negate, absolute, add, subtract, multiply, divide, delay, delta, returns,
rolling_mean, rolling_sum, rolling_std, rolling_min, rolling_max, cs_rank, cs_zscore, winsorize_mad.
Windows must be positive integers <= 252, and total lookback across nested rolling, delay, delta,
and returns operators must also remain <= 252. Keep expressions under 30 nodes. Do not use future
data, fields absent from data_context, unversioned fundamentals, current constituents, Python,
files, network, or tools.
Use equal units for add/subtract. Prefer interpretable mechanisms over parameter search.
Use these exact parameter contracts:
- field: {"operator":"field","arguments":[],"parameters":{"name":"close"}}
- returns/delay/delta: parameters must contain integer "periods"
- rolling operators: parameters must contain integer "window"
- winsorize_mad: parameters must contain numeric "threshold"
expected_direction must be exactly 1 or -1 and describes the final expression. If the expression
already uses negate to encode the expected sign, use 1. Never use field/period as parameter names.
"""
