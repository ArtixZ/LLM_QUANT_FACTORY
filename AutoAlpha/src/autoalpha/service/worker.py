from __future__ import annotations

import asyncio
import hashlib
import json
import os
import uuid
from collections import Counter
from contextlib import suppress
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from autoalpha.config import ResearchConfig
from autoalpha.data.research_fields import (
    build_research_data_capabilities,
    expression_fields,
    products_for_fields,
)
from autoalpha.dsl.expression import FactorDefinition, canonical_family, field, operation
from autoalpha.operations.artifacts import ArtifactRegistry
from autoalpha.research.multiple_testing import probability_of_backtest_overfitting
from autoalpha.research.statistics import benjamini_hochberg
from autoalpha.service.blind_evaluator import BlindEvaluationBoundary
from autoalpha.service.canonical_evaluation import (
    CANONICAL_LIBRARY_PROTOCOL,
    canonical_library_config,
    evaluate_canonical_library_factor,
)
from autoalpha.service.direction import (
    assess_direction_outcome,
    classify_mechanism,
    diagnose_direction,
    direction_definition,
)
from autoalpha.service.evaluator import PriceVolumeEvaluator
from autoalpha.service.factor_behavior import load_behavior_snapshot
from autoalpha.service.full_llm import (
    FACTOR_LIBRARIAN,
    FALSIFICATION_DESIGNER,
    PORTFOLIO_RESEARCHER,
    REVIEWER,
    ROOT_CAUSE_ANALYST,
    TCA_OBSERVER,
    FullLLMResearchTeam,
    RoleOutcome,
    categorical_research_feedback,
    evaluate_falsification_plan,
)
from autoalpha.service.multifactor import (
    MultiFactorResearchEngine,
    PortfolioDecision,
    factor_from_pool_record,
    portfolio_absolute_gate_failures,
)
from autoalpha.service.openai_client import (
    CompatibleChatClient,
    GeneratedProposal,
    ModelInvocationError,
)
from autoalpha.service.research_protocol import research_evidence_tier, task_research_config
from autoalpha.service.store import ServiceStore


class CredentialStore(Protocol):
    def get(self) -> str | None: ...

    def set(self, value: str) -> None: ...


@dataclass
class SecretVault:
    credential_store: CredentialStore | None = None
    api_key: str | None = None

    def configured(self) -> bool:
        return bool(self._resolve())

    def get(self) -> str:
        value = self._resolve()
        if not value:
            raise RuntimeError("OpenAI-compatible API key is not configured")
        return value

    def set(self, value: str) -> None:
        if self.credential_store is None:
            self.api_key = value
            return
        self.credential_store.set(value)
        self.api_key = None

    def _resolve(self) -> str | None:
        return (
            self.api_key
            or os.getenv("AUTOALPHA_API_KEY")
            or (self.credential_store.get() if self.credential_store else None)
        )


class ContinuousResearchWorker:
    def __init__(
        self,
        store: ServiceStore,
        vault: SecretVault,
        *,
        config_path: Path,
        artifact_root: Path,
        task_id: str = "legacy-ashare",
        iteration_semaphore: asyncio.Semaphore | None = None,
    ) -> None:
        self.store = store
        self.vault = vault
        self.config_path = config_path
        self.artifacts = ArtifactRegistry(artifact_root)
        self.task_id = task_id
        self.iteration_semaphore = iteration_semaphore
        self._task: asyncio.Task[None] | None = None
        self._evaluator: PriceVolumeEvaluator | None = None
        self._canonical_evaluator: PriceVolumeEvaluator | None = None
        self._proposal_queue: list[GeneratedProposal] = []
        self._proposal_queue_key: str | None = None

    @property
    def alive(self) -> bool:
        return self._task is not None and not self._task.done()

    async def start(self) -> dict[str, Any]:
        if self.alive:
            return self._state()
        settings = self._settings()
        required = [key for key in ("base_url", "model", "data_path") if not settings.get(key)]
        api_key_configured = self.vault.configured()
        if required or not api_key_configured:
            self._update_state(state="WAITING_CONFIGURATION", phase="CONFIGURE", last_error=None)
            if not api_key_configured:
                raise RuntimeError("缺少 DeepSeek API Key，请先在服务设置中保存凭证")
            raise RuntimeError(f"缺少服务配置: {', '.join(required)}")
        previous = self._state()
        run_id = str(previous.get("run_id") or f"run-{uuid.uuid4().hex[:12]}")
        self._proposal_queue.clear()
        self._proposal_queue_key = None
        self._update_state(
            state="RUNNING",
            phase="MEMORY",
            run_id=run_id,
            stop_requested=0,
            last_error=None,
        )
        self.store.append_event(
            "action",
            "RUN_STARTED",
            "持续研究已启动",
            "后台工作器将持续迭代，直到收到显式停止请求。",
            run_id=run_id,
        )
        self._task = asyncio.create_task(self._loop(run_id), name="autoalpha-research-loop")
        return self._state()

    async def stop(self) -> dict[str, Any]:
        state = self._state()
        if not self.alive:
            self._update_state(stop_requested=0, state="STOPPED", phase="STOPPED")
            return self._state()
        self._update_state(stop_requested=1, state="STOPPING", phase="STOPPING")
        self.store.append_event(
            "action",
            "STOP_REQUESTED",
            "停止请求已登记",
            "当前迭代完成后停止，不中断正在写入的研究制品。",
            run_id=state.get("run_id"),
            iteration=state.get("iteration"),
            level="WARN",
        )
        return self._state()

    async def run_genesis_baseline(self) -> dict[str, Any]:
        return await self._run_baseline(
            _genesis_baseline(),
            protocol="genesis_baseline_v1",
            origin="SYSTEM_GENESIS_BASELINE",
            register_in_factor_pool=False,
        )

    async def run_codex_baseline(self) -> dict[str, Any]:
        return await self._run_baseline(
            _codex_task_baseline(),
            protocol="codex_agent_baseline_v1",
            origin="CODEX_AGENT_BASELINE",
            register_in_factor_pool=True,
        )

    async def _run_baseline(
        self,
        factor: FactorDefinition,
        *,
        protocol: str,
        origin: str,
        register_in_factor_pool: bool,
    ) -> dict[str, Any]:
        if self.alive:
            raise RuntimeError("Stop the continuous research loop before establishing a baseline")
        settings = self._settings()
        if not settings.get("data_path"):
            raise RuntimeError("Research data workspace is not configured")
        previous = self._state()
        run_id = str(previous.get("run_id") or f"run-{uuid.uuid4().hex[:12]}")
        iteration = int(previous["iteration"]) + 1
        if self.store.candidate_exists(factor.factor_id):
            raise RuntimeError(f"Baseline candidate already exists: {factor.factor_id}")
        self._update_state(
            state="RUNNING",
            phase="EVALUATION",
            run_id=run_id,
            iteration=iteration,
            stop_requested=0,
            last_error=None,
        )
        self.store.begin_iteration(run_id, iteration)
        try:
            evaluator = self._get_evaluator(Path(settings["data_path"]))
            workspace = evaluator.workspace
            self.store.append_event(
                "audit",
                "BASELINE_STARTED",
                "初始化研究基准",
                f"使用冻结表达式 {factor.name} 建立可复现的任务级研究基准。",
                run_id=run_id,
                iteration=iteration,
                payload={
                    "task_id": self.task_id,
                    "candidate_id": factor.factor_id,
                    "expression": factor.expression.to_dict(),
                    "data_fingerprint": workspace.fingerprint,
                    "protocol": protocol,
                    "origin": origin,
                    "hidden_test_accessed": False,
                },
            )
            if register_in_factor_pool:
                evaluator.set_trial_count(min(self.store.factor_pool_count(), 5000) + 1)
            result = await asyncio.to_thread(evaluator.evaluate, factor)
            decision = "BASELINE_ESTABLISHED_RESEARCH_ONLY"
            proposal = {
                "name": factor.name,
                "family": factor.family,
                "hypothesis": factor.hypothesis,
                "expected_direction": factor.expected_direction,
                "expression": factor.expression.to_dict(),
                "protocol": protocol,
                "origin": origin,
                "tags": ["baseline", "codex"] if origin == "CODEX_AGENT_BASELINE" else ["baseline"],
            }
            eligible: bool | None = None
            if register_in_factor_pool:
                config = evaluator.config
                result.metrics["research_generation"] = f"{protocol}--{self.task_id}"
                result.metrics.update(
                    _multiple_testing_adjustments(
                        result.metrics,
                        self.store.metric_history(
                            limit=config.budget.max_candidates_per_generation,
                            run_id=run_id,
                        ),
                        generation=str(result.metrics["research_generation"]),
                        alpha=config.evaluation.maximum_net_return_p_value,
                    )
                )
                library_metrics = await asyncio.to_thread(
                    self._canonical_library_metrics,
                    Path(settings["data_path"]),
                    factor,
                    result.metrics,
                )
                engine = MultiFactorResearchEngine(
                    self.store,
                    evaluator,
                    config,
                    maximum_factors=int(settings.get("maximum_active_factors", "5")),
                    run_id=run_id,
                    source_task_id=self.task_id,
                )
                eligible = engine.register_candidate(
                    factor,
                    iteration,
                    proposal,
                    library_metrics,
                    task_metrics=result.metrics,
                )
                decision = (
                    "CODEX_BASELINE_SINGLE_FACTOR_ELIGIBLE"
                    if eligible
                    else "CODEX_BASELINE_SINGLE_FACTOR_SCREENED_OUT"
                )
            artifact_payload = json.dumps(
                {
                    "run_id": run_id,
                    "iteration": iteration,
                    "candidate_id": factor.factor_id,
                    "proposal": proposal,
                    "metrics": result.metrics,
                    "decision": decision,
                    "observations": result.observations,
                    "data_workspace": workspace.to_dict(),
                },
                ensure_ascii=False,
                sort_keys=True,
            ).encode()
            self._update_state(phase="DELIVERY")
            artifact = self.artifacts.publish(
                "codex-task-baseline" if origin == "CODEX_AGENT_BASELINE" else "genesis-baseline",
                artifact_payload,
                owner="autoalpha-service",
                metadata={
                    "run_id": run_id,
                    "iteration": iteration,
                    "data_fingerprint": workspace.fingerprint,
                    "protocol": protocol,
                    "origin": origin,
                    "task_id": self.task_id,
                },
            )
            self.store.finish_iteration(
                run_id,
                iteration,
                status="COMPLETED",
                candidate_id=factor.factor_id,
                proposal=proposal,
                metrics=result.metrics,
                decision=decision,
            )
            self.store.remember(
                run_id,
                iteration,
                "baseline",
                _memory_summary(
                    factor.name,
                    factor.family,
                    result.metrics,
                    decision,
                    proposal=proposal,
                ),
            )
            self.store.append_event(
                "research",
                "BASELINE_EVALUATED",
                "任务基准因子已完成",
                f"完成 {result.observations} 个交易日的真实价量基准评估。",
                run_id=run_id,
                iteration=iteration,
                payload={
                    "task_id": self.task_id,
                    "candidate_id": factor.factor_id,
                    "origin": origin,
                    "eligible": eligible,
                    "hidden_test_accessed": False,
                    **result.metrics,
                },
            )
            self.store.append_event(
                "delivery",
                "BASELINE_PUBLISHED",
                "基准制品已交付",
                f"不可变基准制品 {artifact.artifact_id} 已发布。",
                run_id=run_id,
                iteration=iteration,
                payload={
                    "artifact_id": artifact.artifact_id,
                    "content_hash": artifact.content_hash,
                    "data_fingerprint": workspace.fingerprint,
                    "decision": decision,
                    "origin": origin,
                    "task_id": self.task_id,
                },
            )
            self._update_state(state="READY", phase="WAITING", last_error=None)
            return {
                "candidate_id": factor.factor_id,
                "artifact_id": artifact.artifact_id,
                "decision": decision,
                "eligible": eligible,
                "observations": result.observations,
                "metrics": result.metrics,
                "origin": origin,
                "task_id": self.task_id,
                "state": self._state(),
            }
        except Exception as error:
            message = f"{type(error).__name__}: {error}"
            self.store.finish_iteration(run_id, iteration, status="FAILED", error=message)
            self._update_state(state="READY", phase="RETRY", last_error=message)
            self.store.append_event(
                "audit",
                "BASELINE_FAILED",
                "基准运行失败",
                message,
                run_id=run_id,
                iteration=iteration,
                level="ERROR",
            )
            raise

    async def shutdown(self) -> None:
        if self._task and not self._task.done():
            self._task.cancel()
            with suppress(asyncio.CancelledError):
                await self._task

    async def _loop(self, run_id: str) -> None:
        failures = 0
        failure_fingerprints: dict[str, int] = {}
        while True:
            state = self._state()
            if state["stop_requested"]:
                self._update_state(state="STOPPED", phase="STOPPED", stop_requested=0)
                self.store.append_event(
                    "audit",
                    "RUN_STOPPED",
                    "持续研究已停止",
                    "停止由用户显式请求触发。",
                    run_id=run_id,
                    iteration=state["iteration"],
                )
                return
            iteration = int(state["iteration"]) + 1
            self._update_state(
                iteration=iteration, state="RUNNING", phase="MEMORY", last_error=None
            )
            self.store.begin_iteration(run_id, iteration)
            try:
                if self.iteration_semaphore is None:
                    await self._run_iteration(run_id, iteration)
                else:
                    async with self.iteration_semaphore:
                        await self._run_iteration(run_id, iteration)
                failures = 0
                failure_fingerprints.clear()
            except asyncio.CancelledError:
                raise
            except Exception as error:
                failures += 1
                message = f"{type(error).__name__}: {error}"
                fingerprint = hashlib.sha256(message.encode()).hexdigest()[:16]
                failure_fingerprints[fingerprint] = failure_fingerprints.get(fingerprint, 0) + 1
                self.store.finish_iteration(run_id, iteration, status="FAILED", error=message)
                record = self.store.iteration_record(run_id, iteration)
                with suppress(KeyError, RuntimeError, ValueError):
                    self._fail_direction_attempt(run_id, iteration, error, record)
                if record and record.get("candidate_id"):
                    with suppress(KeyError):
                        self.store.close_generation_experiment(
                            str(record["candidate_id"]),
                            status="CRASHED",
                            public_verdict=f"ITERATION_FAILED:{fingerprint}",
                        )
                self.store.remember(
                    run_id,
                    iteration,
                    "failure",
                    _failure_memory(message, fingerprint, record),
                )
                self._update_state(state="RETRYING", phase="RETRY", last_error=message)
                self.store.append_event(
                    "audit",
                    "ITERATION_FAILED",
                    "迭代失败，等待重试",
                    message,
                    run_id=run_id,
                    iteration=iteration,
                    level="ERROR",
                    payload={
                        "consecutive_failures": failures,
                        "failure_fingerprint": fingerprint,
                        "same_failure_count": failure_fingerprints[fingerprint],
                        "candidate_id": record.get("candidate_id") if record else None,
                    },
                )
                breaker_reason = _circuit_breaker_reason(
                    error,
                    consecutive_failures=failures,
                    same_failure_count=failure_fingerprints[fingerprint],
                )
                if breaker_reason:
                    self._update_state(state="PAUSED_FAILURE", phase="BLOCKED", last_error=message)
                    self.store.append_event(
                        "audit",
                        "CIRCUIT_BREAKER_OPEN",
                        "研究循环已熔断",
                        f"{breaker_reason}，已暂停模型调用，等待人工检查后重新启动。",
                        run_id=run_id,
                        iteration=iteration,
                        level="ERROR",
                        payload={
                            "consecutive_failures": failures,
                            "failure_fingerprint": fingerprint,
                            "same_failure_count": failure_fingerprints[fingerprint],
                        },
                    )
                    return
                await self._interruptible_sleep(min(60.0, 2.0 ** min(failures, 6)))
                continue
            interval = float(self._settings().get("iteration_interval_seconds", "5"))
            await self._interruptible_sleep(max(0.5, interval))

    async def _run_iteration(self, run_id: str, iteration: int) -> None:
        settings = self._settings()
        evaluator = self._get_evaluator(Path(settings["data_path"]))
        workspace = evaluator.workspace
        config = evaluator.config
        generation = self.store.ensure_running_generation(
            base_generation_id=self._generation_base(config.generation),
            protocol_version=config.governance.protocol_version,
            maximum_candidates=config.budget.max_candidates_per_generation,
            maximum_holdout_attempts=(config.governance.maximum_holdout_evaluations_per_generation),
        )
        generation_id = str(generation["generation_id"])
        if generation.get("recovered_cross_scope_seal"):
            self.store.append_event(
                "audit",
                "GENERATION_SCOPE_RECOVERED",
                "任务世代作用域已自动恢复",
                "检测到并发任务误封存且研究预算尚未耗尽，已恢复当前世代；历史记录保持不变。",
                run_id=run_id,
                iteration=iteration,
                level="WARN",
                payload={
                    "task_id": self.task_id,
                    "generation_id": generation_id,
                    "maximum_candidates": generation["maximum_candidates"],
                    "maximum_holdout_attempts": generation["maximum_holdout_attempts"],
                },
            )
        data_capabilities = evaluator.data_capabilities
        self._update_state(phase="DIRECTION")
        # One snapshot serves the research program context, structure dedup, trial
        # count, LLM library context, redundancy references, and canonical library
        # metrics below; the pool only mutates after register_candidate, so
        # refetching it repeatedly per iteration is waste.
        pool_snapshot = await asyncio.to_thread(self.store.factor_pool, limit=5000)
        behavior_snapshot = await asyncio.to_thread(self._behavior_snapshot)
        research_program = await asyncio.to_thread(
            self._research_program_context,
            iteration,
            evaluator,
            run_id=run_id,
            generation_id=generation_id,
            data_capabilities=data_capabilities,
            pool_records=pool_snapshot,
            behavior_snapshot=behavior_snapshot,
        )
        direction_context = self._prepare_direction_campaign(
            run_id,
            iteration,
            generation_id,
            research_program,
            config,
        )
        research_program["adaptive_direction"] = direction_context
        research_program["research_mandate"] = direction_context["objective"]
        visible_start, visible_end = self._visible_data_range(workspace)
        data_context = {
            "fingerprint": workspace.fingerprint,
            "research_task_id": self.task_id,
            "rows": workspace.rows,
            "symbols": workspace.symbols,
            "first_trade_date": visible_start,
            "last_trade_date": visible_end,
            "available_factor_fields": list(evaluator.factor_fields),
            "field_catalog": data_capabilities["field_catalog"],
            "data_products": data_capabilities["data_products"],
            "data_policy": data_capabilities["policy"],
            "signal_timing": data_capabilities["signal_timing"],
            "extended_data_experiment": research_program["data_experiment"],
            "existing_factor_signatures": _pool_expression_signatures(pool_snapshot, limit=250),
            "price_research_ready": workspace.price_research_ready,
            "source_integrity_passed": workspace.source_integrity_passed,
            "institutional_pit_ready": workspace.institutional_pit_ready,
            "blockers": list(workspace.blockers),
        }
        data_context["research_program"] = research_program
        self.store.append_event(
            "audit",
            "DATA_WORKSPACE_BOUND",
            "研究数据已绑定",
            f"本轮绑定 {workspace.rows:,} 行、{workspace.symbols or 0:,} 只证券。",
            run_id=run_id,
            iteration=iteration,
            payload={
                "root_path": workspace.root_path,
                "panel_path": workspace.panel_path,
                "source_path": workspace.source_path,
                "quality_passed": workspace.quality_passed,
                "source_integrity_passed": workspace.source_integrity_passed,
                "fingerprint": workspace.fingerprint,
                "eligible_factor_fields": data_capabilities["eligible_fields"],
                "eligible_extended_fields": data_capabilities["eligible_extended_fields"],
                "staged_extended_fields": data_capabilities["staged_extended_fields"],
                "data_contract_version": data_capabilities["contract_version"],
            },
        )
        memories = _select_research_memories(
            self.store.recent_memories(limit=500, run_id=run_id),
            limit=40,
            current_protocol=config.governance.protocol_version,
        )
        self.store.append_event(
            "action",
            "MEMORY_LOADED",
            "连续记忆已载入",
            f"已向本轮研究上下文载入 {len(memories)} 条长期记忆。",
            run_id=run_id,
            iteration=iteration,
        )
        self._update_state(phase="PROPOSAL")
        self.store.append_event(
            "action",
            "PROPOSAL_REQUESTED",
            "生成研究提案",
            f"调用模型 {settings['model']} 生成第 {iteration} 轮候选。",
            run_id=run_id,
            iteration=iteration,
        )
        client = CompatibleChatClient(
            base_url=settings["base_url"],
            api_key=self.vault.get(),
            model=settings["model"],
            temperature=float(settings.get("temperature", "0.7")),
        )
        llm_team = FullLLMResearchTeam(client) if self._full_llm_enabled(settings) else None
        try:
            proposal = await self._next_proposal(
                client,
                memories,
                iteration,
                data_context=data_context,
                direction_context=direction_context,
                generation_id=generation_id,
                settings=settings,
            )
        except ModelInvocationError as error:
            self.store.append_event(
                "audit",
                "MODEL_INVOCATION_REJECTED",
                "模型响应未通过契约",
                str(error),
                run_id=run_id,
                iteration=iteration,
                level="ERROR",
                payload={
                    "provider": settings["base_url"],
                    "model": settings["model"],
                    **error.audit_payload(),
                },
            )
            raise
        batch_size = int(proposal.usage.get("batch_size", 1))
        if batch_size > 1:
            batch_position = int(proposal.usage.get("batch_position", 1))
            self.store.append_event(
                "action",
                (
                    "PROPOSAL_BATCH_GENERATED"
                    if batch_position == 1
                    else "PROPOSAL_BATCH_REUSED"
                ),
                "批量研究提案已生成" if batch_position == 1 else "复用冻结方向候选",
                (
                    f"一次生成 {batch_size} 个相互区分的候选，当前使用第 "
                    f"{batch_position} 个；方向改变后未消费候选会自动失效。"
                ),
                run_id=run_id,
                iteration=iteration,
                payload={
                    "batch_size": batch_size,
                    "batch_position": batch_position,
                    "buffered_remaining": len(self._proposal_queue),
                    "campaign_id": direction_context.get("campaign_id"),
                },
            )
        self._update_state(phase="SEMANTICS")
        repair_count = 0
        preflight = None
        while True:
            factor = proposal.factor
            try:
                evaluator.validator.validate(factor.expression)
                candidate_fields = expression_fields(factor.expression)
                _validate_data_direction_alignment(
                    candidate_fields,
                    direction_context=direction_context,
                )
                candidate_mechanism = classify_mechanism(
                    fields=candidate_fields,
                    family=factor.family,
                    name=factor.name,
                    hypothesis=factor.hypothesis,
                )
                _validate_mechanism_direction_alignment(
                    candidate_mechanism,
                    direction_context=direction_context,
                )
                if self.store.candidate_exists(factor.factor_id):
                    raise ValueError(f"Duplicate candidate expression: {factor.factor_id}")
                duplicate_structure = _matching_structure(
                    factor.expression.to_dict(), pool_snapshot
                )
                if duplicate_structure:
                    raise ValueError(
                        "Parameter-only duplicate of "
                        f"{duplicate_structure['factor_id']} ({duplicate_structure['name']})"
                    )
                preflight = await asyncio.to_thread(evaluator.preflight, factor)
                if not preflight.passed:
                    raise ValueError(
                        "Candidate signal preflight failed: "
                        + ", ".join(preflight.failures)
                    )
                break
            except (TypeError, ValueError) as error:
                if repair_count >= 2:
                    raise
                repair_count += 1
                self.store.append_event(
                    "action",
                    "PROPOSAL_REPAIR_REQUESTED",
                    "候选表达式请求修复",
                    f"第 {repair_count} 次修复：{type(error).__name__}: {error}",
                    run_id=run_id,
                    iteration=iteration,
                    level="WARN",
                    payload={
                        "candidate_id": factor.factor_id,
                        "repair_attempt": repair_count,
                        "reason": str(error),
                        "expression": factor.expression.to_dict(),
                    },
                )
                proposal = await client.repair(
                    proposal,
                    f"{type(error).__name__}: {error}",
                    memories,
                    iteration,
                    data_context=data_context,
                )
        assert preflight is not None
        self.store.append_event(
            "research",
            "SIGNAL_PREFLIGHT_COMPLETED",
            "候选信号预检完成",
            (
                "候选通过覆盖、截面有效性和退化检查。"
                if not preflight.warnings
                else f"候选通过预检，保留诊断提示：{', '.join(preflight.warnings)}。"
            ),
            run_id=run_id,
            iteration=iteration,
            payload={
                **preflight.metrics,
                "passed": preflight.passed,
                "failures": list(preflight.failures),
                "warnings": list(preflight.warnings),
                "research_budget_consumed": False,
            },
        )
        generation = self.store.reserve_generation_experiment(
            candidate_hash=factor.factor_id,
            generation_id=generation_id,
            factor_id=factor.factor_id,
            family=factor.family,
            iteration=iteration,
            maximum_family_candidates=config.budget.max_candidates_per_family,
        )
        evaluator.set_trial_count(len(pool_snapshot) + 1)
        self.store.append_event(
            "audit",
            "MODEL_INVOCATION",
            "模型调用已审计",
            f"{settings['model']} 返回候选提案。",
            run_id=run_id,
            iteration=iteration,
            payload={
                "provider": settings["base_url"],
                "model": settings["model"],
                "prompt_hash": proposal.prompt_hash,
                "response_hash": proposal.response_hash,
                "usage": proposal.usage,
                "repair_count": repair_count,
            },
        )
        canonical_proposal = {
            **proposal.raw,
            "expected_direction": factor.expected_direction,
            "expression": factor.expression.to_dict(),
            "data_lineage": {
                "factor_fields": sorted(candidate_fields),
                "source_products": products_for_fields(candidate_fields),
                "signal_timing": data_capabilities["signal_timing"],
                "data_contract_version": data_capabilities["contract_version"],
                "signal_preflight": {
                    **preflight.metrics,
                    "warnings": list(preflight.warnings),
                },
            },
            "canonical_mechanism": candidate_mechanism,
        }
        extended_fields_used = sorted(
            candidate_fields & set(data_capabilities["eligible_extended_fields"])
        )
        if extended_fields_used:
            self.store.append_event(
                "research",
                "EXTENDED_DATA_EXPERIMENT_STARTED",
                "扩展数据候选进入实验",
                f"候选使用已解锁字段：{', '.join(extended_fields_used)}。",
                run_id=run_id,
                iteration=iteration,
                payload={
                    "candidate_id": factor.factor_id,
                    "factor_fields": sorted(candidate_fields),
                    "extended_fields": extended_fields_used,
                    "source_products": products_for_fields(candidate_fields),
                    "next_session_timing_enforced": True,
                },
            )
        # The deterministic evaluation does not depend on the advisory pre-review
        # roles, so both run concurrently to cut iteration wall-clock time.
        self.store.append_event(
            "action",
            "EVALUATION_STARTED",
            "执行确定性评估",
            f"候选 {factor.factor_id} 已进入截面研究评估器。",
            run_id=run_id,
            iteration=iteration,
        )
        evaluation_task = asyncio.create_task(asyncio.to_thread(evaluator.evaluate, factor))
        pre_role_outcomes: dict[str, RoleOutcome] = {}
        role_routing = _pre_evaluation_role_route(
            canonical_proposal,
            research_program=research_program,
            preflight_warnings=set(preflight.warnings),
        )
        try:
            if llm_team is not None:
                self._update_state(phase="LLM_PRE_REVIEW")
                library_peers = _retrieve_library_peers(
                    canonical_proposal,
                    pool_snapshot,
                    behavior_snapshot.get("factors", {}),
                    limit=12,
                )
                pre_role_outcomes = await llm_team.pre_evaluation(
                    candidate=canonical_proposal,
                    library_context=library_peers,
                    data_context=data_context,
                    roles=tuple(role_routing["roles"]),
                )
                self._persist_role_outcomes(
                    pre_role_outcomes.values(),
                    run_id=run_id,
                    iteration=iteration,
                    candidate_id=factor.factor_id,
                )
                canonical_proposal["full_llm"] = {
                    role: {
                        "status": outcome.status,
                        "artifact": outcome.artifact,
                    }
                    for role, outcome in pre_role_outcomes.items()
                }
                canonical_proposal["full_llm_routing"] = role_routing
            self.store.stage_iteration_candidate(
                run_id,
                iteration,
                candidate_id=factor.factor_id,
                proposal=canonical_proposal,
            )
            self.store.append_event(
                "research",
                "PROPOSAL_CREATED",
                factor.name,
                factor.hypothesis,
                run_id=run_id,
                iteration=iteration,
                payload={
                    "candidate_id": factor.factor_id,
                    "family": factor.family,
                    "change": proposal.change,
                    "expected": proposal.expected,
                    "expression": factor.expression.to_dict(),
                    "data_lineage": canonical_proposal["data_lineage"],
                    "usage": proposal.usage,
                },
            )
            self._update_state(phase="EVALUATION")
            result = await evaluation_task
        except BaseException:
            if not evaluation_task.done():
                evaluation_task.cancel()
                with suppress(BaseException):
                    await evaluation_task
            raise
        result.metrics["research_generation"] = generation_id
        result.metrics.update(
            _multiple_testing_adjustments(
                result.metrics,
                self.store.metric_history(
                    limit=config.budget.max_candidates_per_generation, run_id=run_id
                ),
                generation=generation_id,
                alpha=config.evaluation.maximum_net_return_p_value,
            )
        )
        redundancy_metrics = await asyncio.to_thread(
            evaluator.library_signal_correlation,
            factor,
            _redundancy_references(
                pool_snapshot,
                exclude_id=factor.factor_id,
                behavior_factors=behavior_snapshot.get("factors", {}),
                candidate_fields=candidate_fields,
                candidate_mechanism=candidate_mechanism,
                limit=48,
            ),
            max_references=48,
        )
        redundancy_metrics.update(
            _online_behavior_assignment(
                redundancy_metrics,
                behavior_snapshot.get("factors", {}),
                threshold=min(
                    config.evaluation.maximum_library_correlation,
                    float(behavior_snapshot.get("cluster_threshold", 0.74) or 0.74),
                ),
            )
        )
        result.metrics.update(redundancy_metrics)
        library_metrics = await asyncio.to_thread(
            self._canonical_library_metrics,
            Path(settings["data_path"]),
            factor,
            result.metrics,
            pool_snapshot,
        )
        library_metrics.update(redundancy_metrics)
        self._update_state(phase="PORTFOLIO")
        engine = MultiFactorResearchEngine(
            self.store,
            evaluator,
            evaluator.config,
            maximum_factors=int(settings.get("maximum_active_factors", "5")),
            run_id=run_id,
            source_task_id=self.task_id,
        )
        imported = engine.sync_legacy_pool()
        bootstrap = await asyncio.to_thread(engine.bootstrap_champion, run_id, iteration)
        if imported or bootstrap:
            self.store.append_event(
                "audit",
                "MULTIFACTOR_STATE_INITIALIZED",
                "多因子研究状态已初始化",
                f"导入 {imported} 个历史因子，并建立确定性冠军组合。",
                run_id=run_id,
                iteration=iteration,
                payload={
                    "imported_factors": imported,
                    "bootstrap_factor_id": bootstrap.candidate_id if bootstrap else None,
                },
            )
        candidate_eligible = engine.register_candidate(
            factor,
            iteration,
            canonical_proposal,
            library_metrics,
            task_metrics=result.metrics,
        )
        self.store.append_event(
            "research",
            "FACTOR_POOL_UPDATED",
            "因子池已更新",
            f"{factor.name} 已进入{'合格' if candidate_eligible else '观察'}因子池。",
            run_id=run_id,
            iteration=iteration,
            payload={
                "candidate_id": factor.factor_id,
                "eligible": candidate_eligible,
                "single_factor_metrics": library_metrics,
                "source_task_metrics": result.metrics,
            },
        )
        portfolio_role_outcome: RoleOutcome | None = None
        if llm_team is not None and candidate_eligible:
            self._update_state(phase="LLM_PORTFOLIO_REVIEW")
            portfolio_role_outcome = await llm_team.portfolio_advisory(
                candidate=canonical_proposal,
                librarian=pre_role_outcomes.get(
                    FACTOR_LIBRARIAN,
                    RoleOutcome(
                        role=FACTOR_LIBRARIAN,
                        stage="PRE_EVALUATION",
                        status="FAILED",
                        artifact={},
                        usage={},
                        prompt_hash=None,
                        response_hash=None,
                    ),
                ).artifact,
                active_members=research_program["active_portfolio"]["members"],
                public_feedback=categorical_research_feedback(
                    result.metrics,
                    decision="PUBLIC_SINGLE_FACTOR_EVALUATED",
                    candidate_eligible=candidate_eligible,
                    portfolio_action="PENDING_DETERMINISTIC_DECISION",
                    portfolio_accepted=False,
                ),
            )
            self._persist_role_outcomes(
                [portfolio_role_outcome],
                run_id=run_id,
                iteration=iteration,
                candidate_id=factor.factor_id,
            )
        portfolio_decision = await asyncio.to_thread(
            engine.decide, factor, candidate_eligible=candidate_eligible
        )
        direction_metrics = self._complete_direction_campaign_attempt(
            run_id=run_id,
            iteration=iteration,
            candidate_id=factor.factor_id,
            candidate_eligible=candidate_eligible,
            portfolio_decision=portfolio_decision,
            direction_context=direction_context,
            config=config,
            candidate_fields=candidate_fields,
        )
        governance_metrics: dict[str, Any] = {
            "research_generation": generation_id,
            "evaluation_protocol": config.governance.protocol_version,
            "public_transition_gate_passed": portfolio_decision.accepted,
            "public_portfolio_action": portfolio_decision.action,
            "holdout_feedback_policy": "categorical_verdict_and_evidence_hash_only",
            "holdout_verdict": "NOT_EVALUATED_PUBLIC_GATES_FAILED",
            "capital_verdict": "NOT_EVALUATED_HOLDOUT_REQUIRED",
            "adaptive_direction": direction_metrics,
            "research_evidence_tier": evaluator.research_evidence_tier,
        }
        feasibility_recovery = _is_public_feasibility_recovery(portfolio_decision)
        governance_metrics["public_protocol_gate_passed"] = bool(
            portfolio_decision.accepted and not feasibility_recovery
        )
        governance_metrics["public_feasibility_recovery"] = feasibility_recovery
        experiment_status = "REJECTED"
        experiment_verdict = "PUBLIC_GATES_FAILED"
        holdout_available = int(generation["holdout_attempts"]) < int(
            generation["maximum_holdout_attempts"]
        )
        proposed_factor_ids = [item.factor_id for item in portfolio_decision.factors]
        contamination = self.store.portfolio_contamination(generation_id, proposed_factor_ids)
        if portfolio_decision.accepted and evaluator.research_evidence_tier != "PRIMARY_DISCOVERY":
            experiment_status = "RETAINED"
            experiment_verdict = "REGIME_SLICE_RESEARCH_ONLY"
            governance_metrics["holdout_verdict"] = "NOT_EVALUATED_REGIME_SLICE_ONLY"
            governance_metrics["capital_verdict"] = "NOT_EVALUATED_PRIMARY_DISCOVERY_REQUIRED"
            self.store.append_event(
                "audit",
                "REGIME_SLICE_PROMOTION_BLOCKED",
                "短窗口任务仅保留状态切片证据",
                "候选可以入库并参与边际研究，但不能访问盲测或成为生产晋级依据。",
                run_id=run_id,
                iteration=iteration,
                payload={
                    "task_id": self.task_id,
                    "research_evidence_tier": evaluator.research_evidence_tier,
                    "holdout_accessed": False,
                },
            )
        elif feasibility_recovery:
            experiment_status = "RETAINED"
            experiment_verdict = "PUBLIC_FEASIBILITY_RECOVERY"
            governance_metrics["holdout_verdict"] = "NOT_EVALUATED_FEASIBILITY_RECOVERY_INCOMPLETE"
            governance_metrics["capital_verdict"] = "NOT_EVALUATED_ABSOLUTE_PUBLIC_GATES_REQUIRED"
            self.store.append_event(
                "audit",
                "PUBLIC_FEASIBILITY_RECOVERY_RETAINED",
                "公开区可行性恢复已保留",
                "组合仍有绝对门禁缺口，本轮不访问隔离盲测。",
                run_id=run_id,
                iteration=iteration,
                payload={
                    "action": portfolio_decision.action,
                    "candidate_id": portfolio_decision.candidate_id,
                    "unresolved_absolute_gates": portfolio_decision.evaluation.metrics.get(
                        "portfolio_proposed_absolute_failures", []
                    ),
                    "holdout_accessed": False,
                    "holdout_budget_consumed": False,
                },
            )
        elif portfolio_decision.accepted and contamination:
            governance_metrics["holdout_verdict"] = "MANUAL_HOLDOUT_CONTAMINATION"
            governance_metrics["manual_holdout_contamination"] = {
                "factor_ids": sorted({str(item["factor_id"]) for item in contamination}),
                "evidence_hashes": sorted({str(item["evidence_hash"]) for item in contamination}),
            }
            portfolio_decision = engine.governance_veto(
                portfolio_decision,
                gate="MANUAL_HOLDOUT_CONTAMINATION",
                reason="MANUAL_HOLDOUT_CONTAMINATION_REQUIRES_FRESH_UNEXPOSED_HOLDOUT",
            )
            experiment_verdict = "MANUAL_HOLDOUT_CONTAMINATION"
            self.store.append_event(
                "audit",
                "CONTAMINATED_PORTFOLIO_HOLDOUT_BLOCKED",
                "人工暴露组合已阻断盲测",
                "组合中的因子已人工查看当前隐藏期；更换世代不能清除污染，必须使用新的未暴露测试期。",
                run_id=run_id,
                iteration=iteration,
                level="WARN",
                payload=governance_metrics["manual_holdout_contamination"],
            )
        elif portfolio_decision.accepted and not holdout_available:
            governance_metrics["holdout_verdict"] = "HOLDOUT_BUDGET_EXHAUSTED"
            portfolio_decision = engine.governance_veto(
                portfolio_decision,
                gate="HOLDOUT_BUDGET",
                reason="HOLDOUT_BUDGET_EXHAUSTED",
            )
            experiment_verdict = "HOLDOUT_BUDGET_EXHAUSTED"
        elif portfolio_decision.accepted:
            active_components = engine.active_components()
            benchmark_factors = [component for component, _ in active_components]
            benchmark_weights = tuple(weight for _, weight in active_components)
            frozen_hash = _frozen_portfolio_hash(
                portfolio_decision.factors,
                portfolio_decision.weights,
                generation=generation_id,
            )
            self._update_state(phase="HOLDOUT")
            boundary = BlindEvaluationBoundary(evaluator.panel_path, config)
            holdout = await asyncio.to_thread(
                boundary.evaluate_holdout,
                list(portfolio_decision.factors),
                portfolio_decision.weights,
                benchmark_factors=benchmark_factors or None,
                benchmark_weights=benchmark_weights or None,
            )
            self.store.record_blind_evaluation(
                candidate_hash=frozen_hash,
                generation_id=generation_id,
                iteration=iteration,
                holdout_verdict=holdout.verdict,
                holdout_passed=holdout.passed,
                holdout_evidence_hash=holdout.evidence_hash,
            )
            governance_metrics.update(
                {
                    "frozen_portfolio_hash": frozen_hash,
                    "holdout_verdict": holdout.verdict,
                    "holdout_passed": holdout.passed,
                    "holdout_evidence_hash": holdout.evidence_hash,
                }
            )
            self.store.append_event(
                "audit",
                "BLIND_HOLDOUT_EVALUATED",
                "隔离盲测已执行",
                holdout.verdict,
                run_id=run_id,
                iteration=iteration,
                payload={
                    "frozen_portfolio_hash": frozen_hash,
                    "verdict": holdout.verdict,
                    "evidence_hash": holdout.evidence_hash,
                    "feedback_policy": "no_exact_metrics",
                },
            )
            if not holdout.passed:
                portfolio_decision = engine.governance_veto(
                    portfolio_decision,
                    gate="HOLDOUT",
                    reason=holdout.verdict,
                )
                experiment_verdict = holdout.verdict
            else:
                self._update_state(phase="CAPITAL_SIMULATION")
                capital = await asyncio.to_thread(
                    boundary.evaluate_capital,
                    list(portfolio_decision.factors),
                    portfolio_decision.weights,
                )
                self.store.update_blind_capital(
                    frozen_hash,
                    capital_verdict=capital.verdict,
                    capital_passed=capital.passed,
                    capital_evidence_hash=capital.evidence_hash,
                )
                governance_metrics.update(
                    {
                        "capital_verdict": capital.verdict,
                        "capital_passed": capital.passed,
                        "capital_evidence_hash": capital.evidence_hash,
                    }
                )
                self.store.append_event(
                    "audit",
                    "CAPITAL_SIMULATION_EVALUATED",
                    "最终资金仿真已执行",
                    capital.verdict,
                    run_id=run_id,
                    iteration=iteration,
                    payload={
                        "frozen_portfolio_hash": frozen_hash,
                        "verdict": capital.verdict,
                        "evidence_hash": capital.evidence_hash,
                        "feedback_policy": "no_exact_metrics",
                    },
                )
                if not capital.passed:
                    portfolio_decision = engine.governance_veto(
                        portfolio_decision,
                        gate="CAPITAL_SIMULATION",
                        reason=capital.verdict,
                    )
                    experiment_verdict = capital.verdict
                else:
                    experiment_status = "RETAINED"
                    experiment_verdict = "APPROVED_FOR_PAPER_RESEARCH_ONLY_DATA_BLOCKED"
        portfolio_version_id = engine.persist(run_id, iteration, portfolio_decision)
        portfolio_metrics = {
            **portfolio_decision.evaluation.metrics,
            "portfolio_action": portfolio_decision.action,
            "portfolio_action_accepted": portfolio_decision.accepted,
            "portfolio_action_reason": portfolio_decision.reason,
            "portfolio_action_gate_failures": list(portfolio_decision.failed_gates),
            "portfolio_utility_change": portfolio_decision.utility_change,
            "portfolio_selection_policy": "hard_gates_then_lexicographic_robustness",
            "portfolio_version_id": portfolio_version_id,
            "portfolio_active_factor_ids": [
                active.factor_id for active in portfolio_decision.factors
            ],
            "portfolio_active_factor_weights": {
                active.factor_id: weight
                for active, weight in zip(
                    portfolio_decision.factors,
                    portfolio_decision.weights,
                    strict=True,
                )
            },
            "portfolio_removed_factor_id": portfolio_decision.removed_factor_id,
            "portfolio_option_diagnostics": list(portfolio_decision.option_diagnostics),
            "portfolio_factor_correlations": (portfolio_decision.evaluation.factor_correlations),
        }
        if feasibility_recovery:
            decision = "PUBLIC_FEASIBILITY_RECOVERY_RESEARCH_ONLY_DATA_BLOCKED"
        elif portfolio_decision.accepted:
            decision = "APPROVED_FOR_PAPER_RESEARCH_ONLY_DATA_BLOCKED"
        else:
            decision = "MULTIFACTOR_HOLD_RESEARCH_ONLY_DATA_BLOCKED"
        combined_metrics = {**result.metrics, **portfolio_metrics, **governance_metrics}
        full_llm_summary: dict[str, Any] = {
            role: {"status": outcome.status, "stage": outcome.stage}
            for role, outcome in pre_role_outcomes.items()
        }
        if portfolio_role_outcome is not None:
            full_llm_summary[PORTFOLIO_RESEARCHER] = {
                "status": portfolio_role_outcome.status,
                "stage": portfolio_role_outcome.stage,
            }
        if llm_team is not None:
            self._update_state(phase="LLM_POST_REVIEW")
            falsification_plan = pre_role_outcomes.get(
                FALSIFICATION_DESIGNER,
                RoleOutcome(
                    role=FALSIFICATION_DESIGNER,
                    stage="PRE_EVALUATION",
                    status="FAILED",
                    artifact={},
                    usage={},
                    prompt_hash=None,
                    response_hash=None,
                ),
            ).artifact
            falsification_results = evaluate_falsification_plan(
                falsification_plan, combined_metrics
            )
            public_feedback = categorical_research_feedback(
                combined_metrics,
                decision=decision,
                candidate_eligible=candidate_eligible,
                portfolio_action=portfolio_decision.action,
                portfolio_accepted=portfolio_decision.accepted,
            )
            post_route = _post_evaluation_role_route(
                combined_metrics,
                candidate_eligible=candidate_eligible,
                portfolio_accepted=portfolio_decision.accepted,
            )
            post_role_outcomes = (
                await llm_team.post_evaluation(
                    candidate=canonical_proposal,
                    public_feedback=public_feedback,
                    falsification_plan=falsification_plan,
                    falsification_results=falsification_results,
                    portfolio_advisory=(
                        portfolio_role_outcome.artifact if portfolio_role_outcome else {}
                    ),
                    roles=tuple(post_route["roles"]),
                )
                if post_route["roles"]
                else {}
            )
            self._persist_role_outcomes(
                post_role_outcomes.values(),
                run_id=run_id,
                iteration=iteration,
                candidate_id=factor.factor_id,
            )
            full_llm_summary.update(
                {
                    role: {"status": outcome.status, "stage": outcome.stage}
                    for role, outcome in post_role_outcomes.items()
                }
            )
            role_routing["post_evaluation"] = post_route
            reviewer = pre_role_outcomes.get(REVIEWER)
            librarian = pre_role_outcomes.get(FACTOR_LIBRARIAN)
            if librarian is not None and librarian.status == "COMPLETED":
                try:
                    self.store.upsert_factor_knowledge(
                        factor_id=factor.factor_id,
                        canonical_mechanism=str(
                            librarian.artifact.get("canonical_mechanism", "OTHER_INTERPRETABLE")
                        ),
                        mechanism_summary=str(librarian.artifact.get("mechanism_summary", "")),
                        tags=list(librarian.artifact.get("tags", [])),
                        review=reviewer.artifact if reviewer else {},
                        falsification={
                            "plan": falsification_plan,
                            "results": falsification_results,
                        },
                        related_factors=list(librarian.artifact.get("related_factors", [])),
                    )
                except (KeyError, TypeError, ValueError) as error:
                    self.store.append_event(
                        "audit",
                        "LLM_KNOWLEDGE_ARCHIVE_FAILED",
                        "LLM 知识档案降级",
                        "知识图谱制品格式异常，确定性研究结果照常交付。",
                        run_id=run_id,
                        iteration=iteration,
                        level="WARN",
                        payload={
                            "candidate_id": factor.factor_id,
                            "error_type": type(error).__name__,
                            "decision_authority": "ADVISORY_ONLY",
                        },
                    )
        combined_metrics["full_llm"] = {
            "enabled": llm_team is not None,
            "roles": full_llm_summary,
            "routing": role_routing if llm_team is not None else {},
            "decision_authority": "ADVISORY_ONLY",
            "feedback_policy": "CATEGORICAL_PUBLIC_ONLY_NO_EXACT_METRICS",
        }
        combined_metrics["planner_features"] = {
            "version": "STRUCTURED_RESEARCH_FEEDBACK_V1",
            "candidate_mechanism": candidate_mechanism,
            "candidate_fields": sorted(candidate_fields),
            "extended_fields": extended_fields_used,
            "signal_preflight": {
                "passed": preflight.passed,
                "failures": list(preflight.failures),
                "warnings": list(preflight.warnings),
                **preflight.metrics,
            },
            "behavior_assignment": {
                key: value
                for key, value in redundancy_metrics.items()
                if key.startswith(("library_signal_", "online_behavior_"))
            },
            "single_factor_gate_failures": list(
                result.metrics.get("exploratory_gate_failures", [])
            ),
            "portfolio_gate_failures": list(portfolio_decision.failed_gates),
            "portfolio_action": portfolio_decision.action,
            "portfolio_accepted": portfolio_decision.accepted,
            "adaptive_direction": direction_metrics,
            "llm_role_routing": role_routing if llm_team is not None else {},
        }
        self.store.close_generation_experiment(
            factor.factor_id,
            status=experiment_status,
            public_verdict=experiment_verdict,
        )
        self.store.append_event(
            "research",
            "PORTFOLIO_ACTION_EVALUATED",
            f"组合动作 {portfolio_decision.action}",
            portfolio_decision.reason,
            run_id=run_id,
            iteration=iteration,
            payload={
                "portfolio_version_id": portfolio_version_id,
                "accepted": portfolio_decision.accepted,
                "candidate_id": portfolio_decision.candidate_id,
                "removed_factor_id": portfolio_decision.removed_factor_id,
                "active_factor_ids": portfolio_metrics["portfolio_active_factor_ids"],
                "active_factor_weights": portfolio_metrics["portfolio_active_factor_weights"],
                "failed_gates": list(portfolio_decision.failed_gates),
                "option_diagnostics": list(portfolio_decision.option_diagnostics),
                **portfolio_decision.evaluation.metrics,
            },
        )
        self._update_state(phase="EVIDENCE")
        artifact_payload = json.dumps(
            {
                "run_id": run_id,
                "iteration": iteration,
                "candidate_id": factor.factor_id,
                "proposal": canonical_proposal,
                "metrics": combined_metrics,
                "decision": decision,
                "observations": result.observations,
                "data_workspace": data_context,
            },
            ensure_ascii=False,
            sort_keys=True,
        ).encode()
        self._update_state(phase="DELIVERY")
        artifact = self.artifacts.publish(
            "research-evaluation",
            artifact_payload,
            owner="autoalpha-service",
            metadata={
                "run_id": run_id,
                "iteration": iteration,
                "data_fingerprint": workspace.fingerprint,
            },
        )
        self.store.finish_iteration(
            run_id,
            iteration,
            status="COMPLETED",
            candidate_id=factor.factor_id,
            proposal=canonical_proposal,
            metrics=combined_metrics,
            decision=decision,
        )
        memory = _memory_summary(
            factor.name,
            factor.family,
            combined_metrics,
            decision,
            proposal=canonical_proposal,
        )
        self.store.remember(run_id, iteration, "evaluation", memory)
        self.store.append_event(
            "research",
            "EVALUATION_COMPLETED",
            "评价矩阵已更新",
            f"{factor.name} 完成 {result.observations} 个交易日的确定性评估。",
            run_id=run_id,
            iteration=iteration,
            payload={"candidate_id": factor.factor_id, **combined_metrics},
        )
        self.store.append_event(
            "delivery",
            "ARTIFACT_PUBLISHED",
            "研究制品已交付",
            f"不可变制品 {artifact.artifact_id} 已写入注册表。",
            run_id=run_id,
            iteration=iteration,
            payload={
                "artifact_id": artifact.artifact_id,
                "content_hash": artifact.content_hash,
                "decision": decision,
            },
        )
        self._update_state(phase="WAITING")

    def _state(self) -> dict[str, Any]:
        if self.task_id == "legacy-ashare":
            return self.store.state()
        return self.store.research_task_state(self.task_id)

    def _update_state(self, **values: Any) -> dict[str, Any]:
        if self.task_id == "legacy-ashare":
            return self.store.update_state(**values)
        return self.store.update_research_task_state(self.task_id, **values)

    def _settings(self) -> dict[str, str]:
        settings = self.store.settings()
        if self.task_id == "legacy-ashare":
            return settings
        task = self.store.research_task(self.task_id)
        if task is None:
            raise RuntimeError(f"Research task no longer exists: {self.task_id}")
        return {**settings, "data_path": str(task["data_path"])}

    @staticmethod
    def _full_llm_enabled(settings: dict[str, str]) -> bool:
        configured = settings.get("full_llm_enabled", os.getenv("AUTOALPHA_FULL_LLM", "1"))
        return str(configured).strip().lower() in {"1", "true", "yes", "on"}

    async def _next_proposal(
        self,
        client: CompatibleChatClient,
        memories: list[dict[str, Any]],
        iteration: int,
        *,
        data_context: dict[str, Any],
        direction_context: dict[str, Any],
        generation_id: str,
        settings: dict[str, str],
    ) -> GeneratedProposal:
        campaign_key = ":".join(
            (
                generation_id,
                str(direction_context.get("campaign_id", "UNCONSTRAINED")),
                str(direction_context.get("direction", "UNCONSTRAINED")),
            )
        )
        if campaign_key != self._proposal_queue_key:
            self._proposal_queue.clear()
            self._proposal_queue_key = campaign_key
        if self._proposal_queue:
            return self._proposal_queue.pop(0)

        configured = settings.get(
            "proposal_batch_size",
            os.getenv("AUTOALPHA_PROPOSAL_BATCH_SIZE", "3"),
        )
        try:
            requested = int(configured)
        except (TypeError, ValueError):
            requested = 3
        remaining = int(direction_context.get("remaining_after_this_attempt", 0) or 0) + 1
        batch_size = min(max(requested, 1), max(remaining, 1), 3)
        proposals = (
            await client.propose_batch(
                memories,
                iteration,
                batch_size=batch_size,
                data_context=data_context,
            )
            if batch_size > 1
            else [await client.propose(memories, iteration, data_context=data_context)]
        )
        self._proposal_queue.extend(proposals)
        return self._proposal_queue.pop(0)

    def _persist_role_outcomes(
        self,
        outcomes: Any,
        *,
        run_id: str,
        iteration: int,
        candidate_id: str,
    ) -> None:
        for outcome in outcomes:
            self.store.record_llm_role_artifact(
                task_id=self.task_id,
                run_id=run_id,
                iteration=iteration,
                candidate_id=candidate_id,
                role=outcome.role,
                stage=outcome.stage,
                status=outcome.status,
                artifact=outcome.artifact,
                usage=outcome.usage,
                prompt_hash=outcome.prompt_hash,
                response_hash=outcome.response_hash,
                error=outcome.error,
            )
            self.store.append_event(
                "audit",
                "LLM_ROLE_COMPLETED" if outcome.status == "COMPLETED" else "LLM_ROLE_FAILED",
                f"LLM 角色：{outcome.role}",
                (
                    "角色建议已归档，确定性门禁与组合引擎不受其直接控制。"
                    if outcome.status == "COMPLETED"
                    else "角色调用失败，本轮按失效开放策略继续确定性研究。"
                ),
                run_id=run_id,
                iteration=iteration,
                level="INFO" if outcome.status == "COMPLETED" else "WARN",
                payload={
                    "task_id": self.task_id,
                    "candidate_id": candidate_id,
                    "role": outcome.role,
                    "stage": outcome.stage,
                    "status": outcome.status,
                    "usage": outcome.usage,
                    "prompt_hash": outcome.prompt_hash,
                    "response_hash": outcome.response_hash,
                    "error": outcome.error,
                    "decision_authority": "ADVISORY_ONLY",
                },
            )

    def _generation_base(self, generation: str) -> str:
        if self.task_id == "legacy-ashare":
            return generation
        return f"{generation}--{self.task_id}"

    def _visible_data_range(self, workspace: Any) -> tuple[str | None, str | None]:
        task = self.store.research_task(self.task_id) or {}
        protocol = task.get("protocol") or {}
        return (
            protocol.get("exploration_start", workspace.first_trade_date),
            protocol.get("validation_end", workspace.last_trade_date),
        )

    def _research_config(self) -> ResearchConfig:
        base = ResearchConfig.from_toml(self.config_path)
        task = self.store.research_task(self.task_id)
        if task is None or not task.get("protocol"):
            return base
        return task_research_config(base, task["protocol"], task_id=self.task_id)

    def _get_evaluator(self, data_path: Path) -> PriceVolumeEvaluator:
        config = self._research_config()
        if (
            self._evaluator is None
            or self._evaluator.data_path != data_path
            or self._evaluator.config.governance.protocol_version
            != config.governance.protocol_version
        ):
            self._evaluator = PriceVolumeEvaluator(data_path, config=config)
        return self._evaluator

    def _canonical_library_metrics(
        self,
        data_path: Path,
        factor: FactorDefinition,
        source_task_metrics: dict[str, Any],
        pool_records: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        evaluator = self._get_canonical_evaluator(data_path)
        records = pool_records if pool_records is not None else self.store.factor_pool(limit=5000)
        metrics = evaluate_canonical_library_factor(
            evaluator,
            factor,
            trials=len(records) + 1,
            source_task_metrics=source_task_metrics,
        )
        metrics["research_generation"] = f"canonical-ingest--{self.task_id}"
        metrics.update(
            _multiple_testing_adjustments(
                metrics,
                [record.get("metrics", {}) for record in records],
                generation=str(metrics["research_generation"]),
                alpha=evaluator.config.evaluation.maximum_net_return_p_value,
            )
        )
        return metrics

    def _get_canonical_evaluator(self, data_path: Path) -> PriceVolumeEvaluator:
        base = ResearchConfig.from_toml(self.config_path)
        task_evaluator = self._get_evaluator(data_path)
        config = canonical_library_config(
            base,
            data_start=date.fromisoformat(task_evaluator.workspace.first_trade_date),
        )
        if (
            self._canonical_evaluator is None
            or self._canonical_evaluator.data_path != data_path
            or self._canonical_evaluator.config.governance.protocol_version
            != CANONICAL_LIBRARY_PROTOCOL
        ):
            self._canonical_evaluator = PriceVolumeEvaluator(data_path, config=config)
        return self._canonical_evaluator

    def _behavior_snapshot(self) -> dict[str, Any]:
        root = self.artifacts.root
        artifacts_root = next(
            (candidate for candidate in (root, *root.parents) if candidate.name == "artifacts"),
            None,
        )
        behavior_root = (
            artifacts_root.parent / "factor-behavior"
            if artifacts_root is not None
            else root.parent / "factor-behavior"
        )
        return load_behavior_snapshot(behavior_root)

    def _prepare_direction_campaign(
        self,
        run_id: str,
        iteration: int,
        generation_id: str,
        research_program: dict[str, Any],
        config: ResearchConfig,
    ) -> dict[str, Any]:
        adaptive = config.adaptive_direction
        if not adaptive.enabled:
            return {
                "enabled": False,
                "direction": "UNCONSTRAINED",
                "objective": "Adaptive direction campaigns are disabled by frozen config.",
            }

        campaign = self.store.active_direction_campaign(generation_id)
        if campaign is None:
            history = self.store.direction_campaign_history(generation_id, limit=50)
            closed = [item for item in history if item["status"] != "ACTIVE"]
            blocked = {str(item["direction"]) for item in closed[: adaptive.cooldown_campaigns]}
            recent = [
                item
                for item in self.store.metric_history(
                    limit=adaptive.recent_candidate_window * 3, run_id=run_id
                )
                if item.get("research_generation") == generation_id
            ][-adaptive.recent_candidate_window :]
            baseline = research_program["active_portfolio"]["metrics"]
            plan = diagnose_direction(
                baseline,
                recent,
                blocked_directions=blocked,
                config=config,
                data_experiment=research_program.get("data_experiment"),
                campaign_history=history,
            )
            plan_payload = plan.to_dict()
            campaign = self.store.start_direction_campaign(
                generation_id=generation_id,
                direction=plan.definition.direction,
                title=plan.definition.title,
                objective=plan.definition.objective,
                diagnostic_score=plan.score,
                rationale=list(plan.rationale),
                evidence=plan.evidence,
                baseline=baseline,
                maximum_attempts=adaptive.maximum_attempts_per_campaign,
                started_iteration=iteration,
            )
            self.store.append_event(
                "audit",
                "DIRECTION_CAMPAIGN_STARTED",
                f"自适应方向已冻结：{campaign['title']}",
                campaign["objective"],
                run_id=run_id,
                iteration=iteration,
                payload={
                    **plan_payload,
                    "campaign_id": campaign["id"],
                    "maximum_attempts": campaign["maximum_attempts"],
                    "holdout_feedback_used": False,
                },
            )

        attempt = self.store.reserve_direction_attempt(
            campaign_id=int(campaign["id"]),
            iteration=iteration,
            baseline=research_program["active_portfolio"]["metrics"],
            run_id=run_id,
        )
        campaign = self.store.direction_campaign(int(campaign["id"]))
        assert campaign is not None
        definition = direction_definition(str(campaign["direction"]))
        data_experiment = research_program.get("data_experiment", {})
        required_factor_fields = (
            list(data_experiment.get("eligible_extended_fields", []))
            if campaign["direction"] == "EXPLORE_EXTENDED_DATA"
            else []
        )
        target_mechanism = str(campaign.get("evidence", {}).get("target_mechanism", ""))
        context = {
            "enabled": True,
            "campaign_id": campaign["id"],
            "direction": campaign["direction"],
            "title": campaign["title"],
            "objective": campaign["objective"],
            "rationale": campaign["rationale"],
            "success_criteria": list(definition.success_criteria),
            "avoid": list(definition.avoid),
            "attempt_number": campaign["attempts_used"],
            "maximum_attempts": campaign["maximum_attempts"],
            "remaining_after_this_attempt": (
                campaign["maximum_attempts"] - campaign["attempts_used"]
            ),
            "frozen_for_campaign": True,
            "extension_allowed": False,
            "feedback_source": "PUBLIC_RESEARCH_ONLY",
            "holdout_feedback_used": False,
            "attempt_id": attempt["id"],
            "required_factor_fields": required_factor_fields,
            "target_mechanism": target_mechanism,
            "mechanism_attempt_budget": adaptive.maximum_attempts_per_mechanism,
        }
        self.store.append_event(
            "action",
            "DIRECTION_ATTEMPT_RESERVED",
            f"方向尝试 {campaign['attempts_used']} / {campaign['maximum_attempts']}",
            f"本轮候选必须服务于：{campaign['title']}。",
            run_id=run_id,
            iteration=iteration,
            payload=context,
        )
        return context

    def _fail_direction_attempt(
        self,
        run_id: str,
        iteration: int,
        error: Exception,
        record: dict[str, Any] | None,
    ) -> None:
        attempt = self.store.direction_attempt(iteration, run_id=run_id)
        if attempt is None or attempt["status"] != "RESERVED":
            return
        error_message = f"{type(error).__name__}: {error}"
        if _is_operational_failure(error):
            campaign = self.store.cancel_direction_attempt(
                attempt_id=int(attempt["id"]),
                outcome="OPERATIONAL_FAILURE",
                diagnostics={
                    "error_category": type(error).__name__,
                    "error_stage": getattr(error, "stage", None),
                    "research_budget_charged": False,
                },
            )
            self.store.append_event(
                "audit",
                "DIRECTION_ATTEMPT_OPERATIONAL_FAILURE",
                f"运维故障未计入方向预算：{campaign['title']}",
                "API、传输、响应封装或本地运行故障不会消耗科研尝试额度。",
                run_id=run_id,
                iteration=iteration,
                level="WARN",
                payload={
                    "campaign_id": campaign["id"],
                    "direction": campaign["direction"],
                    "attempts_used": campaign["attempts_used"],
                    "maximum_attempts": campaign["maximum_attempts"],
                    "error": error_message,
                    "research_budget_charged": False,
                },
            )
            return
        config = self._research_config()
        campaign = self.store.complete_direction_attempt(
            iteration=iteration,
            candidate_id=(
                str(record["candidate_id"]) if record and record.get("candidate_id") else None
            ),
            outcome="ITERATION_FAILED",
            improved=False,
            objective_resolved=False,
            diagnostics={
                "error_category": type(error).__name__,
                "research_budget_charged": True,
            },
            early_stop_consecutive_misses=(config.adaptive_direction.early_stop_consecutive_misses),
            run_id=run_id,
        )
        self.store.append_event(
            "audit",
            "DIRECTION_ATTEMPT_FAILED",
            f"方向尝试已计入预算：{campaign['title']}",
            "候选语义或研究评价失败仍消耗一次方向尝试，禁止通过无效提案绕过预算。",
            run_id=run_id,
            iteration=iteration,
            level="WARN",
            payload={
                "campaign_id": campaign["id"],
                "direction": campaign["direction"],
                "attempts_used": campaign["attempts_used"],
                "maximum_attempts": campaign["maximum_attempts"],
                "campaign_status": campaign["status"],
            },
        )

    def _complete_direction_campaign_attempt(
        self,
        *,
        run_id: str,
        iteration: int,
        candidate_id: str,
        candidate_eligible: bool,
        portfolio_decision: PortfolioDecision,
        direction_context: dict[str, Any],
        config: ResearchConfig,
        candidate_fields: set[str],
    ) -> dict[str, Any]:
        if not direction_context.get("enabled", False):
            return {
                "enabled": False,
                "direction": "UNCONSTRAINED",
                "feedback_source": "PUBLIC_RESEARCH_ONLY",
            }
        direction_attempt = self.store.direction_attempt(iteration, run_id=run_id)
        if direction_attempt is None:
            raise RuntimeError("Adaptive direction attempt was not reserved")
        proposed_metrics = _direction_proposed_metrics(portfolio_decision, config)
        direction_outcome = assess_direction_outcome(
            str(direction_context["direction"]),
            direction_attempt["baseline"],
            proposed_metrics,
            accepted=portfolio_decision.accepted,
            candidate_eligible=candidate_eligible,
            config=config,
            candidate_fields=candidate_fields,
            required_data_fields=set(direction_context.get("required_factor_fields", [])),
        )
        campaign = self.store.complete_direction_attempt(
            iteration=iteration,
            candidate_id=candidate_id,
            outcome=(
                "PUBLIC_DIRECTION_IMPROVED"
                if direction_outcome["direction_improved"]
                else "PUBLIC_DIRECTION_MISSED"
            ),
            improved=bool(direction_outcome["direction_improved"]),
            objective_resolved=bool(direction_outcome["objective_resolved"]),
            diagnostics=direction_outcome,
            early_stop_consecutive_misses=(config.adaptive_direction.early_stop_consecutive_misses),
            run_id=run_id,
        )
        direction_metrics = {
            **direction_outcome,
            "enabled": True,
            "campaign_id": campaign["id"],
            "campaign_status": campaign["status"],
            "attempts_used": campaign["attempts_used"],
            "maximum_attempts": campaign["maximum_attempts"],
            "closure_reason": campaign["closure_reason"],
            "feedback_source": "PUBLIC_RESEARCH_ONLY",
        }
        self.store.append_event(
            "research",
            "DIRECTION_ATTEMPT_COMPLETED",
            f"方向尝试 {campaign['attempts_used']} / {campaign['maximum_attempts']}",
            (
                "公开组合动作改善了冻结方向。"
                if direction_outcome["direction_improved"]
                else "本次候选未形成冻结方向上的公开组合改善。"
            ),
            run_id=run_id,
            iteration=iteration,
            payload=direction_metrics,
        )
        if campaign["status"] != "ACTIVE":
            self.store.append_event(
                "audit",
                "DIRECTION_CAMPAIGN_CLOSED",
                f"方向战役已结束：{campaign['title']}",
                str(campaign["closure_reason"]),
                run_id=run_id,
                iteration=iteration,
                payload={
                    "campaign_id": campaign["id"],
                    "direction": campaign["direction"],
                    "status": campaign["status"],
                    "attempts_used": campaign["attempts_used"],
                    "successful_attempts": campaign["successful_attempts"],
                    "closure_reason": campaign["closure_reason"],
                    "holdout_feedback_used": False,
                },
            )
        return direction_metrics

    def _research_program_context(
        self,
        iteration: int,
        evaluator: PriceVolumeEvaluator,
        *,
        run_id: str | None = None,
        generation_id: str | None = None,
        data_capabilities: dict[str, Any] | None = None,
        pool_records: list[dict[str, Any]] | None = None,
        behavior_snapshot: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        active = self.store.active_portfolio(run_id=run_id)
        stored_protocol = active["metrics"].get("portfolio_evaluation_protocol") if active else None
        protocol_stale = bool(
            active and stored_protocol != evaluator.config.governance.protocol_version
        )
        active_members = []
        active_factors: list[FactorDefinition] = []
        active_weights: list[float] = []
        if active and not protocol_stale:
            for member in active["members"]:
                record = self.store.factor_pool_record(str(member["factor_id"]))
                if record is None:
                    continue
                proposal = record["proposal"]
                active_factors.append(factor_from_pool_record(record))
                active_weights.append(float(member["weight"]))
                active_members.append(
                    {
                        "factor_id": member["factor_id"],
                        "weight": member["weight"],
                        "source_iteration": member["source_iteration"],
                        "name": member["name"],
                        "family": member["family"],
                        "expected_direction": proposal.get("expected_direction", 1),
                        "hypothesis": proposal.get("hypothesis"),
                        "expression": proposal.get("expression"),
                    }
                )

        current_protocol_metrics: dict[str, Any] = {}
        if active_factors:
            current_protocol_metrics = evaluator.evaluate_portfolio(
                active_factors, weights=active_weights
            ).metrics
            try:
                control_failures = portfolio_absolute_gate_failures(
                    current_protocol_metrics, evaluator.config
                )
            except (KeyError, TypeError, ValueError):
                control_failures = ["INCOMPLETE_CONTROL_METRICS"]
            current_protocol_metrics.update(
                {
                    "portfolio_proposed_absolute_failures": control_failures,
                    "portfolio_absolute_gate_passed": not control_failures,
                    "portfolio_control_state": (
                        "FEASIBLE_BASELINE"
                        if not control_failures
                        else "NEEDS_REHABILITATION"
                    ),
                }
            )

        history = [
            version
            for version in self.store.portfolio_history(limit=100, run_id=run_id)
            if version["metrics"].get("portfolio_evaluation_protocol")
            == evaluator.config.governance.protocol_version
        ]
        last_accepted_iteration = int(active["iteration"]) if active and not protocol_stale else 0
        near_misses = []
        for version in history:
            diagnostics = version["metrics"].get("portfolio_option_diagnostics", [])
            near_misses.extend(
                {
                    "iteration": version["iteration"],
                    "candidate_id": version["candidate_id"],
                    "action": diagnostic.get("action"),
                    "weights": diagnostic.get("weights"),
                    "failed_gates": diagnostic.get("failed_gates", []),
                    "utility_change": diagnostic.get("utility_change"),
                    "sharpe_change": diagnostic.get("metrics", {}).get("portfolio_sharpe_change"),
                    "annual_return_change": diagnostic.get("metrics", {}).get(
                        "portfolio_annual_return_change"
                    ),
                    "turnover": diagnostic.get("metrics", {}).get("portfolio_annual_turnover"),
                    "max_correlation": diagnostic.get("metrics", {}).get(
                        "portfolio_max_factor_correlation"
                    ),
                    "annual_dispersion": diagnostic.get("metrics", {}).get(
                        "portfolio_annual_return_dispersion"
                    ),
                    "worst_fold_sharpe": diagnostic.get("metrics", {}).get(
                        "portfolio_walk_forward_worst_sharpe"
                    ),
                    "feasibility_recovery": diagnostic.get("feasibility_recovery", False),
                    "gate_failure_count": diagnostic.get(
                        "gate_failure_count", len(diagnostic.get("failed_gates", []))
                    ),
                    "absolute_violation_score": diagnostic.get("absolute_violation_score"),
                }
                for diagnostic in diagnostics
                if not diagnostic.get("accepted", False)
                and diagnostic.get("action") in {"ADD", "REPLACE"}
            )
        near_misses.sort(
            key=lambda item: (
                bool(item.get("feasibility_recovery", False)),
                -int(item.get("gate_failure_count", 100)),
                -float(item.get("absolute_violation_score") or 1_000.0),
                float(item.get("utility_change") or -1_000.0),
            ),
            reverse=True,
        )

        pool = [
            item
            for item in (
                pool_records if pool_records is not None else self.store.factor_pool(limit=5000)
            )
            if item.get("source_task_id", "legacy-ashare") == self.task_id
        ]
        family_counts = Counter(canonical_family(str(item["family"])) for item in pool)
        status_counts = Counter(str(item["status"]) for item in pool)
        capabilities = data_capabilities or build_research_data_capabilities(evaluator.workspace)
        eligible_extended = set(capabilities.get("eligible_extended_fields", []))
        recent_pool = pool[:40]
        field_usage: Counter[str] = Counter()
        mechanism_counts: Counter[str] = Counter()
        mechanism_eligible: Counter[str] = Counter()
        mechanism_promoted: Counter[str] = Counter()
        mechanism_clusters: dict[str, set[str]] = {}
        recent_extended_experiments = 0
        behavior = behavior_snapshot or {}
        behavior_factors = behavior.get("factors", {})
        for item in pool:
            proposal = item.get("proposal", {})
            used = expression_fields(proposal.get("expression"))
            mechanism = str(
                proposal.get("canonical_mechanism")
                or classify_mechanism(
                    fields=used,
                    family=str(item.get("family", "")),
                    name=str(item.get("name", "")),
                    hypothesis=str(proposal.get("hypothesis", "")),
                )
            )
            mechanism_counts[mechanism] += 1
            if item.get("status") in {"ELIGIBLE", "ACTIVE"}:
                mechanism_eligible[mechanism] += 1
            if item.get("metrics", {}).get("production_promotion_gate_passed", False):
                mechanism_promoted[mechanism] += 1
            cluster_id = behavior_factors.get(str(item["factor_id"]), {}).get(
                "behavior_cluster_id"
            )
            if cluster_id:
                mechanism_clusters.setdefault(mechanism, set()).add(str(cluster_id))
        for item in recent_pool:
            proposal = item.get("proposal", {})
            used = expression_fields(proposal.get("expression"))
            field_usage.update(used & eligible_extended)
            if used & eligible_extended:
                recent_extended_experiments += 1
        mechanism_stats = {
            mechanism: {
                "attempted": mechanism_counts[mechanism],
                "library_eligible": mechanism_eligible[mechanism],
                "promotion_passed": mechanism_promoted[mechanism],
                "behavior_clusters": len(mechanism_clusters.get(mechanism, set())),
                "smoothed_library_yield": (
                    (mechanism_eligible[mechanism] + 1.0)
                    / (mechanism_counts[mechanism] + 2.0)
                ),
            }
            for mechanism in sorted(mechanism_counts)
        }
        active_extended_fields = sorted(
            {
                field_name
                for member in active_members
                for field_name in expression_fields(member.get("expression"))
                if field_name in eligible_extended
            }
        )
        data_experiment = {
            "enabled": bool(eligible_extended),
            "eligible_extended_fields": sorted(eligible_extended),
            "under_tested_fields": sorted(
                field_name for field_name in eligible_extended if field_usage[field_name] == 0
            ),
            "recent_extended_experiments": recent_extended_experiments,
            "recent_field_usage": dict(sorted(field_usage.items())),
            "mechanism_counts": dict(sorted(mechanism_counts.items())),
            "mechanism_stats": mechanism_stats,
            "active_portfolio_extended_fields": active_extended_fields,
            "maximum_direction_attempts": (
                evaluator.config.adaptive_direction.maximum_attempts_per_campaign
            ),
            "instruction": (
                "When the frozen direction is EXPLORE_EXTENDED_DATA, use one coherent mechanism "
                "with at least one eligible extended field and compare its marginal portfolio "
                "value."
            ),
        }
        policy = evaluator.config.evaluation
        evidence_tier = getattr(
            evaluator, "research_evidence_tier", research_evidence_tier(evaluator.config)
        )
        resolved_generation = generation_id or evaluator.config.generation
        generation_state = self.store.generation_state(resolved_generation) or {}
        return {
            "objective": (
                "Improve the A-share long-only active portfolio through incremental value or "
                "diversification gates; long-short Alpha and IC are secondary diagnostics and "
                "cannot rescue a weak long-only candidate."
            ),
            "primary_evaluation_basis": "A_SHARE_LONG_ONLY_WEEKLY_NON_PIT_PROXY",
            "research_evidence_tier": evidence_tier,
            "production_promotion_allowed": (
                evidence_tier == "PRIMARY_DISCOVERY"
            ),
            "secondary_diagnostics": ["LONG_SHORT_ALPHA", "IC", "RANK_IC"],
            "iteration": iteration,
            "plateau_iterations": max(0, iteration - last_accepted_iteration),
            "research_mandate": "assigned by the frozen adaptive direction campaign",
            "data_experiment": data_experiment,
            "active_portfolio": {
                "version_id": active["id"] if active and not protocol_stale else None,
                "source_iteration": (
                    active["iteration"] if active and not protocol_stale else None
                ),
                "metrics": current_protocol_metrics,
                "metrics_source": "current_protocol_revaluation",
                "legacy_portfolio_withheld": protocol_stale,
                "stored_protocol": stored_protocol,
                "members": active_members,
            },
            "action_weight_grid": [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.55, 0.60, 0.70],
            "gate_policy": {
                "minimum_incremental_net_ir": policy.minimum_incremental_net_ir,
                "minimum_incremental_annual_return": policy.minimum_incremental_annual_return,
                "maximum_incremental_drawdown_deterioration": (
                    policy.maximum_incremental_drawdown_deterioration
                ),
                "maximum_annual_turnover": policy.maximum_annual_turnover,
                "maximum_annual_return_dispersion": policy.maximum_annual_return_dispersion,
                "maximum_factor_correlation": policy.maximum_library_correlation,
                "minimum_capacity_cny": policy.minimum_capacity_cny,
                "minimum_walk_forward_positive_fraction": (policy.minimum_positive_fold_fraction),
                "minimum_deflated_sharpe_probability": (policy.minimum_deflated_sharpe_probability),
                "maximum_probability_backtest_overfitting": (
                    policy.maximum_probability_backtest_overfitting
                ),
                "minimum_parameter_positive_fraction": (policy.minimum_parameter_positive_fraction),
            },
            "research_generation": {
                "id": resolved_generation,
                "protocol_version": evaluator.config.governance.protocol_version,
                "candidate_attempts": generation_state.get("candidate_attempts", 0),
                "maximum_candidates": generation_state.get(
                    "maximum_candidates", evaluator.config.budget.max_candidates_per_generation
                ),
                "holdout_attempts": generation_state.get("holdout_attempts", 0),
                "maximum_holdout_attempts": generation_state.get(
                    "maximum_holdout_attempts",
                    evaluator.config.governance.maximum_holdout_evaluations_per_generation,
                ),
                "holdout_feedback": "categorical verdict only; exact metrics are inaccessible",
            },
            "factor_pool": {
                "count": len(pool),
                "global_count": len(pool_records or pool),
                "family_counts": dict(family_counts.most_common()),
                "status_counts": dict(status_counts.most_common()),
                "behavior_snapshot": {
                    "protocol": behavior.get("protocol"),
                    "evaluated_count": len(behavior_factors),
                    "cluster_count": int(behavior.get("cluster_count", 0) or 0),
                    "pending_count": max(0, len(pool_records or pool) - len(behavior_factors)),
                },
                "recent_expressions": [
                    {
                        "factor_id": item["factor_id"],
                        "name": item["name"],
                        "family": item["family"],
                        "signature": _expression_signature(item["proposal"].get("expression")),
                    }
                    for item in pool[:40]
                ],
            },
            "best_recent_near_misses": near_misses[:8],
            "data_limitations": list(evaluator.workspace.blockers),
            "evaluation_split": {
                "exploration_start": evaluator.config.splits.train.start.isoformat(),
                "exploration_end": evaluator.config.splits.train.end.isoformat(),
                "walk_forward_first_validation_year": (
                    evaluator.config.walk_forward.first_validation_year
                ),
                "walk_forward_last_validation_year": (
                    evaluator.config.walk_forward.last_validation_year
                ),
                "holdout": "withheld from researcher",
            },
        }

    async def _interruptible_sleep(self, seconds: float) -> None:
        remaining = seconds
        while remaining > 0:
            if self._state()["stop_requested"]:
                return
            duration = min(0.5, remaining)
            await asyncio.sleep(duration)
            remaining -= duration


def _memory_summary(
    name: str,
    family: str,
    metrics: dict[str, Any],
    decision: str,
    *,
    proposal: dict[str, Any] | None = None,
) -> str:
    diagnostics = metrics.get("portfolio_option_diagnostics", [])
    accepted_diagnostics = [item for item in diagnostics if item.get("accepted")]
    candidate_diagnostics = accepted_diagnostics or [
        item
        for item in diagnostics
        if item.get("action") in {"ADD", "REPLACE"} and not item.get("accepted", False)
    ]
    best = max(
        candidate_diagnostics or diagnostics,
        key=lambda item: float(item.get("utility_change", -1_000.0)),
        default=None,
    )
    best_summary = (
        {
            "action": best.get("action"),
            "weights": best.get("weights"),
            "utility_change": best.get("utility_change"),
            "failed_gates": best.get("failed_gates", []),
            "sharpe_change": best.get("metrics", {}).get("portfolio_sharpe_change"),
            "annual_return_change": best.get("metrics", {}).get("portfolio_annual_return_change"),
            "turnover": best.get("metrics", {}).get("portfolio_annual_turnover"),
            "max_correlation": best.get("metrics", {}).get("portfolio_max_factor_correlation"),
            "annual_dispersion": best.get("metrics", {}).get("portfolio_annual_return_dispersion"),
            "worst_fold_sharpe": best.get("metrics", {}).get("portfolio_walk_forward_worst_sharpe"),
            "feasibility_recovery": best.get("feasibility_recovery", False),
            "absolute_violation_score": best.get("absolute_violation_score"),
        }
        if best
        else None
    )
    payload = {
        "name": name,
        "family": family,
        "decision": decision,
        "expression_signature": _expression_signature(
            proposal.get("expression") if proposal else None
        ),
        "single_factor": {
            "evaluation_basis": "A_SHARE_LONG_ONLY_WEEKLY_NON_PIT_PROXY",
            "sharpe": metrics.get("long_only_sharpe_ratio"),
            "annual_return": metrics.get("long_only_simple_annual_return"),
            "compound_annual_return": metrics.get("long_only_compound_annual_return"),
            "max_drawdown": metrics.get("long_only_max_drawdown"),
            "rank_ic": metrics.get("rank_ic_mean"),
            "turnover": metrics.get("long_only_annual_turnover"),
            "coverage": metrics.get("long_only_coverage"),
            "walk_forward_positive_fraction": metrics.get(
                "long_only_walk_forward_positive_fraction"
            ),
            "walk_forward_worst_sharpe": metrics.get("long_only_walk_forward_worst_sharpe"),
            "deflated_sharpe_probability": metrics.get("long_only_deflated_sharpe_probability"),
            "probability_backtest_overfitting": metrics.get("probability_backtest_overfitting"),
            "parameter_stability_positive_fraction": metrics.get(
                "parameter_stability_positive_fraction"
            ),
            "library_redundancy": {
                "max_signal_correlation": metrics.get("library_signal_correlation_max"),
                "peer": (
                    metrics.get("library_signal_correlation_peer_name")
                    or metrics.get("library_signal_correlation_peer")
                ),
            },
            "exploratory_failures": [
                failure
                for failure in metrics.get("exploratory_gate_failures", [])
                if failure not in {"incremental_drawdown", "return_drawdown_efficiency"}
            ],
            "alpha_diagnostic": {
                "sharpe": metrics.get("sharpe_ratio"),
                "annual_return": metrics.get("simple_annual_return"),
                "rank_ic": metrics.get("rank_ic_mean"),
            },
        },
        "portfolio": {
            "action": metrics.get("portfolio_action"),
            "accepted": metrics.get("portfolio_action_accepted"),
            "failed_gates": metrics.get("portfolio_action_gate_failures", []),
            "active_factor_ids": metrics.get("portfolio_active_factor_ids", []),
            "best_option": best_summary,
            "holdout_verdict": metrics.get("holdout_verdict"),
            "capital_verdict": metrics.get("capital_verdict"),
        },
        "protocol": {
            "generation": metrics.get("research_generation"),
            "version": metrics.get("evaluation_protocol"),
            "feedback_policy": metrics.get("holdout_feedback_policy"),
        },
        "adaptive_direction": metrics.get("adaptive_direction"),
        "planner_features": metrics.get("planner_features"),
    }
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def _select_research_memories(
    memories: list[dict[str, Any]],
    *,
    limit: int = 40,
    recent: int = 16,
    current_protocol: str | None = None,
) -> list[dict[str, Any]]:
    if limit <= 0:
        return []
    ordered = sorted(memories, key=lambda item: int(item["id"]), reverse=True)
    parsed: list[tuple[dict[str, Any], dict[str, Any] | None]] = []
    for row in ordered:
        payload = None
        if row.get("kind") == "evaluation":
            try:
                candidate = json.loads(str(row.get("content", "")))
                if isinstance(candidate, dict):
                    payload = candidate
            except json.JSONDecodeError:
                pass
        sanitized = _sanitize_memory_row(
            row,
            payload,
            current_protocol=current_protocol,
        )
        sanitized_payload = payload
        if payload is not None:
            sanitized_payload = json.loads(str(sanitized["content"]))
        parsed.append((sanitized, sanitized_payload))

    selected: dict[int, dict[str, Any]] = {
        int(row["id"]): row for row, _ in parsed[: min(recent, limit)]
    }

    accepted = [
        (row, payload)
        for row, payload in parsed
        if payload and payload.get("portfolio", {}).get("accepted")
    ]
    near_misses = sorted(
        (
            (row, payload)
            for row, payload in parsed
            if payload and isinstance(payload.get("portfolio", {}).get("best_option"), dict)
        ),
        key=lambda item: float(
            item[1]["portfolio"]["best_option"].get("utility_change") or -1_000.0
        ),
        reverse=True,
    )
    for row, _ in [*accepted[:8], *near_misses[:10]]:
        selected.setdefault(int(row["id"]), row)
        if len(selected) >= limit:
            break

    represented_families = {
        canonical_family(str(payload.get("family") or ""))
        for row, payload in parsed
        if int(row["id"]) in selected and payload
    }
    for row, payload in parsed:
        if len(selected) >= limit:
            break
        family = canonical_family(str(payload.get("family") or "")) if payload else None
        if family and family != "unknown" and family not in represented_families:
            selected[int(row["id"])] = row
            represented_families.add(family)

    for row, _ in parsed:
        if len(selected) >= limit:
            break
        selected.setdefault(int(row["id"]), row)
    return sorted(selected.values(), key=lambda item: int(item["id"]))


def _sanitize_memory_row(
    row: dict[str, Any],
    payload: dict[str, Any] | None,
    *,
    current_protocol: str | None = None,
) -> dict[str, Any]:
    sanitized = dict(row)
    if payload is None:
        return sanitized
    clean = json.loads(json.dumps(payload, ensure_ascii=False))
    protocol = clean.get("protocol", {})
    stored_protocol = protocol.get("version")
    if current_protocol and stored_protocol != current_protocol:
        clean["single_factor"] = {
            "stale_protocol": True,
            "exploratory_failures": ["STALE_PROTOCOL_REEVALUATION_REQUIRED"],
        }
        old_portfolio = clean.get("portfolio", {})
        clean["portfolio"] = {
            "action": old_portfolio.get("action"),
            "accepted": False,
            "failed_gates": ["STALE_PROTOCOL_REEVALUATION_REQUIRED"],
            "active_factor_ids": [],
            "best_option": None,
            "holdout_verdict": None,
            "capital_verdict": None,
        }
        clean["protocol"] = {
            "version": stored_protocol,
            "current_version": current_protocol,
            "promotion_evidence_allowed": False,
        }
        clean["adaptive_direction"] = None
        clean["feedback_version"] = "actionable_memory_v3"
        sanitized["content"] = json.dumps(clean, ensure_ascii=False, separators=(",", ":"))
        return sanitized
    single = clean.get("single_factor", {})
    single["exploratory_failures"] = [
        failure
        for failure in single.get("exploratory_failures", [])
        if failure not in {"incremental_drawdown", "return_drawdown_efficiency"}
    ]
    clean["feedback_version"] = "actionable_memory_v3"
    sanitized["content"] = json.dumps(clean, ensure_ascii=False, separators=(",", ":"))
    return sanitized


def _multiple_testing_adjustments(
    metrics: dict[str, Any],
    history: list[dict[str, Any]],
    *,
    generation: str,
    alpha: float,
) -> dict[str, Any]:
    # Public validation remains adaptively observed across generation boundaries.
    # Renaming a generation must not reset multiple-testing penalties.
    del generation
    prior = list(history)
    p_value_key = (
        "long_only_net_return_hac_p_value"
        if "long_only_net_return_hac_p_value" in metrics
        else "net_return_hac_p_value"
    )
    fold_key = (
        "long_only_walk_forward_folds"
        if "long_only_walk_forward_folds" in metrics
        else "walk_forward_folds"
    )
    p_values = [
        float(item[p_value_key]) for item in prior if isinstance(item.get(p_value_key), int | float)
    ]
    p_values.append(float(metrics[p_value_key]))
    rejected = benjamini_hochberg(p_values, alpha=alpha)

    profiles = [_fold_profile(item.get(fold_key, [])) for item in prior]
    profiles.append(_fold_profile(metrics.get(fold_key, [])))
    profiles = [profile for profile in profiles if profile]
    pbo = 0.0
    if len(profiles) >= 2:
        common = sorted(set.intersection(*(set(profile) for profile in profiles)))
        if len(common) >= 4:
            if len(common) % 2:
                common = common[:-1]
            matrix = np.asarray(
                [[profile[year] for profile in profiles] for year in common], dtype=float
            )
            pbo = probability_of_backtest_overfitting(matrix)
    return {
        "multiple_testing_family_size": len(p_values),
        "multiple_testing_scope": "CUMULATIVE_ALL_GENERATIONS",
        "multiple_testing_fdr_alpha": alpha,
        "multiple_testing_fdr_passed": bool(rejected[-1]),
        "probability_backtest_overfitting": float(pbo),
        "multiple_testing_primary_basis": (
            "A_SHARE_LONG_ONLY" if p_value_key.startswith("long_only_") else "ALPHA_DIAGNOSTIC"
        ),
    }


def _fold_profile(folds: list[dict[str, Any]]) -> dict[int, float]:
    return {
        int(str(fold["validation_start"])[:4]): float(fold["annual_return"])
        for fold in folds
        if fold.get("validation_start") and isinstance(fold.get("annual_return"), int | float)
    }


def _frozen_portfolio_hash(
    factors: tuple[FactorDefinition, ...],
    weights: tuple[float, ...],
    *,
    generation: str,
) -> str:
    payload = {
        "generation": generation,
        "members": [
            {"factor_id": factor.factor_id, "weight": round(float(weight), 12)}
            for factor, weight in zip(factors, weights, strict=True)
        ],
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def _is_public_feasibility_recovery(decision: PortfolioDecision) -> bool:
    """Keep protocol-migration steps public until every absolute gate passes."""
    metrics = decision.evaluation.metrics
    return bool(
        decision.accepted
        and metrics.get("portfolio_feasibility_recovery", False)
        and metrics.get("portfolio_proposed_absolute_failures", [])
    )


def _direction_proposed_metrics(
    decision: PortfolioDecision,
    config: ResearchConfig,
) -> dict[str, Any]:
    """Expose the best public candidate option to the planner, even when gates reject it."""
    if decision.accepted:
        return dict(decision.evaluation.metrics)
    candidates = [
        diagnostic
        for diagnostic in decision.option_diagnostics
        if diagnostic.get("action") in {"ADD", "REPLACE"}
        and isinstance(diagnostic.get("metrics"), dict)
    ]
    if not candidates:
        return dict(decision.evaluation.metrics)

    def key(diagnostic: dict[str, Any]) -> tuple[bool, int, float, float]:
        violation = diagnostic.get("absolute_violation_score")
        return (
            bool(diagnostic.get("feasibility_recovery", False)),
            -int(diagnostic.get("gate_failure_count", len(diagnostic.get("failed_gates", [])))),
            -float(violation if isinstance(violation, int | float) else 1_000.0),
            float(diagnostic.get("utility_change") or -1_000.0),
        )

    best = max(candidates, key=key)
    metrics = dict(best["metrics"])
    metrics.setdefault(
        "portfolio_proposed_absolute_failures",
        portfolio_absolute_gate_failures(metrics, config),
    )
    metrics["direction_evidence_action"] = best.get("action")
    metrics["direction_evidence_gate_failures"] = list(best.get("failed_gates", []))
    metrics["direction_evidence_accepted"] = False
    return metrics


def _expression_signature(expression: dict[str, Any] | None) -> str | None:
    if not expression:
        return None
    operator = str(expression.get("operator"))
    parameters = expression.get("parameters", {})
    parameter_text = ",".join(f"{key}={value}" for key, value in sorted(parameters.items()))
    children = ",".join(
        filter(
            None,
            (_expression_signature(argument) for argument in expression.get("arguments", [])),
        )
    )
    return f"{operator}[{parameter_text}]({children})" if parameter_text or children else operator


def _validate_data_direction_alignment(
    candidate_fields: set[str], *, direction_context: dict[str, Any]
) -> None:
    if direction_context.get("direction") != "EXPLORE_EXTENDED_DATA":
        return
    required = {str(field) for field in direction_context.get("required_factor_fields", [])}
    if not required:
        raise ValueError("Extended-data campaign has no research-eligible extended fields")
    if not candidate_fields & required:
        raise ValueError(
            "EXPLORE_EXTENDED_DATA requires at least one eligible extended field; "
            f"allowed extended fields: {', '.join(sorted(required))}"
        )


def _validate_mechanism_direction_alignment(
    candidate_mechanism: str, *, direction_context: dict[str, Any]
) -> None:
    if direction_context.get("direction") not in {
        "EXPLORE_EXTENDED_DATA",
        "EXPLORE_NEW_MECHANISM",
    }:
        return
    target = str(direction_context.get("target_mechanism", ""))
    if target and candidate_mechanism != target:
        raise ValueError(
            f"Frozen mechanism campaign requires {target}; proposal classified as "
            f"{candidate_mechanism}. Change the economic mechanism, not only its parameters."
        )


def _pool_expression_signatures(
    pool_snapshot: list[dict[str, Any]], *, limit: int = 250
) -> list[str]:
    """Compact signatures of recently tested expressions for duplicate avoidance."""
    signatures: dict[str, None] = {}
    for record in pool_snapshot:
        if len(signatures) >= limit:
            break
        signature = _expression_signature((record.get("proposal") or {}).get("expression"))
        if signature:
            signatures.setdefault(signature, None)
    return list(signatures)


def _redundancy_references(
    pool_snapshot: list[dict[str, Any]],
    *,
    exclude_id: str,
    behavior_factors: dict[str, dict[str, Any]] | None = None,
    candidate_fields: set[str] | None = None,
    candidate_mechanism: str | None = None,
    limit: int = 48,
) -> list[FactorDefinition]:
    """Build a compact, globally representative redundancy panel.

    The panel keeps every active member, favors offline behavior-cluster leaders,
    and then covers distinct clusters and candidate-adjacent mechanisms. This is
    more informative than taking the latest factors from the current task.
    """
    if limit <= 0:
        return []
    behavior = behavior_factors or {}
    fields = candidate_fields or set()

    def priority(index_record: tuple[int, dict[str, Any]]) -> tuple[float, int]:
        index, record = index_record
        factor_id = str(record.get("factor_id", ""))
        evidence = behavior.get(factor_id, {})
        proposal = record.get("proposal") or {}
        record_fields = expression_fields(proposal.get("expression"))
        mechanism = str(
            proposal.get("canonical_mechanism")
            or classify_mechanism(
                fields=record_fields,
                family=str(record.get("family", "")),
                name=str(record.get("name", "")),
                hypothesis=str(proposal.get("hypothesis", "")),
            )
        )
        status_score = {
            "ACTIVE": 1_000.0,
            "ELIGIBLE": 300.0,
            "SCREENED_OUT": 20.0,
        }.get(str(record.get("status", "")), 0.0)
        score = status_score
        if evidence.get("behavior_cluster_role") == "LEADER":
            score += 180.0
        if candidate_mechanism and mechanism == candidate_mechanism:
            score += 120.0
        score += 12.0 * len(fields & record_fields)
        if not evidence:
            score += 5.0
        return score, -index

    ranked = [
        record
        for _, record in sorted(
            enumerate(pool_snapshot),
            key=priority,
            reverse=True,
        )
        if str(record.get("factor_id", "")) != exclude_id
    ]
    ordered: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    selected_clusters: set[str] = set()

    def add(record: dict[str, Any]) -> None:
        factor_id = str(record.get("factor_id", ""))
        if not factor_id or factor_id in selected_ids:
            return
        ordered.append(record)
        selected_ids.add(factor_id)
        cluster_id = behavior.get(factor_id, {}).get("behavior_cluster_id")
        if cluster_id:
            selected_clusters.add(str(cluster_id))

    for record in ranked:
        if record.get("status") == "ACTIVE":
            add(record)
    for record in ranked:
        evidence = behavior.get(str(record.get("factor_id", "")), {})
        cluster_id = evidence.get("behavior_cluster_id")
        if evidence.get("behavior_cluster_role") == "LEADER" and (
            not cluster_id or str(cluster_id) not in selected_clusters
        ):
            add(record)
        if len(ordered) >= limit:
            break
    for record in ranked:
        cluster_id = behavior.get(str(record.get("factor_id", "")), {}).get(
            "behavior_cluster_id"
        )
        if not cluster_id or str(cluster_id) not in selected_clusters:
            add(record)
        if len(ordered) >= limit:
            break
    for record in ranked:
        add(record)
        if len(ordered) >= limit:
            break

    references: list[FactorDefinition] = []
    for record in ordered:
        if len(references) >= limit:
            break
        try:
            references.append(factor_from_pool_record(record))
        except Exception:
            continue
    return references


def _online_behavior_assignment(
    redundancy_metrics: dict[str, Any],
    behavior_factors: dict[str, dict[str, Any]],
    *,
    threshold: float,
) -> dict[str, Any]:
    """Assign a provisional searchable behavior label before the next full clustering run."""
    peer_id = redundancy_metrics.get("library_signal_correlation_peer")
    similarity = float(redundancy_metrics.get("library_signal_correlation_max") or 0.0)
    signed = float(redundancy_metrics.get("library_signal_correlation_signed") or 0.0)
    peer = behavior_factors.get(str(peer_id), {}) if peer_id else {}
    related = bool(peer and similarity >= threshold)
    if related:
        cluster_id = str(peer.get("behavior_cluster_id") or "PENDING_RELATED_CLUSTER")
        label = str(peer.get("behavior_cluster_label") or "临时相关行为簇")
        role = "MEMBER"
        redundancy = (
            "NEAR_DUPLICATE"
            if similarity >= 0.95
            else "SUBSTITUTE"
            if similarity >= 0.82
            else "RELATED"
        )
    else:
        cluster_id = "PENDING_NEW_CLUSTER"
        label = "待全库复核的新行为簇"
        role = "PROVISIONAL_LEADER"
        redundancy = "DISTINCT" if peer_id else "UNASSESSED"
    return {
        "online_behavior_cluster_id": cluster_id,
        "online_behavior_cluster_label": label,
        "online_behavior_cluster_size": (
            int(peer.get("behavior_cluster_size", 1) or 1) + 1 if related else 1
        ),
        "online_behavior_cluster_role": role,
        "online_behavior_nearest_factor_id": peer_id,
        "online_behavior_nearest_similarity": round(similarity, 6),
        "online_behavior_signal_correlation": round(signed, 6),
        "online_behavior_redundancy": redundancy,
        "online_behavior_cluster_method": "ONLINE_NEAREST_SIGNAL_V1",
        "online_behavior_pending_full_recluster": True,
        "online_behavior_assignment_threshold": float(threshold),
    }


def _retrieve_library_peers(
    candidate: dict[str, Any],
    pool_snapshot: list[dict[str, Any]],
    behavior_factors: dict[str, dict[str, Any]],
    *,
    limit: int,
) -> list[dict[str, Any]]:
    """Retrieve a small mechanism-aware, behavior-diverse context for LLM roles."""
    if limit <= 0:
        return []
    candidate_fields = expression_fields(candidate.get("expression"))
    candidate_mechanism = str(candidate.get("canonical_mechanism", ""))

    def score(index_record: tuple[int, dict[str, Any]]) -> tuple[float, int]:
        index, record = index_record
        proposal = record.get("proposal") or {}
        fields = expression_fields(proposal.get("expression"))
        mechanism = str(
            proposal.get("canonical_mechanism")
            or classify_mechanism(
                fields=fields,
                family=str(record.get("family", "")),
                name=str(record.get("name", "")),
                hypothesis=str(proposal.get("hypothesis", "")),
            )
        )
        evidence = behavior_factors.get(str(record.get("factor_id", "")), {})
        value = 20.0 * len(candidate_fields & fields)
        if candidate_mechanism and mechanism == candidate_mechanism:
            value += 80.0
        if record.get("status") == "ACTIVE":
            value += 45.0
        elif record.get("status") == "ELIGIBLE":
            value += 20.0
        if evidence.get("behavior_cluster_role") == "LEADER":
            value += 30.0
        return value, -index

    ranked = [
        record
        for _, record in sorted(enumerate(pool_snapshot), key=score, reverse=True)
    ]
    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    selected_clusters: set[str] = set()
    for record in ranked:
        factor_id = str(record.get("factor_id", ""))
        cluster_id = behavior_factors.get(factor_id, {}).get("behavior_cluster_id")
        if cluster_id and str(cluster_id) in selected_clusters:
            continue
        selected.append(record)
        selected_ids.add(factor_id)
        if cluster_id:
            selected_clusters.add(str(cluster_id))
        if len(selected) >= limit:
            return selected
    for record in ranked:
        factor_id = str(record.get("factor_id", ""))
        if factor_id in selected_ids:
            continue
        selected.append(record)
        if len(selected) >= limit:
            break
    return selected


def _pre_evaluation_role_route(
    candidate: dict[str, Any],
    *,
    research_program: dict[str, Any],
    preflight_warnings: set[str],
) -> dict[str, Any]:
    """Select only roles that can add evidence for this candidate."""
    fields = expression_fields(candidate.get("expression"))
    mechanism = str(candidate.get("canonical_mechanism", "OTHER_INTERPRETABLE"))
    data_experiment = research_program.get("data_experiment", {})
    extended = fields & set(data_experiment.get("eligible_extended_fields", []))
    statistics = data_experiment.get("mechanism_stats", {}).get(mechanism, {})
    direction = research_program.get("adaptive_direction", {}).get("direction")
    nodes = _expression_node_count(candidate.get("expression"))
    roles = [FALSIFICATION_DESIGNER]
    reasons: dict[str, list[str]] = {
        FALSIFICATION_DESIGNER: ["每个候选都必须预注册证伪计划"]
    }
    if nodes >= 10 or extended or preflight_warnings:
        roles.insert(0, REVIEWER)
        reasons[REVIEWER] = [
            "复杂表达式、扩展字段或预检提示需要独立语义与时序复核"
        ]
    under_represented = (
        int(statistics.get("attempted", 0) or 0) < 3
        or int(statistics.get("behavior_clusters", 0) or 0) < 2
        or direction in {"EXPLORE_EXTENDED_DATA", "EXPLORE_NEW_MECHANISM"}
    )
    if under_represented:
        roles.append(FACTOR_LIBRARIAN)
        reasons[FACTOR_LIBRARIAN] = ["机制覆盖不足，需要规范分类和同业关系"]
    return {
        "policy": "CONDITIONAL_ROLE_ROUTING_V1",
        "roles": roles,
        "reasons": reasons,
        "expression_nodes": nodes,
        "extended_fields": sorted(extended),
        "preflight_warnings": sorted(preflight_warnings),
        "candidate_mechanism": mechanism,
    }


def _post_evaluation_role_route(
    metrics: dict[str, Any],
    *,
    candidate_eligible: bool,
    portfolio_accepted: bool,
) -> dict[str, Any]:
    failures = {
        str(item)
        for key in ("exploratory_gate_failures", "portfolio_action_gate_failures")
        for item in metrics.get(key, [])
    }
    roles: list[str] = []
    reasons: dict[str, list[str]] = {}
    if not portfolio_accepted and (candidate_eligible or len(failures) <= 4):
        roles.append(ROOT_CAUSE_ANALYST)
        reasons[ROOT_CAUSE_ANALYST] = ["候选接近可用或失败边界集中，需要归因下一步"]
    execution_failures = failures & {"turnover", "cost_stress", "capacity", "coverage"}
    if portfolio_accepted or execution_failures:
        roles.append(TCA_OBSERVER)
        reasons[TCA_OBSERVER] = [
            "组合已晋级或存在明确交易成本、容量与覆盖风险"
        ]
    return {
        "policy": "CONDITIONAL_ROLE_ROUTING_V1",
        "roles": roles,
        "reasons": reasons,
        "failure_categories": sorted(failures),
        "candidate_eligible": candidate_eligible,
        "portfolio_accepted": portfolio_accepted,
    }


def _expression_node_count(expression: dict[str, Any] | None) -> int:
    if not isinstance(expression, dict):
        return 0
    return 1 + sum(
        _expression_node_count(argument)
        for argument in expression.get("arguments", [])
        if isinstance(argument, dict)
    )


def _matching_structure(
    expression: dict[str, Any], factor_pool: list[dict[str, Any]]
) -> dict[str, Any] | None:
    signature = _expression_structure(expression)
    for record in factor_pool:
        proposal = record.get("proposal", {})
        existing = proposal.get("expression")
        if existing and _expression_structure(existing) == signature:
            return record
    return None


def _expression_structure(expression: dict[str, Any]) -> tuple[Any, ...]:
    operator = str(expression.get("operator"))
    parameters = expression.get("parameters", {})
    retained_parameters = (("name", parameters.get("name")),) if operator == "field" else ()
    return (
        operator,
        retained_parameters,
        tuple(_expression_structure(argument) for argument in expression.get("arguments", [])),
    )


def _failure_memory(message: str, fingerprint: str, record: dict[str, Any] | None) -> str:
    if not record or not record.get("proposal"):
        return f"FAILED [{fingerprint}] before candidate persistence: {message}"
    proposal = record["proposal"]
    expression = json.dumps(proposal.get("expression"), sort_keys=True, separators=(",", ":"))
    return (
        f"FAILED [{fingerprint}] candidate={record.get('candidate_id')} "
        f"name={proposal.get('name')} error={message}; expression={expression}"
    )


def _circuit_breaker_reason(
    error: Exception, *, consecutive_failures: int, same_failure_count: int
) -> str | None:
    deterministic = not isinstance(error, ModelInvocationError) or error.stage != "transport"
    if deterministic and same_failure_count >= 3:
        return "同一确定性错误连续出现三次"
    if consecutive_failures >= 5:
        return "连续失败达到五次"
    return None


def _is_operational_failure(error: Exception) -> bool:
    if isinstance(error, ModelInvocationError):
        return error.stage in {
            "transport",
            "response_envelope",
            "proposal_contract",
            "role_contract",
        }
    return isinstance(error, TimeoutError | ConnectionError | OSError)


def _genesis_baseline() -> FactorDefinition:
    return FactorDefinition(
        name="baseline_reversal_20_v1",
        family="reversal",
        hypothesis=(
            "A-share twenty-day relative winners exhibit short-horizon cross-sectional reversal."
        ),
        expression=operation(
            "cs_zscore",
            operation(
                "negate",
                operation("returns", field("adj_close"), periods=20),
            ),
        ),
    )


def _codex_task_baseline() -> FactorDefinition:
    return FactorDefinition(
        name="Codex_ShortTerm_Reversal_5d_Baseline",
        family="ShortTermReversal",
        hypothesis=(
            "A-share five-day relative winners tend to mean-revert over the next tradable "
            "holding period."
        ),
        expression=operation(
            "cs_zscore",
            operation(
                "negate",
                operation("returns", field("adj_close"), periods=5),
            ),
        ),
    )
