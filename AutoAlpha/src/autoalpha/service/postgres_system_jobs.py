from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from typing import Any, Protocol

from autoalpha.service.store import (
    _bounded_confidence,
    _canonical,
    _hash,
    _materialized_cache_state,
    _now,
)
from autoalpha.service.system_job_sql import SystemJobSql, system_job_sql


class DbCursor(Protocol):
    rowcount: int

    def execute(self, sql: str, parameters: tuple[Any, ...] = ()) -> DbCursor: ...

    def fetchone(self) -> Any | None: ...

    def fetchall(self) -> list[Any]: ...


class DbConnection(Protocol):
    def cursor(self) -> DbCursor: ...

    def commit(self) -> None: ...

    def rollback(self) -> None: ...

    def close(self) -> None: ...


ConnectionFactory = Callable[[], DbConnection]


class PostgresSystemJobStore:
    """PostgreSQL adapter for the Job Center control-plane tables.

    This intentionally covers only the queue hot path first. The broader
    `ServiceStore` remains SQLite-backed until strategy, factor, and event
    tables receive their own tested adapters.
    """

    def __init__(
        self,
        *,
        database_url: str | None = None,
        connection_factory: ConnectionFactory | None = None,
    ) -> None:
        if not database_url and connection_factory is None:
            raise ValueError("database_url or connection_factory is required")
        self.database_url = database_url
        self.connection_factory = connection_factory
        self.sql: SystemJobSql = system_job_sql("postgresql")

    @contextmanager
    def connection(self) -> Iterator[DbConnection]:
        connection = self._connect()
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def enqueue_system_job(
        self,
        *,
        job_id: str,
        queue: str,
        job_type: str,
        payload: dict[str, Any],
        priority: int = 100,
        resource_group: str = "default",
        max_workers: int = 1,
        progress_total: int = 0,
        max_attempts: int = 3,
    ) -> dict[str, Any]:
        now = _now()
        with self.connection() as connection:
            cursor = connection.cursor()
            cursor.execute(
                self.sql.enqueue_job_sql(),
                (
                    job_id,
                    queue,
                    job_type,
                    int(priority),
                    resource_group,
                    max(1, int(max_workers)),
                    max(0, int(progress_total)),
                    _canonical(payload),
                    max(1, int(max_attempts)),
                    now,
                    now,
                ),
            )
            self._insert_system_job_log(
                cursor,
                job_id,
                level="INFO",
                event="ENQUEUED",
                message=f"{job_type} queued on {queue}",
                payload={
                    "queue": queue,
                    "job_type": job_type,
                    "priority": int(priority),
                    "resource_group": resource_group,
                    "max_workers": max(1, int(max_workers)),
                },
            )
        return self.system_job(job_id)

    def update_system_job(self, job_id: str, **values: Any) -> dict[str, Any]:
        encoded = _encode_job_update(values)
        encoded["updated_at"] = _now()
        with self.connection() as connection:
            cursor = connection.cursor()
            cursor.execute(self.sql.select_job_by_id_sql(projection="status"), (job_id,))
            previous = cursor.fetchone()
            if previous is None:
                raise KeyError(f"System job not found: {job_id}")
            cursor.execute(self.sql.update_job_sql(list(encoded)), (*encoded.values(), job_id))
            if cursor.rowcount != 1:
                raise KeyError(f"System job not found: {job_id}")
            previous_status = str(_row_get(previous, "status"))
            next_status = values.get("status")
            if next_status is not None and previous_status != str(next_status):
                self._insert_system_job_log(
                    cursor,
                    job_id,
                    level="ERROR" if str(next_status) == "FAILED" else "INFO",
                    event="STATUS_CHANGED",
                    message=f"{previous_status} -> {next_status}",
                    payload={
                        "from_status": previous_status,
                        "to_status": str(next_status),
                        "error": values.get("error"),
                    },
                )
        return self.system_job(job_id)

    def claim_system_job(
        self,
        *,
        queue: str,
        worker_id: str,
        lease_seconds: int = 300,
        resource_group: str | None = None,
    ) -> dict[str, Any] | None:
        now = _now()
        lease_expires_at = (
            datetime.now(UTC) + timedelta(seconds=max(30, int(lease_seconds)))
        ).isoformat()
        clauses = [
            "queue=%s",
            "(status='QUEUED' OR (status='RUNNING' AND lease_expires_at IS NOT NULL "
            "AND lease_expires_at < %s))",
            "attempts < max_attempts",
        ]
        parameters: list[Any] = [queue, now]
        if resource_group:
            clauses.append("resource_group=%s")
            parameters.append(resource_group)
        where = " AND ".join(clauses)
        claimed_job_id: str | None = None
        with self.connection() as connection:
            cursor = connection.cursor()
            cursor.execute(self.sql.claim_candidates_sql(where), tuple(parameters))
            rows = cursor.fetchall()
            for row in rows:
                job_id = str(_row_get(row, "job_id"))
                group = str(_row_get(row, "resource_group"))
                cursor.execute(self.sql.resource_capacity_sql(), (queue, group))
                capacity_row = cursor.fetchone()
                capacity = max(
                    1,
                    int(
                        _row_first(capacity_row)
                        if capacity_row and _row_first(capacity_row) is not None
                        else _row_get(row, "max_workers")
                    ),
                )
                cursor.execute(self.sql.active_resource_sql(), (queue, group, job_id, now))
                active = cursor.fetchone()
                if int(_row_first(active) if active else 0) >= capacity:
                    continue
                cursor.execute(
                    self.sql.claim_update_sql(),
                    (worker_id, lease_expires_at, now, now, now, job_id),
                )
                self._insert_system_job_log(
                    cursor,
                    job_id,
                    level="INFO",
                    event="CLAIMED",
                    message=f"Lease acquired by {worker_id}",
                    payload={
                        "worker_id": worker_id,
                        "lease_expires_at": lease_expires_at,
                        "attempt": int(_row_get(row, "attempts")) + 1,
                    },
                )
                claimed_job_id = job_id
                break
        return self.system_job(claimed_job_id) if claimed_job_id else None

    def recover_expired_system_jobs(self, *, queue: str | None = None) -> int:
        now = _now()
        clauses = ["status='RUNNING'", "lease_expires_at IS NOT NULL", "lease_expires_at < %s"]
        parameters: list[Any] = [now]
        if queue:
            clauses.append("queue=%s")
            parameters.append(queue)
        where = " AND ".join(clauses)
        with self.connection() as connection:
            cursor = connection.cursor()
            cursor.execute(self.sql.recover_expired_select_sql(where), tuple(parameters))
            expired = cursor.fetchall()
            cursor.execute(self.sql.recover_expired_update_sql(where), (now, *parameters))
            recovered = int(cursor.rowcount)
            for row in expired:
                exhausted = int(_row_get(row, "attempts")) >= int(_row_get(row, "max_attempts"))
                self._insert_system_job_log(
                    cursor,
                    str(_row_get(row, "job_id")),
                    level="ERROR" if exhausted else "WARNING",
                    event="LEASE_EXPIRED_RECOVERED",
                    message="Lease expired; job failed"
                    if exhausted
                    else "Lease expired; job returned to queue",
                    payload={
                        "attempts": int(_row_get(row, "attempts")),
                        "max_attempts": int(_row_get(row, "max_attempts")),
                        "recovered_status": "FAILED" if exhausted else "QUEUED",
                    },
                )
        return recovered

    def system_job(self, job_id: str) -> dict[str, Any]:
        with self.connection() as connection:
            cursor = connection.cursor()
            cursor.execute(self.sql.select_job_by_id_sql(), (job_id,))
            row = cursor.fetchone()
        if row is None:
            raise KeyError(f"System job not found: {job_id}")
        return _system_job_record(row)

    def system_jobs(
        self, *, queue: str | None = None, status: str | None = None, limit: int = 500
    ) -> list[dict[str, Any]]:
        parameters: list[Any] = []
        if queue:
            parameters.append(queue)
        if status:
            parameters.append(status)
        parameters.append(min(max(int(limit), 1), 5000))
        with self.connection() as connection:
            cursor = connection.cursor()
            cursor.execute(
                self.sql.list_jobs_sql(
                    queue=bool(queue),
                    status=bool(status),
                    limit_index=len(parameters),
                ),
                tuple(parameters),
            )
            rows = cursor.fetchall()
        return [_system_job_record(row) for row in rows]

    def system_job_logs_for_jobs(
        self, job_ids: list[str], *, limit_per_job: int = 3
    ) -> dict[str, list[dict[str, Any]]]:
        if not job_ids:
            return {}
        result: dict[str, list[dict[str, Any]]] = {job_id: [] for job_id in job_ids}
        with self.connection() as connection:
            cursor = connection.cursor()
            cursor.execute(self.sql.logs_for_jobs_sql(len(job_ids)), tuple(job_ids))
            rows = cursor.fetchall()
        for row in rows:
            job_id = str(_row_get(row, "job_id"))
            if len(result.setdefault(job_id, [])) < max(1, int(limit_per_job)):
                result[job_id].append(_system_job_log_record(row))
        return result

    def upsert_materialized_snapshot(
        self,
        key: str,
        payload: dict[str, Any],
        *,
        fingerprint: str | None = None,
        ttl_seconds: int | None = None,
        source: str = "unknown",
        status: str = "READY",
    ) -> dict[str, Any]:
        now = _now()
        expires_at = (
            (datetime.now(UTC) + timedelta(seconds=max(1, int(ttl_seconds)))).isoformat()
            if ttl_seconds is not None
            else None
        )
        encoded = _canonical(payload)
        digest = fingerprint or hashlib.sha256(encoded.encode()).hexdigest()
        with self.connection() as connection:
            cursor = connection.cursor()
            cursor.execute(
                """INSERT INTO materialized_snapshots
                (key, payload_json, fingerprint, source, status, expires_at, updated_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT(key) DO UPDATE SET
                payload_json=excluded.payload_json,
                fingerprint=excluded.fingerprint,
                source=excluded.source,
                status=excluded.status,
                expires_at=excluded.expires_at,
                updated_at=excluded.updated_at""",
                (key, encoded, digest, source, status, expires_at, now),
            )
        snapshot = self.materialized_snapshot(key)
        assert snapshot is not None
        return snapshot

    def materialized_snapshot(self, key: str) -> dict[str, Any] | None:
        with self.connection() as connection:
            cursor = connection.cursor()
            cursor.execute("SELECT * FROM materialized_snapshots WHERE key=%s", (key,))
            row = cursor.fetchone()
        return _materialized_snapshot_record(row) if row is not None else None

    def upsert_strategy_experiment_object(
        self,
        *,
        experiment_id: str,
        stage: str,
        object_type: str,
        source_system: str,
        source_id: str,
        title: str,
        status: str,
        market: str | None = None,
        protocol: dict[str, Any] | None = None,
        metrics: dict[str, Any] | None = None,
        evidence: dict[str, Any] | None = None,
        tags: list[str] | None = None,
    ) -> dict[str, Any]:
        now = _now()
        with self.connection() as connection:
            cursor = connection.cursor()
            cursor.execute(
                """INSERT INTO strategy_experiment_objects
                (experiment_id, stage, object_type, source_system, source_id, title, status,
                 market, protocol_json, metrics_json, evidence_json, tags_json,
                 created_at, updated_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT(experiment_id) DO UPDATE SET
                stage=excluded.stage,
                object_type=excluded.object_type,
                source_system=excluded.source_system,
                source_id=excluded.source_id,
                title=excluded.title,
                status=excluded.status,
                market=excluded.market,
                protocol_json=excluded.protocol_json,
                metrics_json=excluded.metrics_json,
                evidence_json=excluded.evidence_json,
                tags_json=excluded.tags_json,
                updated_at=excluded.updated_at""",
                (
                    experiment_id,
                    stage,
                    object_type,
                    source_system,
                    source_id,
                    title,
                    status,
                    market,
                    _canonical(protocol or {}),
                    _canonical(metrics or {}),
                    _canonical(evidence or {}),
                    _canonical(tags or []),
                    now,
                    now,
                ),
            )
        record = self.strategy_experiment_object(experiment_id)
        assert record is not None
        return record

    def upsert_strategy_experiment_edge(
        self,
        source_experiment_id: str,
        target_experiment_id: str,
        relation: str,
        *,
        evidence: dict[str, Any] | None = None,
    ) -> None:
        now = _now()
        with self.connection() as connection:
            cursor = connection.cursor()
            cursor.execute(
                """INSERT INTO strategy_experiment_edges
                (source_experiment_id, target_experiment_id, relation, evidence_json, created_at)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT(source_experiment_id, target_experiment_id, relation)
                DO UPDATE SET evidence_json=excluded.evidence_json""",
                (
                    source_experiment_id,
                    target_experiment_id,
                    relation,
                    _canonical(evidence or {}),
                    now,
                ),
            )

    def strategy_experiment_object(self, experiment_id: str) -> dict[str, Any] | None:
        with self.connection() as connection:
            cursor = connection.cursor()
            cursor.execute(
                "SELECT * FROM strategy_experiment_objects WHERE experiment_id=%s",
                (experiment_id,),
            )
            row = cursor.fetchone()
        return _strategy_experiment_record(row) if row is not None else None

    def strategy_experiment_objects(
        self, *, stage: str | None = None, limit: int = 1000
    ) -> list[dict[str, Any]]:
        parameters: tuple[Any, ...] = (
            (stage, min(max(int(limit), 1), 10_000))
            if stage
            else (min(max(int(limit), 1), 10_000),)
        )
        where = "WHERE stage=%s" if stage else ""
        with self.connection() as connection:
            cursor = connection.cursor()
            cursor.execute(
                f"""SELECT * FROM strategy_experiment_objects {where}
                ORDER BY updated_at DESC LIMIT %s""",
                parameters,
            )
            rows = cursor.fetchall()
        return [_strategy_experiment_record(row) for row in rows]

    def create_formal_strategy_version(
        self,
        *,
        strategy_uid: str,
        source_experiment_id: str | None,
        name: str,
        market: str,
        lifecycle: str,
        signal_policy: dict[str, Any],
        rebalance_policy: dict[str, Any],
        execution_policy: dict[str, Any],
        risk_policy: dict[str, Any],
        cost_policy: dict[str, Any],
        monitoring_policy: dict[str, Any],
        evidence: dict[str, Any],
    ) -> dict[str, Any]:
        now = _now()
        specification = {
            "strategy_uid": strategy_uid,
            "name": name.strip(),
            "market": market,
            "lifecycle": lifecycle,
            "source_experiment_id": source_experiment_id,
            "signal_policy": signal_policy,
            "rebalance_policy": rebalance_policy,
            "execution_policy": execution_policy,
            "risk_policy": risk_policy,
            "cost_policy": cost_policy,
            "monitoring_policy": monitoring_policy,
            "evidence": evidence,
        }
        specification_hash = hashlib.sha256(_canonical(specification).encode()).hexdigest()
        with self.connection() as connection:
            cursor = connection.cursor()
            cursor.execute(
                """SELECT COALESCE(MAX(version), 0) AS version
                FROM formal_strategy_versions WHERE strategy_uid=%s""",
                (strategy_uid,),
            )
            version = int(_row_first(cursor.fetchone()) or 0) + 1
            cursor.execute(
                """INSERT INTO formal_strategy_versions
                (strategy_uid, version, source_experiment_id, name, market, lifecycle,
                 signal_policy_json, rebalance_policy_json, execution_policy_json,
                 risk_policy_json, cost_policy_json, monitoring_policy_json, evidence_json,
                 specification_hash, created_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""",
                (
                    strategy_uid,
                    version,
                    source_experiment_id,
                    name.strip(),
                    market,
                    lifecycle,
                    _canonical(signal_policy),
                    _canonical(rebalance_policy),
                    _canonical(execution_policy),
                    _canonical(risk_policy),
                    _canonical(cost_policy),
                    _canonical(monitoring_policy),
                    _canonical(evidence),
                    specification_hash,
                    now,
                ),
            )
        return self.formal_strategy_version(strategy_uid, version)

    def formal_strategy_versions(
        self, *, lifecycle: str | None = None, limit: int = 500
    ) -> list[dict[str, Any]]:
        parameters: tuple[Any, ...] = (
            (lifecycle, min(max(int(limit), 1), 5000))
            if lifecycle
            else (min(max(int(limit), 1), 5000),)
        )
        where = "WHERE lifecycle=%s" if lifecycle else ""
        with self.connection() as connection:
            cursor = connection.cursor()
            cursor.execute(
                f"""SELECT * FROM formal_strategy_versions {where}
                ORDER BY created_at DESC LIMIT %s""",
                parameters,
            )
            rows = cursor.fetchall()
        return [_formal_strategy_record(row) for row in rows]

    def formal_strategy_version(self, strategy_uid: str, version: int) -> dict[str, Any]:
        with self.connection() as connection:
            cursor = connection.cursor()
            cursor.execute(
                """SELECT * FROM formal_strategy_versions
                WHERE strategy_uid=%s AND version=%s""",
                (strategy_uid, int(version)),
            )
            row = cursor.fetchone()
        if row is None:
            raise KeyError(f"Formal strategy version not found: {strategy_uid} v{version}")
        return _formal_strategy_record(row)

    def update_formal_strategy_lifecycle(
        self,
        strategy_uid: str,
        version: int,
        *,
        lifecycle: str,
        evidence: dict[str, Any],
    ) -> dict[str, Any]:
        with self.connection() as connection:
            cursor = connection.cursor()
            cursor.execute(
                """UPDATE formal_strategy_versions
                SET lifecycle=%s, evidence_json=%s
                WHERE strategy_uid=%s AND version=%s""",
                (lifecycle, _canonical(evidence), strategy_uid, int(version)),
            )
            if cursor.rowcount != 1:
                raise KeyError(f"Formal strategy version not found: {strategy_uid} v{version}")
        return self.formal_strategy_version(strategy_uid, version)

    def upsert_factor_knowledge(
        self,
        *,
        factor_id: str,
        canonical_mechanism: str,
        mechanism_summary: str,
        tags: list[str],
        review: dict[str, Any],
        falsification: dict[str, Any],
        related_factors: list[dict[str, Any]],
    ) -> dict[str, Any]:
        now = _now()
        with self.connection() as connection:
            cursor = connection.cursor()
            cursor.execute("SELECT 1 FROM factor_pool WHERE factor_id=%s", (factor_id,))
            if cursor.fetchone() is None:
                raise KeyError(f"Factor not found: {factor_id}")
            cursor.execute(
                """INSERT INTO factor_knowledge
                (factor_id, canonical_mechanism, mechanism_summary, tags_json,
                 review_json, falsification_json, updated_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT(factor_id) DO UPDATE SET
                canonical_mechanism=excluded.canonical_mechanism,
                mechanism_summary=excluded.mechanism_summary,
                tags_json=excluded.tags_json,
                review_json=excluded.review_json,
                falsification_json=excluded.falsification_json,
                updated_at=excluded.updated_at""",
                (
                    factor_id,
                    canonical_mechanism,
                    mechanism_summary,
                    _canonical(tags),
                    _canonical(review),
                    _canonical(falsification),
                    now,
                ),
            )
            for relation in related_factors[:20]:
                target = str(relation.get("factor_id", ""))
                if not target or target == factor_id:
                    continue
                cursor.execute("SELECT 1 FROM factor_pool WHERE factor_id=%s", (target,))
                if cursor.fetchone() is None:
                    continue
                confidence = _bounded_confidence(relation.get("confidence", 0.0))
                cursor.execute(
                    """INSERT INTO factor_knowledge_edges
                    (source_factor_id, target_factor_id, relation, confidence,
                     rationale, created_at)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    ON CONFLICT(source_factor_id, target_factor_id, relation) DO UPDATE SET
                    confidence=excluded.confidence,
                    rationale=excluded.rationale""",
                    (
                        factor_id,
                        target,
                        str(relation.get("relation", "RELATED"))[:80],
                        confidence,
                        str(relation.get("rationale", ""))[:2000],
                        now,
                    ),
                )
        knowledge = self.factor_knowledge(factor_id)
        assert knowledge is not None
        return knowledge

    def factor_knowledge(self, factor_id: str) -> dict[str, Any] | None:
        with self.connection() as connection:
            cursor = connection.cursor()
            cursor.execute("SELECT * FROM factor_knowledge WHERE factor_id=%s", (factor_id,))
            row = cursor.fetchone()
            cursor.execute(
                """SELECT edge.*, pool.name AS target_name
                FROM factor_knowledge_edges AS edge
                JOIN factor_pool AS pool ON pool.factor_id=edge.target_factor_id
                WHERE edge.source_factor_id=%s
                ORDER BY edge.confidence DESC, edge.target_factor_id""",
                (factor_id,),
            )
            edges = cursor.fetchall()
        if row is None:
            return None
        return _factor_knowledge_record(row, edges=edges)

    def factor_knowledge_catalog(
        self, *, task_id: str | None = None, limit: int = 500
    ) -> list[dict[str, Any]]:
        parameters: tuple[Any, ...] = (
            (task_id, min(max(int(limit), 1), 5000))
            if task_id
            else (min(max(int(limit), 1), 5000),)
        )
        where = "WHERE pool.source_task_id=%s" if task_id else ""
        with self.connection() as connection:
            cursor = connection.cursor()
            cursor.execute(
                f"""SELECT knowledge.*, pool.name, pool.family, pool.source_task_id,
                pool.source_iteration
                FROM factor_knowledge AS knowledge
                JOIN factor_pool AS pool ON pool.factor_id=knowledge.factor_id
                {where} ORDER BY knowledge.updated_at DESC LIMIT %s""",
                parameters,
            )
            rows = cursor.fetchall()
        return [_factor_knowledge_record(row, include_edges=False) for row in rows]

    def upsert_factor_pool(
        self,
        *,
        factor_id: str,
        source_iteration: int,
        proposal: dict[str, Any],
        metrics: dict[str, Any],
        status: str,
        status_reason: str,
        source_task_id: str = "legacy-ashare",
    ) -> None:
        now = _now()
        initial_state = (
            "SHADOW"
            if status == "ACTIVE"
            else "QUALIFIED"
            if status == "ELIGIBLE"
            else "RESEARCH"
        )
        with self.connection() as connection:
            cursor = connection.cursor()
            cursor.execute(
                """INSERT INTO factor_pool
                (factor_id, source_task_id, source_iteration, name, family, proposal_json,
                 metrics_json, status, status_reason, created_at, updated_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT(factor_id) DO UPDATE SET
                source_task_id=excluded.source_task_id,
                source_iteration=excluded.source_iteration,
                name=excluded.name,
                family=excluded.family,
                proposal_json=excluded.proposal_json,
                metrics_json=excluded.metrics_json,
                status=excluded.status,
                status_reason=excluded.status_reason,
                updated_at=excluded.updated_at""",
                (
                    factor_id,
                    source_task_id,
                    int(source_iteration),
                    str(proposal.get("name", factor_id)),
                    str(proposal.get("family", "unknown")),
                    _canonical(proposal),
                    _canonical(metrics),
                    status,
                    status_reason,
                    now,
                    now,
                ),
            )
            cursor.execute(
                """INSERT INTO factor_lifecycle_events
                (factor_id, previous_state, state, actor, reason, created_at)
                SELECT %s, NULL, %s, 'AUTO_RESEARCH', %s, %s
                WHERE NOT EXISTS (
                    SELECT 1 FROM factor_lifecycle_events WHERE factor_id=%s
                )""",
                (factor_id, initial_state, status_reason, now, factor_id),
            )

    def factor_pool(self, *, limit: int = 500) -> list[dict[str, Any]]:
        with self.connection() as connection:
            cursor = connection.cursor()
            cursor.execute(
                """SELECT * FROM factor_pool
                ORDER BY source_iteration DESC LIMIT %s""",
                (min(max(int(limit), 1), 5000),),
            )
            rows = cursor.fetchall()
        return [_factor_pool_record(row) for row in rows]

    def factor_pool_count(self) -> int:
        with self.connection() as connection:
            cursor = connection.cursor()
            cursor.execute("SELECT COUNT(*) AS count FROM factor_pool")
            row = cursor.fetchone()
        return int(_row_first(row) or 0)

    def factor_pool_record(self, factor_id: str) -> dict[str, Any] | None:
        with self.connection() as connection:
            cursor = connection.cursor()
            cursor.execute("SELECT * FROM factor_pool WHERE factor_id=%s", (factor_id,))
            row = cursor.fetchone()
        return _factor_pool_record(row) if row is not None else None

    def merge_factor_pool_metrics(self, factor_id: str, metrics: dict[str, Any]) -> dict[str, Any]:
        if not metrics:
            record = self.factor_pool_record(factor_id)
            if record is None:
                raise KeyError(f"Factor not found: {factor_id}")
            return record["metrics"]
        now = _now()
        with self.connection() as connection:
            cursor = connection.cursor()
            cursor.execute(
                "SELECT metrics_json FROM factor_pool WHERE factor_id=%s", (factor_id,)
            )
            row = cursor.fetchone()
            if row is None:
                raise KeyError(f"Factor not found: {factor_id}")
            current = json.loads(str(_row_get(row, "metrics_json") or "{}"))
            merged = {**current, **metrics}
            cursor.execute(
                "UPDATE factor_pool SET metrics_json=%s, updated_at=%s WHERE factor_id=%s",
                (_canonical(merged), now, factor_id),
            )
        return merged

    def settings(self) -> dict[str, str]:
        with self.connection() as connection:
            cursor = connection.cursor()
            cursor.execute("SELECT key, value FROM settings")
            rows = cursor.fetchall()
        return {str(_row_get(row, "key")): str(_row_get(row, "value")) for row in rows}

    def save_settings(self, values: dict[str, str]) -> None:
        now = _now()
        with self.connection() as connection:
            cursor = connection.cursor()
            for key, value in values.items():
                cursor.execute(
                    """INSERT INTO settings(key, value, updated_at) VALUES (%s, %s, %s)
                    ON CONFLICT(key) DO UPDATE SET
                    value=excluded.value, updated_at=excluded.updated_at""",
                    (key, value, now),
                )

    def save_settings_revision(
        self,
        values: dict[str, str],
        *,
        change_note: str,
        changed_by: str = "local-operator",
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        if not values:
            return None
        secret_keys = {"api_key", "openai_api_key", "tushare_token"}
        forbidden = secret_keys.intersection(key.casefold() for key in values)
        if forbidden:
            raise ValueError(f"Secrets cannot be stored in settings revisions: {sorted(forbidden)}")
        now = _now()
        with self.connection() as connection:
            cursor = connection.cursor()
            cursor.execute("SELECT key, value FROM settings")
            rows = cursor.fetchall()
            previous = {str(_row_get(row, "key")): str(_row_get(row, "value")) for row in rows}
            changed = {
                str(key): str(value)
                for key, value in values.items()
                if previous.get(str(key)) != str(value)
            }
            if not changed:
                return None
            current = {**previous, **changed}
            fingerprint = hashlib.sha256(
                _canonical(
                    {
                        "created_at": now,
                        "changed_keys": sorted(changed),
                        "values": current,
                        "metadata": metadata or {},
                    }
                ).encode()
            ).hexdigest()
            for key, value in changed.items():
                cursor.execute(
                    """INSERT INTO settings(key, value, updated_at) VALUES (%s, %s, %s)
                    ON CONFLICT(key) DO UPDATE SET
                    value=excluded.value, updated_at=excluded.updated_at""",
                    (key, value, now),
                )
            cursor.execute(
                """INSERT INTO settings_revisions
                (created_at, change_note, changed_by, changed_keys_json,
                 previous_values_json, values_json, metadata_json, fingerprint)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING id""",
                (
                    now,
                    change_note.strip() or "更新全局设置",
                    changed_by.strip() or "local-operator",
                    _canonical(sorted(changed)),
                    _canonical(previous),
                    _canonical(current),
                    _canonical(metadata or {}),
                    fingerprint,
                ),
            )
            revision_id = int(_row_first(cursor.fetchone()) or 0)
        return self.settings_revision(revision_id)

    def settings_revisions(self, *, limit: int = 30) -> list[dict[str, Any]]:
        with self.connection() as connection:
            cursor = connection.cursor()
            cursor.execute(
                "SELECT * FROM settings_revisions ORDER BY id DESC LIMIT %s",
                (min(max(int(limit), 1), 200),),
            )
            rows = cursor.fetchall()
        return [_settings_revision_record(row) for row in rows]

    def settings_revision(self, revision_id: int) -> dict[str, Any] | None:
        with self.connection() as connection:
            cursor = connection.cursor()
            cursor.execute("SELECT * FROM settings_revisions WHERE id=%s", (int(revision_id),))
            row = cursor.fetchone()
        return _settings_revision_record(row) if row else None

    def append_event(
        self,
        category: str,
        event: str,
        title: str,
        message: str,
        *,
        level: str = "INFO",
        run_id: str | None = None,
        iteration: int | None = None,
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        body_payload = payload or {}
        timestamp = _now()
        with self.connection() as connection:
            cursor = connection.cursor()
            cursor.execute("SELECT record_hash FROM events ORDER BY id DESC LIMIT 1")
            previous = cursor.fetchone()
            previous_hash = str(_row_get(previous, "record_hash")) if previous else "0" * 64
            body = {
                "timestamp_utc": timestamp,
                "run_id": run_id,
                "iteration": iteration,
                "category": category,
                "level": level,
                "event": event,
                "title": title,
                "message": message,
                "payload": body_payload,
                "previous_hash": previous_hash,
            }
            record_hash = _hash(body)
            cursor.execute(
                """INSERT INTO events
                (timestamp_utc, run_id, iteration, category, level, event, title, message,
                 payload_json, previous_hash, record_hash)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING id""",
                (
                    timestamp,
                    run_id,
                    iteration,
                    category,
                    level,
                    event,
                    title,
                    message,
                    _canonical(body_payload),
                    previous_hash,
                    record_hash,
                ),
            )
            event_id = int(_row_first(cursor.fetchone()) or 0)
        return {"id": event_id, **body, "record_hash": record_hash}

    def events(
        self,
        *,
        after_id: int = 0,
        category: str | None = None,
        run_id: str | None = None,
        task_id: str | None = None,
        limit: int = 200,
    ) -> list[dict[str, Any]]:
        clauses = ["id > %s"]
        parameters: list[Any] = [int(after_id)]
        if category and category != "all":
            clauses.append("category = %s")
            parameters.append(category)
        if task_id and run_id:
            clauses.append("(run_id = %s OR payload_json::jsonb ->> 'task_id' = %s)")
            parameters.extend((run_id, task_id))
        elif task_id:
            clauses.append("payload_json::jsonb ->> 'task_id' = %s")
            parameters.append(task_id)
        elif run_id:
            clauses.append("run_id = %s")
            parameters.append(run_id)
        parameters.append(min(max(int(limit), 1), 1000))
        with self.connection() as connection:
            cursor = connection.cursor()
            cursor.execute(
                f"""SELECT * FROM events WHERE {' AND '.join(clauses)}
                ORDER BY id DESC LIMIT %s""",
                tuple(parameters),
            )
            rows = cursor.fetchall()
        return list(reversed([_event_record(row) for row in rows]))

    def create_research_task(
        self,
        *,
        task_id: str,
        name: str,
        market: str,
        data_path: str,
        data_start: str | None,
        data_end: str | None,
        snapshot_hash: str | None,
        status: str = "DRAFT",
        run_id: str | None = None,
        protocol: dict[str, Any] | None = None,
        protocol_hash: str | None = None,
        protocol_revision: int = 1,
        notes: str = "",
    ) -> dict[str, Any]:
        now = _now()
        with self.connection() as connection:
            cursor = connection.cursor()
            cursor.execute(
                """INSERT INTO research_tasks
                (task_id, name, market, data_path, data_start, data_end, snapshot_hash,
                 status, run_id, phase, protocol_json, protocol_hash, protocol_revision,
                 notes, created_at, updated_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""",
                (
                    task_id,
                    name,
                    market,
                    data_path,
                    data_start,
                    data_end,
                    snapshot_hash,
                    status,
                    run_id,
                    "WAITING" if status == "READY" else "CONFIGURE",
                    _canonical(protocol or {}),
                    protocol_hash,
                    int(protocol_revision),
                    notes,
                    now,
                    now,
                ),
            )
        task = self.research_task(task_id)
        assert task is not None
        return task

    def research_tasks(self) -> list[dict[str, Any]]:
        with self.connection() as connection:
            cursor = connection.cursor()
            cursor.execute(
                "SELECT * FROM research_tasks ORDER BY updated_at DESC, created_at DESC"
            )
            rows = cursor.fetchall()
        return [_research_task_record(row) for row in rows]

    def research_task(self, task_id: str) -> dict[str, Any] | None:
        with self.connection() as connection:
            cursor = connection.cursor()
            cursor.execute("SELECT * FROM research_tasks WHERE task_id=%s", (task_id,))
            row = cursor.fetchone()
        return _research_task_record(row) if row is not None else None

    def research_task_stats(self, task_id: str, run_id: str | None) -> dict[str, int]:
        with self.connection() as connection:
            cursor = connection.cursor()
            cursor.execute(
                "SELECT COUNT(*) AS count FROM factor_pool WHERE source_task_id=%s",
                (task_id,),
            )
            factor_count = int(_row_first(cursor.fetchone()) or 0)
            iteration_count = 0
            if run_id:
                cursor.execute(
                    "SELECT COUNT(*) AS count FROM iterations WHERE run_id=%s",
                    (run_id,),
                )
                iteration_count = int(_row_first(cursor.fetchone()) or 0)
        return {"factor_count": factor_count, "iteration_count": iteration_count}

    def research_task_state(self, task_id: str) -> dict[str, Any]:
        task = self.research_task(task_id)
        if task is None:
            raise KeyError(f"Research task not found: {task_id}")
        return {
            "state": task["status"],
            "phase": task["phase"],
            "run_id": task["run_id"],
            "iteration": int(task["iteration"]),
            "stop_requested": int(task["stop_requested"]),
            "updated_at": task["updated_at"],
            "last_error": task["last_error"],
        }

    def update_research_task_state(self, task_id: str, **values: Any) -> dict[str, Any]:
        allowed = {"state", "phase", "run_id", "iteration", "stop_requested", "last_error"}
        invalid = set(values) - allowed
        if invalid:
            raise ValueError(f"Unknown research-task state fields: {sorted(invalid)}")
        task_values = {
            ("status" if key == "state" else key): value for key, value in values.items()
        }
        self.update_research_task(task_id, **task_values)
        return self.research_task_state(task_id)

    def update_research_task(self, task_id: str, **values: Any) -> dict[str, Any]:
        allowed = {
            "name",
            "market",
            "data_path",
            "data_start",
            "data_end",
            "snapshot_hash",
            "status",
            "run_id",
            "phase",
            "iteration",
            "stop_requested",
            "last_error",
            "protocol_json",
            "protocol_hash",
            "protocol_revision",
            "notes",
        }
        invalid = set(values) - allowed
        if invalid:
            raise ValueError(f"Unknown research-task fields: {sorted(invalid)}")
        if not values:
            task = self.research_task(task_id)
            if task is None:
                raise KeyError(f"Research task not found: {task_id}")
            return task
        values["updated_at"] = _now()
        assignments = ", ".join(f"{key}=%s" for key in values)
        with self.connection() as connection:
            cursor = connection.cursor()
            cursor.execute(
                f"UPDATE research_tasks SET {assignments} WHERE task_id=%s",
                (*values.values(), task_id),
            )
            if cursor.rowcount != 1:
                raise KeyError(f"Research task not found: {task_id}")
        task = self.research_task(task_id)
        assert task is not None
        return task

    def begin_iteration(self, run_id: str, iteration: int) -> None:
        with self.connection() as connection:
            cursor = connection.cursor()
            cursor.execute(
                """INSERT INTO iterations(run_id, iteration, status, started_at)
                VALUES (%s, %s, 'RUNNING', %s)""",
                (run_id, int(iteration), _now()),
            )

    def stage_iteration_candidate(
        self,
        run_id: str,
        iteration: int,
        *,
        candidate_id: str,
        proposal: dict[str, Any],
    ) -> None:
        with self.connection() as connection:
            cursor = connection.cursor()
            cursor.execute(
                """UPDATE iterations SET candidate_id=%s, proposal_json=%s
                WHERE run_id=%s AND iteration=%s AND status='RUNNING'""",
                (candidate_id, _canonical(proposal), run_id, int(iteration)),
            )
            if cursor.rowcount != 1:
                raise KeyError(f"Running iteration not found: {run_id}/{iteration}")

    def finish_iteration(
        self,
        run_id: str,
        iteration: int,
        *,
        status: str,
        candidate_id: str | None = None,
        proposal: dict[str, Any] | None = None,
        metrics: dict[str, Any] | None = None,
        decision: str | None = None,
        error: str | None = None,
    ) -> None:
        with self.connection() as connection:
            cursor = connection.cursor()
            cursor.execute(
                """UPDATE iterations SET status=%s,
                candidate_id=COALESCE(%s, candidate_id),
                proposal_json=COALESCE(%s, proposal_json),
                metrics_json=COALESCE(%s, metrics_json),
                decision=COALESCE(%s, decision), error=%s, finished_at=%s
                WHERE run_id=%s AND iteration=%s""",
                (
                    status,
                    candidate_id,
                    _canonical(proposal) if proposal else None,
                    _canonical(metrics) if metrics else None,
                    decision,
                    error,
                    _now(),
                    run_id,
                    int(iteration),
                ),
            )

    def iteration_record(self, run_id: str, iteration: int) -> dict[str, Any] | None:
        with self.connection() as connection:
            cursor = connection.cursor()
            cursor.execute(
                "SELECT * FROM iterations WHERE run_id=%s AND iteration=%s",
                (run_id, int(iteration)),
            )
            row = cursor.fetchone()
        return _iteration_record(row) if row is not None else None

    def iteration_stats(self, *, run_id: str | None = None) -> dict[str, Any]:
        where = "WHERE run_id=%s" if run_id else ""
        parameters: tuple[Any, ...] = (run_id,) if run_id else ()
        with self.connection() as connection:
            cursor = connection.cursor()
            cursor.execute(
                f"SELECT status, COUNT(*) AS count FROM iterations {where} GROUP BY status",
                parameters,
            )
            rows = cursor.fetchall()
        counts = {str(_row_get(row, "status")): int(_row_get(row, "count")) for row in rows}
        total = sum(counts.values())
        completed = counts.get("COMPLETED", 0)
        return {
            "total": total,
            "completed": completed,
            "failed": counts.get("FAILED", 0),
            "running": counts.get("RUNNING", 0),
            "success_rate": completed / total if total else 0.0,
        }

    def iteration_history(
        self, limit: int = 100, *, run_id: str | None = None
    ) -> list[dict[str, Any]]:
        where = "WHERE run_id=%s" if run_id else ""
        parameters: tuple[Any, ...] = (
            (run_id, min(max(int(limit), 1), 500))
            if run_id
            else (min(max(int(limit), 1), 500),)
        )
        with self.connection() as connection:
            cursor = connection.cursor()
            cursor.execute(
                f"SELECT * FROM iterations {where} ORDER BY id DESC LIMIT %s",
                parameters,
            )
            rows = cursor.fetchall()
        return [_iteration_record(row) for row in rows]

    def metric_history(
        self, limit: int = 500, *, run_id: str | None = None
    ) -> list[dict[str, Any]]:
        where = "WHERE metrics_json IS NOT NULL"
        parameters: list[Any] = []
        if run_id:
            where += " AND run_id=%s"
            parameters.append(run_id)
        parameters.append(min(max(int(limit), 1), 5000))
        with self.connection() as connection:
            cursor = connection.cursor()
            cursor.execute(
                f"""SELECT iteration, candidate_id, decision, metrics_json, finished_at
                FROM iterations {where} ORDER BY id DESC LIMIT %s""",
                tuple(parameters),
            )
            rows = cursor.fetchall()
        result = []
        for row in reversed(rows):
            result.append(
                {
                    "iteration": _row_get(row, "iteration"),
                    "candidate_id": _row_get(row, "candidate_id"),
                    "decision": _row_get(row, "decision"),
                    "finished_at": _row_get(row, "finished_at"),
                    **json.loads(_row_get(row, "metrics_json")),
                }
            )
        return result

    def candidate_exists(self, candidate_id: str) -> bool:
        with self.connection() as connection:
            cursor = connection.cursor()
            cursor.execute(
                "SELECT 1 FROM iterations WHERE candidate_id=%s LIMIT 1",
                (candidate_id,),
            )
            row = cursor.fetchone()
        return row is not None

    def record_llm_role_artifact(
        self,
        *,
        task_id: str,
        run_id: str,
        iteration: int,
        candidate_id: str | None,
        role: str,
        stage: str,
        status: str,
        artifact: dict[str, Any],
        usage: dict[str, int],
        prompt_hash: str | None,
        response_hash: str | None,
        error: str | None = None,
    ) -> dict[str, Any]:
        now = _now()
        with self.connection() as connection:
            cursor = connection.cursor()
            cursor.execute(
                """INSERT INTO llm_role_artifacts
                (task_id, run_id, iteration, candidate_id, role, stage, status,
                 artifact_json, usage_json, prompt_hash, response_hash, error, created_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING id""",
                (
                    task_id,
                    run_id,
                    int(iteration),
                    candidate_id,
                    role,
                    stage,
                    status,
                    _canonical(artifact),
                    _canonical(usage),
                    prompt_hash,
                    response_hash,
                    error,
                    now,
                ),
            )
            artifact_id = int(_row_first(cursor.fetchone()) or 0)
        return {
            "id": artifact_id,
            "task_id": task_id,
            "run_id": run_id,
            "iteration": iteration,
            "candidate_id": candidate_id,
            "role": role,
            "stage": stage,
            "status": status,
            "artifact": artifact,
            "usage": usage,
            "prompt_hash": prompt_hash,
            "response_hash": response_hash,
            "error": error,
            "created_at": now,
        }

    def llm_role_artifacts(
        self,
        *,
        task_id: str | None = None,
        run_id: str | None = None,
        candidate_id: str | None = None,
        limit: int = 200,
    ) -> list[dict[str, Any]]:
        clauses = []
        parameters: list[Any] = []
        for name, value in (
            ("task_id", task_id),
            ("run_id", run_id),
            ("candidate_id", candidate_id),
        ):
            if value:
                clauses.append(f"{name}=%s")
                parameters.append(value)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        parameters.append(min(max(int(limit), 1), 2000))
        with self.connection() as connection:
            cursor = connection.cursor()
            cursor.execute(
                f"SELECT * FROM llm_role_artifacts {where} ORDER BY id DESC LIMIT %s",
                tuple(parameters),
            )
            rows = cursor.fetchall()
        return [_llm_role_artifact_record(row) for row in rows]

    def llm_role_summary(self, *, task_id: str | None = None) -> dict[str, Any]:
        where = "WHERE task_id=%s" if task_id else ""
        parameters: tuple[Any, ...] = (task_id,) if task_id else ()
        with self.connection() as connection:
            cursor = connection.cursor()
            cursor.execute(
                f"""SELECT role, status, COUNT(*) AS count,
                COALESCE(SUM((usage_json::jsonb ->> 'total_tokens')::int), 0) AS total_tokens,
                MAX(created_at) AS latest_at
                FROM llm_role_artifacts {where}
                GROUP BY role, status ORDER BY role, status""",
                parameters,
            )
            rows = cursor.fetchall()
        roles: dict[str, dict[str, Any]] = {}
        for row in rows:
            role = roles.setdefault(
                str(_row_get(row, "role")),
                {"completed": 0, "failed": 0, "total_tokens": 0, "latest_at": None},
            )
            status = str(_row_get(row, "status"))
            if status == "COMPLETED":
                role["completed"] += int(_row_get(row, "count"))
            else:
                role["failed"] += int(_row_get(row, "count"))
            role["total_tokens"] += int(_row_get(row, "total_tokens") or 0)
            role["latest_at"] = max(
                filter(None, (role["latest_at"], _row_get(row, "latest_at"))),
                default=None,
            )
        artifact_count = sum(v["completed"] + v["failed"] for v in roles.values())
        return {"roles": roles, "artifact_count": artifact_count}

    def _connect(self) -> DbConnection:
        if self.connection_factory is not None:
            return self.connection_factory()
        try:
            import psycopg
            from psycopg.rows import dict_row
        except ImportError as error:  # pragma: no cover - covered by runtime readiness.
            raise RuntimeError("psycopg is required for PostgreSQL Job Center") from error
        assert self.database_url is not None
        return psycopg.connect(self.database_url, row_factory=dict_row)

    def _insert_system_job_log(
        self,
        cursor: DbCursor,
        job_id: str,
        *,
        level: str,
        event: str,
        message: str,
        payload: dict[str, Any] | None = None,
    ) -> None:
        cursor.execute(
            self.sql.insert_log_sql(),
            (job_id, _now(), level.upper(), event, message, _canonical(payload or {})),
        )


def _encode_job_update(values: dict[str, Any]) -> dict[str, Any]:
    allowed = {
        "status",
        "progress_current",
        "progress_total",
        "checkpoint",
        "result",
        "error",
        "attempts",
        "started_at",
        "finished_at",
        "lease_owner",
        "lease_expires_at",
        "heartbeat_at",
    }
    invalid = set(values) - allowed
    if invalid:
        raise ValueError(f"Unknown system job fields: {sorted(invalid)}")
    encoded: dict[str, Any] = {}
    for key, value in values.items():
        if key in {"checkpoint", "result"}:
            encoded[f"{key}_json"] = _canonical(value)
        else:
            encoded[key] = value
    return encoded


def _system_job_record(row: Any) -> dict[str, Any]:
    item = dict(row)
    item["payload"] = json.loads(item.pop("payload_json"))
    item["checkpoint"] = json.loads(item.pop("checkpoint_json", "{}") or "{}")
    item["result"] = json.loads(item.pop("result_json", "{}") or "{}")
    return item


def _system_job_log_record(row: Any) -> dict[str, Any]:
    item = dict(row)
    item["payload"] = json.loads(item.pop("payload_json", "{}") or "{}")
    return item


def _materialized_snapshot_record(row: Any) -> dict[str, Any]:
    item = dict(row)
    item["payload"] = json.loads(item.pop("payload_json", "{}") or "{}")
    item["cache_state"] = _materialized_cache_state(item)
    return item


def _strategy_experiment_record(row: Any) -> dict[str, Any]:
    item = dict(row)
    item["protocol"] = json.loads(item.pop("protocol_json", "{}") or "{}")
    item["metrics"] = json.loads(item.pop("metrics_json", "{}") or "{}")
    item["evidence"] = json.loads(item.pop("evidence_json", "{}") or "{}")
    item["tags"] = json.loads(item.pop("tags_json", "[]") or "[]")
    return item


def _formal_strategy_record(row: Any) -> dict[str, Any]:
    item = dict(row)
    for key in (
        "signal_policy",
        "rebalance_policy",
        "execution_policy",
        "risk_policy",
        "cost_policy",
        "monitoring_policy",
        "evidence",
    ):
        item[key] = json.loads(item.pop(f"{key}_json", "{}") or "{}")
    return item


def _factor_knowledge_record(
    row: Any,
    *,
    edges: list[Any] | None = None,
    include_edges: bool = True,
) -> dict[str, Any]:
    item = dict(row)
    item["tags"] = json.loads(item.pop("tags_json", "[]") or "[]")
    item["review"] = json.loads(item.pop("review_json", "{}") or "{}")
    item["falsification"] = json.loads(item.pop("falsification_json", "{}") or "{}")
    if include_edges:
        item["edges"] = [dict(edge) for edge in edges or []]
    return item


def _factor_pool_record(row: Any) -> dict[str, Any]:
    item = dict(row)
    item["proposal"] = json.loads(item.pop("proposal_json", "{}") or "{}")
    item["metrics"] = json.loads(item.pop("metrics_json", "{}") or "{}")
    return item


def _settings_revision_record(row: Any) -> dict[str, Any]:
    item = dict(row)
    item["changed_keys"] = json.loads(item.pop("changed_keys_json", "[]") or "[]")
    item["previous_values"] = json.loads(item.pop("previous_values_json", "{}") or "{}")
    item["values"] = json.loads(item.pop("values_json", "{}") or "{}")
    item["metadata"] = json.loads(item.pop("metadata_json", "{}") or "{}")
    return item


def _event_record(row: Any) -> dict[str, Any]:
    item = dict(row)
    item["payload"] = json.loads(item.pop("payload_json", "{}") or "{}")
    return item


def _research_task_record(row: Any) -> dict[str, Any]:
    item = dict(row)
    item["protocol"] = json.loads(item.pop("protocol_json", "{}") or "{}")
    return item


def _iteration_record(row: Any) -> dict[str, Any]:
    item = dict(row)
    item["proposal"] = json.loads(item.pop("proposal_json", "null") or "null")
    item["metrics"] = json.loads(item.pop("metrics_json", "null") or "null")
    return item


def _llm_role_artifact_record(row: Any) -> dict[str, Any]:
    item = dict(row)
    item["artifact"] = json.loads(item.pop("artifact_json", "{}") or "{}")
    item["usage"] = json.loads(item.pop("usage_json", "{}") or "{}")
    return item


def _row_get(row: Any, key: str) -> Any:
    if isinstance(row, dict):
        return row[key]
    return getattr(row, key)


def _row_first(row: Any) -> Any:
    if row is None:
        return None
    if isinstance(row, dict):
        return next(iter(row.values()))
    if isinstance(row, (list, tuple)):
        return row[0]
    return row
