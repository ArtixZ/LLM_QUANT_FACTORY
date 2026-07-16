from __future__ import annotations

import asyncio

import httpx
import pytest

from autoalpha.service.openai_client import (
    CompatibleChatClient,
    ModelInvocationError,
    _direction,
    _parse_json,
)


def test_parse_json_accepts_fenced_compatible_response() -> None:
    proposal = _parse_json(
        """```json
        {"name":"mom","family":"momentum","hypothesis":"trend persists",
        "change":"add 20d momentum","expected":"positive spread",
        "expression":{"operator":"returns","arguments":[{"operator":"field",
        "parameters":{"name":"close"}}],"parameters":{"periods":20}}}
        ```"""
    )

    assert proposal["name"] == "mom"


def test_parse_json_requires_proposal_contract() -> None:
    with pytest.raises(ValueError, match="missing fields"):
        _parse_json('{"name":"incomplete"}')


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("positive", 1),
        ("higher", 1),
        ("Bullish", 1),
        ("long", 1),
        ("+1", 1),
        ("negative", -1),
        ("lower", -1),
        ("short", -1),
    ],
)
def test_compatible_direction_labels_are_normalized(value: object, expected: int) -> None:
    assert _direction(value) == expected


def test_client_normalizes_compatible_proposal_contract() -> None:
    response = {
        "choices": [
            {
                "message": {
                    "content": """{"name":"mom","family":"momentum",
                    "hypothesis":"trend persists","change":"new factor",
                    "expected":"higher return","expected_direction":"higher",
                    "expression":{"operator":"returns","arguments":[
                    {"operator":"field","parameters":{"field":"close"}}],
                    "parameters":{"window":20}}}"""
                }
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
    }
    transport = httpx.MockTransport(lambda _: httpx.Response(200, json=response))
    client = CompatibleChatClient(
        base_url="https://provider.test",
        api_key="secret",
        model="research-model",
        transport=transport,
    )

    proposal = asyncio.run(client.propose([], 1))

    assert proposal.factor.expected_direction == 1
    assert proposal.factor.expression.to_dict()["parameters"] == {"periods": 20}
    assert proposal.factor.expression.arguments[0].parameters == (("name", "close"),)


def test_client_preserves_audit_metadata_when_contract_is_invalid() -> None:
    response = {
        "choices": [{"message": {"content": "not-json"}}],
        "usage": {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18},
    }
    client = CompatibleChatClient(
        base_url="https://provider.test",
        api_key="secret",
        model="research-model",
        transport=httpx.MockTransport(lambda _: httpx.Response(200, json=response)),
    )

    with pytest.raises(ModelInvocationError) as raised:
        asyncio.run(client.propose([], 2))

    assert raised.value.stage == "proposal_contract"
    assert raised.value.response_hash is not None
    assert raised.value.usage["total_tokens"] == 18


def test_client_repairs_invalid_contract_once() -> None:
    calls = 0

    def handler(_: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        content = (
            "not-json"
            if calls == 1
            else """{"name":"repaired","family":"liquidity",
            "hypothesis":"slow liquidity persists","change":"correct contract",
            "expected":"positive spread","expected_direction":1,
            "expression":{"operator":"cs_rank","arguments":[{"operator":"field",
            "arguments":[],"parameters":{"name":"amount"}}],"parameters":{}}}"""
        )
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": content}}], "usage": {}},
        )

    client = CompatibleChatClient(
        base_url="https://provider.test",
        api_key="secret",
        model="research-model",
        transport=httpx.MockTransport(handler),
    )

    proposal = asyncio.run(client.propose([], 3))

    assert proposal.factor.name == "repaired"
    assert calls == 2


def test_client_retries_transient_transport_failure() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise httpx.ReadTimeout("slow provider", request=request)
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": """{"name":"retry","family":"liquidity",
                            "hypothesis":"liquidity persists","change":"retry request",
                            "expected":"positive spread","expected_direction":1,
                            "expression":{"operator":"cs_rank","arguments":[
                            {"operator":"field","arguments":[],
                            "parameters":{"name":"amount"}}],"parameters":{}}}"""
                        }
                    }
                ],
                "usage": {},
            },
        )

    client = CompatibleChatClient(
        base_url="https://provider.test",
        api_key="secret",
        model="research-model",
        transport=httpx.MockTransport(handler),
    )

    proposal = asyncio.run(client.propose([], 4))

    assert proposal.factor.name == "retry"
    assert calls == 2
