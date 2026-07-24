from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal


@dataclass(frozen=True)
class ProductTemplate:
    template_id: str
    name: str
    objective: str
    portfolio_mode: Literal["long_short", "long_only"]
    benchmark_mode: Literal["cash", "universe_equal_weight"]
    execution_mode: Literal["research_vector", "a_share_capital_ledger"]
    default_gross_exposure: float
    maximum_positions: int
    maximum_weight: float
    maximum_active_weight: float
    maximum_turnover: float
    maximum_tracking_error: float | None
    hedge_benchmark: bool = False
    production_eligible: bool = False
    limitation: str = ""

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


PRODUCT_TEMPLATES: tuple[ProductTemplate, ...] = (
    ProductTemplate(
        template_id="MARKET_NEUTRAL_RESEARCH",
        name="市场中性研究",
        objective="检验纯截面选股信号，不模拟融券或股指期货成交。",
        portfolio_mode="long_short",
        benchmark_mode="cash",
        execution_mode="research_vector",
        default_gross_exposure=0.50,
        maximum_positions=60,
        maximum_weight=0.025,
        maximum_active_weight=0.025,
        maximum_turnover=0.30,
        maximum_tracking_error=None,
        limitation="研究口径；空头可得性、融券成本和期货基差尚未进入成交账本。",
    ),
    ProductTemplate(
        template_id="LONG_ONLY_CAPITAL",
        name="A股仅多头资金回放",
        objective="以真实现金、整手和成交约束检验可部署的绝对收益组合。",
        portfolio_mode="long_only",
        benchmark_mode="cash",
        execution_mode="a_share_capital_ledger",
        default_gross_exposure=0.95,
        maximum_positions=30,
        maximum_weight=0.05,
        maximum_active_weight=0.05,
        maximum_turnover=0.30,
        maximum_tracking_error=None,
        limitation=(
            "需要未复权历史成交价、股数口径成交量及交易所状态字段；前复权面板会被资金账本拒绝。"
        ),
    ),
    ProductTemplate(
        template_id="UNIVERSE_INDEX_ENHANCED_PROXY",
        name="全市场等权指数增强代理",
        objective="在主动权重、换手、风险暴露和跟踪误差约束下检验指数增强能力。",
        portfolio_mode="long_only",
        benchmark_mode="universe_equal_weight",
        execution_mode="a_share_capital_ledger",
        default_gross_exposure=0.95,
        maximum_positions=80,
        maximum_weight=0.03,
        maximum_active_weight=0.02,
        maximum_turnover=0.20,
        maximum_tracking_error=0.08,
        limitation=(
            "需要未复权历史成交价、股数口径成交量和正式指数成分权重；前复权面板会被资金账本拒绝。"
        ),
    ),
    ProductTemplate(
        template_id="UNIVERSE_HEDGED_PROXY",
        name="股票多头加全市场对冲代理",
        objective="检验股票多头收益在扣除全市场等权风险后是否仍有稳定超额。",
        portfolio_mode="long_only",
        benchmark_mode="universe_equal_weight",
        execution_mode="a_share_capital_ledger",
        default_gross_exposure=0.50,
        maximum_positions=40,
        maximum_weight=0.04,
        maximum_active_weight=0.03,
        maximum_turnover=0.25,
        maximum_tracking_error=0.10,
        hedge_benchmark=True,
        limitation=(
            "需要未复权历史成交价和股数口径成交量；对冲腿仍需补齐合约、展期、保证金和基差。"
        ),
    ),
)

_BY_ID = {template.template_id: template for template in PRODUCT_TEMPLATES}


def product_template(template_id: str) -> ProductTemplate:
    try:
        return _BY_ID[template_id]
    except KeyError as error:
        raise ValueError(f"Unknown product template: {template_id}") from error


def product_template_catalog() -> list[dict[str, object]]:
    return [template.to_dict() for template in PRODUCT_TEMPLATES]
