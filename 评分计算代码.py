#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
赛题训练项目本地评分代码整理版

本文件独立复现项目中两种本地评分实现：
1. walkforward：004至007代使用的滚动验证评分口径。
2. legacy：早期 train_models.py 使用的固定切分评分口径。

这不是组委会官方评分器。请只把它用于带有历史标签的验证预测文件；
测试集只有预测值、没有可用标签，不能据此计算比赛得分。

输入CSV必须包含：
ts_code, trade_date, flag_limit_up, y_ret_1d, pred

运行示例：
python 评分计算代码.py 验证预测.csv --metric-version walkforward --output 评分结果.json

主分数公式：
0.4 × 平均每日Rank IC
+ 0.3 × 算术年化Top10%超额收益
+ 0.3 × (1 − 平均持仓集合换手率)

注意：
- 超额收益按日均值乘252年化，保持项目现有口径。
- 换手率按相邻有效交易日Top10%证券集合的1−交并比计算。
- 当前主评分排序只使用pred，代码没有定义并列预测的第二排序键。
- walkforward版本中，Rank IC/超额要求标签可用；换手率样本池只要求非涨停。
- 常数预测可能产生非有限Rank IC；本文件不把它任意替换为0。
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


REQUIRED_COLUMNS = {
    "ts_code",
    "trade_date",
    "flag_limit_up",
    "y_ret_1d",
    "pred",
}

WALKFORWARD_VERSION = "local_walkforward_v1_label_boundary_purged"
LEGACY_VERSION = "local_legacy_fixed_split_v1"


def _mean(values: Any) -> float:
    """项目滚动验证实现使用的普通算术平均。"""
    return float(np.mean(values)) if len(values) else float("nan")


def _safe_spearman(x: pd.Series, y: pd.Series) -> float:
    """早期固定切分实现使用的常数序列保护。"""
    if len(x) < 2 or x.nunique(dropna=True) < 2 or y.nunique(dropna=True) < 2:
        return float("nan")
    value, _ = spearmanr(x, y)
    return float(value) if np.isfinite(value) else float("nan")


def _tie_diagnostics(tradable: pd.DataFrame, n_top: int) -> dict[str, Any]:
    """记录Top10%预测并列，不改变官方换手率或综合得分。"""
    top = tradable.iloc[:n_top]
    cutoff = top["pred"].iloc[-1]
    tied_candidates = int((tradable["pred"] == cutoff).sum())
    strictly_above = int((tradable["pred"] > cutoff).sum())
    boundary_slots = int(n_top - strictly_above)
    return {
        "top10_distinct_prediction_ratio": float(top["pred"].nunique(dropna=False) / n_top),
        "top10_boundary_tied_candidates": tied_candidates,
        "top10_boundary_slots": boundary_slots,
        "top10_boundary_ambiguous": tied_candidates > boundary_slots,
    }


def score_legacy(predictions: pd.DataFrame) -> dict[str, Any]:
    """
    复现早期 train_models.py 的固定切分评分。

    早期版本会跳过常数预测日；换手率遇到少于100只证券的日期时
    跳过该日，但保留上一有效持仓集合。
    """
    ic_values: list[float] = []
    excess_values: list[float] = []

    for _, group in predictions.groupby("trade_date", sort=True):
        valid = group.dropna(subset=["pred", "y_ret_1d"])
        ic = _safe_spearman(valid["pred"], valid["y_ret_1d"])
        if np.isfinite(ic):
            ic_values.append(ic)

        investable = group[
            (group["flag_limit_up"] == 0) & group["y_ret_1d"].notna()
        ].copy()
        if len(investable) >= 100:
            # 与原实现一致：只按预测值排序，没有显式并列处理键。
            investable = investable.sort_values(
                "pred", ascending=False
            ).reset_index(drop=True)
            n_top = max(len(investable) // 10, 1)
            excess_values.append(
                float(
                    investable.iloc[:n_top]["y_ret_1d"].mean()
                    - investable["y_ret_1d"].mean()
                )
            )

    turnover_values: list[float] = []
    previous: set[str] | None = None
    for _, group in predictions.groupby("trade_date", sort=True):
        investable = group[group["flag_limit_up"] == 0].sort_values(
            "pred", ascending=False
        )
        if len(investable) < 100:
            continue
        n_top = max(len(investable) // 10, 1)
        current = set(investable.iloc[:n_top]["ts_code"].astype(str))
        if previous:
            union = current | previous
            turnover_values.append(1.0 - len(current & previous) / len(union))
        previous = current

    ic_mean = float(np.nanmean(ic_values)) if ic_values else float("nan")
    ic_std = (
        float(np.nanstd(ic_values, ddof=1))
        if len(ic_values) > 1
        else float("nan")
    )
    annual_excess = (
        float(np.mean(excess_values) * 252)
        if excess_values
        else float("nan")
    )
    mean_turnover = (
        float(np.mean(turnover_values))
        if turnover_values
        else float("nan")
    )
    final_score = (
        ic_mean * 0.4
        + annual_excess * 0.3
        + (1 - mean_turnover) * 0.3
        if all(
            np.isfinite(value)
            for value in (ic_mean, annual_excess, mean_turnover)
        )
        else float("nan")
    )

    labeled = predictions.dropna(subset=["y_ret_1d", "pred"])
    residual = labeled["pred"] - labeled["y_ret_1d"]
    return {
        "metric_version": LEGACY_VERSION,
        "rmse": (
            float(np.sqrt(np.mean(residual**2)))
            if len(residual)
            else float("nan")
        ),
        "mae": (
            float(np.mean(np.abs(residual)))
            if len(residual)
            else float("nan")
        ),
        "ic_mean": ic_mean,
        "ic_std": ic_std,
        "ic_positive_ratio": (
            float(np.mean(np.array(ic_values) > 0))
            if ic_values
            else float("nan")
        ),
        "annual_excess": annual_excess,
        "mean_turnover": mean_turnover,
        "stability_score": (
            1 - mean_turnover
            if np.isfinite(mean_turnover)
            else float("nan")
        ),
        "final_score": final_score,
        "ic_days": len(ic_values),
        "excess_days": len(excess_values),
        "turnover_days": len(turnover_values),
    }


def _aggregate_walkforward(daily: pd.DataFrame) -> dict[str, Any]:
    ics = daily.loc[daily["ic_eligible"], "ic"].to_numpy()
    mean_ic = _mean(ics)
    std_ic = (
        float(np.std(ics, ddof=1))
        if len(ics) > 1
        else float("nan")
    )
    annual_excess = (
        _mean(daily["excess"].dropna().to_numpy()) * 252
    )
    mean_turnover = _mean(daily["turnover"].dropna().to_numpy())
    tie_ratios = daily["top10_distinct_prediction_ratio"].dropna().to_numpy()
    ambiguous = daily["top10_boundary_ambiguous"].to_numpy(dtype=bool)

    return {
        "ic_mean": mean_ic,
        "ic_std": std_ic,
        "icir": mean_ic / std_ic if std_ic > 0 else 0.0,
        "ic_positive_ratio": (
            float(np.mean(ics > 0)) if len(ics) else float("nan")
        ),
        "annual_excess": annual_excess,
        "top1_annual_ret": (
            _mean(daily["top1_return"].dropna().to_numpy()) * 252
        ),
        "mean_turnover": mean_turnover,
        "stability_score": 1 - mean_turnover,
        "final_score": (
            0.4 * mean_ic
            + 0.3 * annual_excess
            + 0.3 * (1 - mean_turnover)
        ),
        "ic_days": len(ics),
        "excess_days": int(daily["excess"].notna().sum()),
        "turnover_days": int(daily["turnover"].notna().sum()),
        "top10_distinct_prediction_ratio_mean": _mean(tie_ratios),
        "top10_distinct_prediction_ratio_min": (
            float(np.min(tie_ratios)) if len(tie_ratios) else float("nan")
        ),
        "top10_boundary_ambiguous_days": int(np.sum(ambiguous)),
        "top10_boundary_ambiguous_ratio": (
            float(np.mean(ambiguous)) if len(ambiguous) else float("nan")
        ),
        "top10_boundary_tie_size_max": (
            int(daily["top10_boundary_tied_candidates"].max())
            if len(daily) else 0
        ),
    }


def score_walkforward(
    predictions: pd.DataFrame,
) -> tuple[dict[str, Any], pd.DataFrame, dict[str, Any]]:
    """
    复现004至007代的 walkforward.metric_bundle 主评分和期间统计。

    输入应是经过滚动验证边界purge的逐日预测。该函数不负责切分数据、
    训练模型或判断标签是否在某个未来日期才可见。
    """
    if not np.isfinite(predictions["pred"].to_numpy()).all():
        raise ValueError("预测含非有限值，停止评分")

    records: list[dict[str, Any]] = []
    previous: set[str] | None = None

    for date, group in predictions.groupby("trade_date", sort=True):
        valid = group.dropna(subset=["y_ret_1d"])
        ic = (
            float(spearmanr(valid["pred"], valid["y_ret_1d"])[0])
            if len(valid) >= 30
            else float("nan")
        )

        eligible = group[
            (group["flag_limit_up"] == 0)
            & group["y_ret_1d"].notna()
        ].sort_values("pred", ascending=False)
        excess = top_return = float("nan")
        if len(eligible) >= 100:
            n_top = max(len(eligible) // 10, 1)
            top_return = float(
                eligible.iloc[:n_top]["y_ret_1d"].mean()
            )
            excess = top_return - float(eligible["y_ret_1d"].mean())

        # 原口径的换手率候选池不要求标签可用。
        tradable = group[group["flag_limit_up"] == 0].sort_values(
            "pred", ascending=False
        )
        turnover = float("nan")
        tie_info: dict[str, Any] = {
            "top10_distinct_prediction_ratio": float("nan"),
            "top10_boundary_tied_candidates": 0,
            "top10_boundary_slots": 0,
            "top10_boundary_ambiguous": False,
        }
        if len(tradable) >= 100:
            n_top = max(len(tradable) // 10, 1)
            tie_info = _tie_diagnostics(tradable, n_top)
            current = set(
                tradable.iloc[:n_top]["ts_code"].astype(str)
            )
            if previous:
                turnover = 1 - len(current & previous) / len(
                    current | previous
                )
            previous = current
        else:
            previous = None

        records.append(
            {
                "trade_date": int(date),
                "ic": ic,
                "ic_eligible": len(valid) >= 30,
                "labeled_rows": len(valid),
                "excess": excess,
                "top1_return": top_return,
                "turnover": turnover,
                **tie_info,
            }
        )

    daily = pd.DataFrame(records)
    if daily.empty:
        raise ValueError("评估期无样本")

    metrics = _aggregate_walkforward(daily)
    labeled = predictions.dropna(subset=["y_ret_1d"])
    residual = labeled["pred"] - labeled["y_ret_1d"]
    metrics.update(
        {
            "metric_version": WALKFORWARD_VERSION,
            "rmse": (
                float(np.sqrt(np.mean(residual**2)))
                if len(residual)
                else float("nan")
            ),
            "mae": (
                float(np.mean(np.abs(residual)))
                if len(residual)
                else float("nan")
            ),
        }
    )

    dates = pd.to_datetime(
        daily["trade_date"].astype(str),
        format="%Y%m%d",
    )
    months = [
        {"period": str(period), **_aggregate_walkforward(group)}
        for period, group in daily.groupby(dates.dt.to_period("M"))
    ]
    quarters = [
        {"period": str(period), **_aggregate_walkforward(group)}
        for period, group in daily.groupby(dates.dt.to_period("Q"))
    ]
    known_monthly_ic = [
        item["ic_mean"]
        for item in months
        if math.isfinite(item["ic_mean"])
    ]
    metrics["monthly_ic_positive_ratio"] = _mean(
        np.array(known_monthly_ic) > 0
    )
    metrics["monthly_ic_valid_months"] = len(known_monthly_ic)

    periods = {
        "monthly": months,
        "quarterly": quarters,
        "turnover_period_boundary": (
            "retain previous-day transition within the evaluated series"
        ),
    }
    return metrics, daily, periods


def _json_safe(value: Any) -> Any:
    """把NaN和Infinity转成JSON null，避免输出非标准JSON数字。"""
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, pd.Period):
        return str(value)
    return value


def main() -> None:
    parser = argparse.ArgumentParser(
        description="计算本地历史验证预测的比赛评分指标"
    )
    parser.add_argument("predictions_csv", type=Path)
    parser.add_argument(
        "--metric-version",
        choices=("walkforward", "legacy"),
        default="walkforward",
        help="walkforward为004至007代口径；legacy为早期固定切分口径",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="可选JSON输出文件；省略时打印到屏幕",
    )
    args = parser.parse_args()

    predictions = pd.read_csv(
        args.predictions_csv,
        encoding="utf-8-sig",
    )
    missing = REQUIRED_COLUMNS - set(predictions.columns)
    if missing:
        raise ValueError(
            "评分文件缺少字段: " + ", ".join(sorted(missing))
        )

    if args.metric_version == "walkforward":
        metrics, daily, periods = score_walkforward(predictions)
        result = {
            "metric_version": WALKFORWARD_VERSION,
            "metrics": metrics,
            "daily": daily.to_dict(orient="records"),
            "periods": periods,
        }
    else:
        result = {
            "metric_version": LEGACY_VERSION,
            "metrics": score_legacy(predictions),
        }

    rendered = json.dumps(
        _json_safe(result),
        ensure_ascii=False,
        indent=2,
        allow_nan=False,
    )
    if args.output is None:
        print(rendered)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
        print("Score output saved to: " + str(args.output.resolve()))


if __name__ == "__main__":
    main()
