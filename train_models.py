# -*- coding: utf-8 -*-
"""隔离的 LightGBM 多模型训练器。

特点：
1. 每次运行先展示完整参数，只有输入配置中的 confirmation_token 才会训练。
2. 每次运行创建新的 run 目录和新的 LGBMRegressor，不加载任何旧模型。
3. 保留旧固定切分记录；新训练采用金融逻辑门禁、三折历史扩展窗口与2024最终诊断。
   比赛测试仍为2025-01-02~2026-06-08。
4. 保存模型、预测、特征重要性、数据清单、参数快照和本代摘要，并重建总报告。
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from reporting import rebuild_report

WORKSPACE = Path(__file__).parent.absolute()
DEFAULT_DATA_DIR = Path(r"D:\competition\赛题五数据").absolute()
PROVIDED_TRAIN_PATH = (DEFAULT_DATA_DIR / "训练集.csv").absolute()
PROVIDED_TEST_PATH = (DEFAULT_DATA_DIR / "测试集_X.csv").absolute()
EXPECTED_SPLIT = {
    "train_start": 20180102,
    "train_end": 20231231,
    "valid_start": 20240101,
    "valid_end": 20241231,
    "test_start": 20250102,
    "test_end": 20260608,
}

TRAIN_DTYPES = {
    "ts_code": "string",
    "trade_date": "int32",
    "open": "float32",
    "high": "float32",
    "low": "float32",
    "close": "float32",
    "vol": "float32",
    "amount": "float32",
    "flag_limit_up": "int8",
    "flag_limit_down": "int8",
    "y_ret_1d": "float32",
}
TEST_DTYPES = {k: v for k, v in TRAIN_DTYPES.items() if k != "y_ret_1d"}
FORBIDDEN_FEATURE_TOKENS = ("y_ret_1d", "target", "label", "future", "next", "pred", "old_model")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="隔离的 LightGBM 多模型训练器")
    parser.add_argument("--config", required=True, help="模型配置 JSON 路径")
    parser.add_argument("--preview", action="store_true", help="只展示日期、参数和拟合次数；不读大文件、不训练")
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        config = json.load(f)
    required = ["model_name", "confirmation_token", "feature_mode", "model_params", "split"]
    missing = [key for key in required if key not in config]
    if missing:
        raise ValueError(f"配置缺少字段: {missing}")
    model_name = str(config["model_name"])
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{1,80}", model_name):
        raise ValueError("model_name 只能包含字母、数字、下划线、点和短横线")
    if config["feature_mode"] not in {"raw", "technical", "technical_momentum_skip", "overnight_pressure", "impact_liquidity", "information_liquidity", "overnight_market_state", "downside_liquidity_stress", "market_liquidity_state", "conditional_information_state", "stress_transmission", "stress_core", "stress_recovery", "constraint_stress", "value_pressure", "breadth_transition", "beta_idiosyncratic", "asymmetric_impact", "attention_concentration", "stress_asymmetric_impact", "risk_adjusted_impact", "downside_drawdown", "market_relative", "intraday_acceptance", "price_discovery_efficiency", "research_1_031", "research_1_032", "research_1_033", "research_1_034", "research_1_035", "research_1_036", "research_1_037", "research_1_038"}:
        raise ValueError("feature_mode 不在已登记候选中")
    if config.get("evaluation"):
        from walkforward import validate_protocol
        validate_protocol(config)
    elif config["split"] != EXPECTED_SPLIT:
        raise ValueError(f"split 必须严格为 {EXPECTED_SPLIT}")
    if not isinstance(config["model_params"], dict):
        raise ValueError("model_params 必须是 JSON 对象")
    if config.get("financial_logic") is not None:
        from financial_logic import validate_policy
        validate_policy(config)
    return config


def resolve_data_paths(config: dict[str, Any]) -> tuple[Path, Path]:
    """只允许读取题目提供的两份 CSV，拒绝配置指向外部或额外数据。"""
    data_cfg = config.get("data", {})
    configured_train = Path(data_cfg.get("train_path", PROVIDED_TRAIN_PATH)).absolute()
    configured_test = Path(data_cfg.get("test_path", PROVIDED_TEST_PATH)).absolute()
    if configured_train != PROVIDED_TRAIN_PATH or configured_test != PROVIDED_TEST_PATH:
        raise ValueError(
            "数据路径被拒绝：本训练器只允许读取 "
            f"{PROVIDED_TRAIN_PATH} 和 {PROVIDED_TEST_PATH}"
        )
    if not PROVIDED_TRAIN_PATH.exists():
        raise FileNotFoundError(f"训练集不存在: {PROVIDED_TRAIN_PATH}")
    if not PROVIDED_TEST_PATH.exists():
        raise FileNotFoundError(f"测试集不存在: {PROVIDED_TEST_PATH}")
    return PROVIDED_TRAIN_PATH, PROVIDED_TEST_PATH


def print_parameter_confirmation(config: dict[str, Any], train_path: Path, test_path: Path, preview_only: bool = False) -> None:
    from financial_logic import selection_details
    try:
        details = selection_details(config)
        financial_review = {
            "status": "等待用户确认金融逻辑及参数",
            "version": details["version"],
            "registry_fingerprint": details["registry_fingerprint"],
            "feature_count": details["feature_count"],
            "retained": [{k: row[k] for k in
                          ["feature", "logic", "expected_turnover", "failure_risk"]}
                         for row in details["retained"]],
            "excluded": [{"feature": row["feature"], "reason": row["decision_reason"]}
                         for row in details["excluded"]],
            "hypothesis_limit": details["hypothesis_limit"],
        }
        financial_error = None
    except ValueError as exc:
        financial_error = str(exc)
        financial_review = {"status": "禁止训练，旧配置仅供查阅", "reason": financial_error}
    print("\n" + "=" * 78)
    print("训练尚未开始。请先检查逐项金融逻辑、剔除原因、参数和时间划分。")
    print("=" * 78)
    print(json.dumps({
        "model_name": config["model_name"],
        "feature_mode": config["feature_mode"],
        "train_path": str(train_path),
        "test_path": str(test_path),
        "split": config["split"],
        "model_params": config["model_params"],
        "evaluation": config.get("evaluation", {"method": "legacy_fixed_split"}),
        "financial_logic_review": financial_review,
        "save": config.get("save", {}),
    }, ensure_ascii=False, indent=2))
    if financial_error:
        print("\n该配置没有通过金融逻辑门禁；任何旧口令都不能启动本次训练。")
        if preview_only:
            print("仅预览完成：未读取训练/测试数据，未创建模型。")
            return
        raise SystemExit(financial_error)
    print("\n只有确认以上金融逻辑、风险、参数及全部预定拟合后，输入下面口令才会训练：")
    print(config["confirmation_token"])
    if preview_only:
        print("仅预览完成：未读取训练/测试数据，未创建模型。")
        return
    if config.get("auto_confirm") is True:
        print("研究候选已按用户要求自动确认，开始准备数据和训练。")
        return
    answer = input("确认金融逻辑和参数后请输入口令，其他输入都会取消训练：").strip()
    if answer != config["confirmation_token"]:
        raise SystemExit("未收到正确确认口令，训练已取消，未创建模型。")
    print("已确认，开始准备数据和训练。")


def file_manifest(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path),
        "bytes": stat.st_size,
        "last_modified": datetime.fromtimestamp(stat.st_mtime).isoformat(timespec="seconds"),
    }


def audit_source_schema(path: Path, expected_columns: dict[str, str]) -> None:
    header = list(pd.read_csv(path, nrows=0).columns)
    if set(header) != set(expected_columns):
        missing = sorted(set(expected_columns) - set(header))
        unexpected = sorted(set(header) - set(expected_columns))
        raise ValueError(
            f"数据字段不符合题目原始 schema: {path}; missing={missing}; unexpected={unexpected}"
        )


def audit_keys_and_dates(train: pd.DataFrame, test: pd.DataFrame) -> dict[str, Any]:
    for name, frame in [("train", train), ("test", test)]:
        if frame[["ts_code", "trade_date"]].duplicated().any():
            raise ValueError(f"{name} 存在重复的 ts_code + trade_date 键")
        if frame["trade_date"].isna().any() or frame["ts_code"].isna().any():
            raise ValueError(f"{name} 的 ts_code 或 trade_date 存在缺失")
    train_codes = set(train["ts_code"].dropna().astype(str).unique())
    test_codes = set(test["ts_code"].dropna().astype(str).unique())
    if train_codes != test_codes:
        raise ValueError("训练集与测试集股票代码集合不一致，已停止以避免样本范围混淆")
    if int(train["trade_date"].max()) >= int(test["trade_date"].min()):
        raise ValueError("训练集和测试集日期发生重叠，可能造成时间泄漏")
    return {
        "train_duplicate_keys": 0,
        "test_duplicate_keys": 0,
        "same_stock_universe": True,
        "date_ranges_disjoint": True,
    }


def read_inputs(train_path: Path, test_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    audit_source_schema(train_path, TRAIN_DTYPES)
    audit_source_schema(test_path, TEST_DTYPES)
    print(f"读取训练集: {train_path}")
    train = pd.read_csv(
        train_path,
        dtype=TRAIN_DTYPES,
        usecols=list(TRAIN_DTYPES),
    )
    print(f"读取测试集: {test_path}")
    test = pd.read_csv(
        test_path,
        dtype=TEST_DTYPES,
        usecols=list(TEST_DTYPES),
    )
    return train, test


def make_features(train: pd.DataFrame, test: pd.DataFrame, feature_mode: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    research_bases = {
        "research_1_031": ["asymmetric_impact"],
        "research_1_032": ["stress_asymmetric_impact", "asymmetric_impact", "market_relative"],
        "research_1_033": ["risk_adjusted_impact", "asymmetric_impact"],
        "research_1_034": ["stress_transmission"],
        "research_1_035": ["breadth_transition", "asymmetric_impact"],
        "research_1_036": ["stress_asymmetric_impact", "asymmetric_impact"],
        "research_1_037": ["risk_adjusted_impact", "asymmetric_impact"],
        "research_1_038": ["breadth_transition", "risk_adjusted_impact", "asymmetric_impact"],
    }
    if feature_mode in research_bases:
        train_parts, test_parts = [], []
        for base_mode in research_bases[feature_mode]:
            base_train, base_test = make_features(train, test, base_mode)
            train_parts.append(base_train)
            test_parts.append(base_test)
        train_out = train_parts[0].copy()
        test_out = test_parts[0].copy()
        for part in train_parts[1:]:
            train_out = train_out.join(part.drop(columns=[c for c in train_out.columns if c in part.columns], errors="ignore"))
        for part in test_parts[1:]:
            test_out = test_out.join(part.drop(columns=[c for c in test_out.columns if c in part.columns], errors="ignore"))
        return train_out, test_out
    train_x = train.drop(columns=["y_ret_1d"]).copy()
    test_x = test.copy()
    if feature_mode == "raw":
        return train_x, test_x

    # 把训练和测试特征放在一起计算历史窗口，确保测试集起始日能使用训练集末尾历史。
    # 数据按股票和日期升序；pct_change/rolling 只读取当前行及其之前的行。
    # 同一交易日的横截面排名只使用该交易日提供的股票字段，不读取任何标签或外部数据。
    train_x["__source"] = "train"
    train_x["__row_id"] = np.arange(len(train_x), dtype=np.int64)
    test_x["__source"] = "test"
    test_x["__row_id"] = np.arange(len(test_x), dtype=np.int64)
    all_x = pd.concat([train_x, test_x], ignore_index=True)
    all_x = all_x.sort_values(["ts_code", "trade_date"], kind="mergesort").reset_index(drop=True)
    grouped = all_x.groupby("ts_code", sort=False)

    if feature_mode in {"value_pressure", "breadth_transition", "beta_idiosyncratic", "asymmetric_impact", "attention_concentration"}:
        ret_1 = grouped["close"].pct_change(fill_method=None)
        codes = all_x["ts_code"]
        dates = all_x["trade_date"]
        if feature_mode == "value_pressure":
            amount = all_x["amount"].where(all_x["amount"] > 0)
            for window in (5, 20):
                numerator = (ret_1 * amount).groupby(codes, sort=False).transform(lambda s: s.rolling(window, min_periods=window).sum())
                denominator = amount.groupby(codes, sort=False).transform(lambda s: s.rolling(window, min_periods=window).sum())
                all_x[f"amount_weighted_return_{window}"] = numerator / denominator
            log_amount_change = np.log1p(all_x["amount"].clip(lower=0)).groupby(codes, sort=False).diff()
            concordance = (np.sign(ret_1) * np.sign(log_amount_change)).groupby(codes, sort=False).transform(lambda s: s.rolling(20, min_periods=20).mean())
            all_x["price_volume_concordance_20"] = concordance
            all_x["pressure_horizon_spread"] = all_x["amount_weighted_return_5"] - all_x["amount_weighted_return_20"]
            del amount, log_amount_change, concordance, numerator, denominator
        elif feature_mode == "breadth_transition":
            daily = pd.DataFrame({"ret": ret_1, "trade_date": dates}).groupby("trade_date", sort=True)["ret"].agg(
                breadth=lambda s: (s > 0).mean(), dispersion="std"
            )
            daily["breadth_mean_5"] = daily["breadth"].rolling(5, min_periods=5).mean()
            daily["breadth_mean_20"] = daily["breadth"].rolling(20, min_periods=20).mean()
            daily["breadth_change_5"] = daily["breadth"] - daily["breadth"].shift(5)
            daily["dispersion_mean_20"] = daily["dispersion"].rolling(20, min_periods=20).mean()
            daily["breadth_dispersion_alignment"] = daily["breadth_change_5"] * daily["dispersion_mean_20"]
            for target, source in [("market_breadth_mean_5", "breadth_mean_5"), ("market_breadth_mean_20", "breadth_mean_20"), ("market_breadth_change_5", "breadth_change_5"), ("market_dispersion_mean_20", "dispersion_mean_20"), ("breadth_dispersion_alignment", "breadth_dispersion_alignment")]:
                all_x[target] = dates.map(daily[source])
            del daily
        elif feature_mode == "beta_idiosyncratic":
            market_return = ret_1.groupby(dates, sort=False).transform("median")
            mean_stock = ret_1.groupby(codes, sort=False).transform(lambda s: s.rolling(60, min_periods=40).mean())
            mean_market = market_return.groupby(codes, sort=False).transform(lambda s: s.rolling(60, min_periods=40).mean())
            mean_cross = (ret_1 * market_return).groupby(codes, sort=False).transform(lambda s: s.rolling(60, min_periods=40).mean())
            mean_market_sq = market_return.pow(2).groupby(codes, sort=False).transform(lambda s: s.rolling(60, min_periods=40).mean())
            covariance = mean_cross - mean_stock * mean_market
            market_variance = mean_market_sq - mean_market.pow(2)
            beta = covariance / market_variance.where(market_variance > 1e-12)
            all_x["market_beta_60"] = beta
            all_x["market_beta_change_20"] = beta.groupby(codes, sort=False).transform(lambda s: s - s.shift(20))
            residual = ret_1 - beta * market_return
            all_x["idiosyncratic_volatility_20"] = residual.groupby(codes, sort=False).transform(lambda s: s.rolling(20, min_periods=20).std())
            all_x["idiosyncratic_momentum_20"] = residual.groupby(codes, sort=False).transform(lambda s: s.rolling(20, min_periods=20).sum())
            del market_return, mean_stock, mean_market, mean_cross, mean_market_sq, covariance, market_variance, beta, residual
        elif feature_mode == "asymmetric_impact":
            amount = all_x["amount"].where(all_x["amount"] > 0)
            upside_daily = ret_1.where(ret_1 > 0) / amount
            downside_daily = -ret_1.where(ret_1 < 0) / amount
            upside = upside_daily.groupby(codes, sort=False).transform(lambda s: s.rolling(20, min_periods=3).mean())
            downside = downside_daily.groupby(codes, sort=False).transform(lambda s: s.rolling(20, min_periods=3).mean())
            total = upside + downside
            all_x["upside_amihud_20"] = upside
            all_x["downside_amihud_20"] = downside
            all_x["impact_asymmetry_20"] = (downside - upside) / total.where(total > 0)
            all_x["downside_impact_share_20"] = downside / total.where(total > 0)
            del amount, upside_daily, downside_daily, upside, downside, total
        else:
            amount = all_x["amount"].where(all_x["amount"] > 0)
            market_total = amount.groupby(dates, sort=False).transform("sum")
            amount_share = amount / market_total.where(market_total > 0)
            all_x["amount_share_rank"] = amount_share.groupby(dates, sort=False).rank(pct=True)
            hhi = amount_share.pow(2).groupby(dates, sort=False).transform("sum")
            all_x["market_attention_hhi"] = hhi
            amount_average = amount.groupby(codes, sort=False).transform(lambda s: s.rolling(20, min_periods=20).mean())
            amount_ratio = amount / amount_average.where(amount_average > 0)
            all_x["attention_activity_breadth"] = (amount_ratio > 1).astype("float32").groupby(dates, sort=False).transform("mean")
            all_x["attention_return_alignment"] = ret_1 * all_x["amount_share_rank"]
            del amount, market_total, amount_share, hhi, amount_average, amount_ratio
        del grouped, ret_1, train_x, test_x
        sources = all_x.pop("__source").to_numpy()
        row_ids = all_x.pop("__row_id").to_numpy()
        train_order = np.flatnonzero(sources == "train")
        test_order = np.flatnonzero(sources == "test")
        train_order = train_order[np.argsort(row_ids[train_order], kind="stable")]
        test_order = test_order[np.argsort(row_ids[test_order], kind="stable")]
        train_columns, test_columns = {}, {}
        for column in list(all_x.columns):
            series = all_x.pop(column)
            train_columns[column] = series.array.take(train_order)
            test_columns[column] = series.array.take(test_order)
        return pd.DataFrame(train_columns, copy=False), pd.DataFrame(test_columns, copy=False)

    if feature_mode in {"stress_asymmetric_impact", "risk_adjusted_impact"}:
        prev_close = grouped["close"].shift(1)
        ret_1 = grouped["close"].pct_change(fill_method=None)
        dates = all_x["trade_date"]
        codes = all_x["ts_code"]
        all_x["market_breadth"] = ret_1.groupby(dates, sort=False).transform(lambda s: (s > 0).mean())
        all_x["market_dispersion"] = ret_1.groupby(dates, sort=False).transform("std")
        if feature_mode == "risk_adjusted_impact":
            all_x["market_median_return"] = ret_1.groupby(dates, sort=False).transform("median")
        tr = pd.concat([
            (all_x["high"] - all_x["low"]).rename("a"),
            (all_x["high"] - prev_close).abs().rename("b"),
            (all_x["low"] - prev_close).abs().rename("c"),
        ], axis=1).max(axis=1)
        all_x["true_range_pct"] = tr / prev_close
        atr = all_x.groupby("ts_code")["true_range_pct"].transform(
            lambda s: s.rolling(20, min_periods=20).mean()
        )
        all_x["range_shock"] = all_x["true_range_pct"] / atr
        amount = all_x["amount"].where(all_x["amount"] > 0)
        amihud_daily = ret_1.abs() / amount
        all_x["amihud_20"] = amihud_daily.groupby(codes, sort=False).transform(
            lambda s: s.rolling(20, min_periods=20).mean()
        )
        downside_freq = (ret_1 < 0).astype("float32").groupby(codes, sort=False).transform(
            lambda s: s.rolling(20, min_periods=20).mean()
        )
        if feature_mode == "stress_asymmetric_impact":
            all_x["downside_freq_20"] = downside_freq
        upside_daily = ret_1.where(ret_1 > 0) / amount
        downside_daily = -ret_1.where(ret_1 < 0) / amount
        upside_impact = upside_daily.groupby(codes, sort=False).transform(
            lambda s: s.rolling(20, min_periods=3).mean()
        )
        downside_impact = downside_daily.groupby(codes, sort=False).transform(
            lambda s: s.rolling(20, min_periods=3).mean()
        )
        impact_total = upside_impact + downside_impact
        all_x["downside_amihud_20"] = downside_impact
        all_x["impact_asymmetry_20"] = (downside_impact - upside_impact) / impact_total.where(impact_total > 0)
        if feature_mode == "stress_asymmetric_impact":
            all_x["stress_breadth_alignment"] = downside_freq * (1 - all_x["market_breadth"])
            all_x["liquidity_stress_gap"] = all_x["amihud_20"] * (1 - all_x["market_breadth"])
        else:
            market_return = all_x["market_median_return"]
            mean_stock = ret_1.groupby(codes, sort=False).transform(lambda s: s.rolling(60, min_periods=40).mean())
            mean_market = market_return.groupby(codes, sort=False).transform(lambda s: s.rolling(60, min_periods=40).mean())
            mean_cross = (ret_1 * market_return).groupby(codes, sort=False).transform(lambda s: s.rolling(60, min_periods=40).mean())
            mean_market_sq = market_return.pow(2).groupby(codes, sort=False).transform(lambda s: s.rolling(60, min_periods=40).mean())
            covariance = mean_cross - mean_stock * mean_market
            market_variance = mean_market_sq - mean_market.pow(2)
            beta = covariance / market_variance.where(market_variance > 1e-12)
            all_x["market_beta_60"] = beta
            all_x["market_beta_change_20"] = beta.groupby(codes, sort=False).transform(lambda s: s - s.shift(20))
            residual = ret_1 - beta * market_return
            all_x["idiosyncratic_volatility_20"] = residual.groupby(codes, sort=False).transform(lambda s: s.rolling(20, min_periods=20).std())
            all_x["idiosyncratic_momentum_20"] = residual.groupby(codes, sort=False).transform(lambda s: s.rolling(20, min_periods=20).sum())
            del market_return, mean_stock, mean_market, mean_cross, mean_market_sq, covariance, market_variance, beta, residual
        del grouped, prev_close, ret_1, tr, atr, amount, amihud_daily, downside_freq, upside_daily, downside_daily, upside_impact, downside_impact, impact_total, train_x, test_x
        sources = all_x.pop("__source").to_numpy()
        row_ids = all_x.pop("__row_id").to_numpy()
        train_order = np.flatnonzero(sources == "train")
        test_order = np.flatnonzero(sources == "test")
        train_order = train_order[np.argsort(row_ids[train_order], kind="stable")]
        test_order = test_order[np.argsort(row_ids[test_order], kind="stable")]
        train_columns, test_columns = {}, {}
        for column in list(all_x.columns):
            series = all_x.pop(column)
            train_columns[column] = series.array.take(train_order)
            test_columns[column] = series.array.take(test_order)
        return pd.DataFrame(train_columns, copy=False), pd.DataFrame(test_columns, copy=False)

    if feature_mode == "overnight_pressure":
        prev_close = grouped["close"].shift(1)
        all_x["gap_1"] = all_x["open"] / prev_close - 1
        all_x["oc_return"] = all_x["close"] / all_x["open"] - 1
        for column in ["gap_1", "oc_return"]:
            all_x[column + "_rank"] = all_x.groupby("trade_date")[column].rank(pct=True)
        del grouped, prev_close, train_x, test_x
        sources = all_x.pop("__source").to_numpy()
        row_ids = all_x.pop("__row_id").to_numpy()
        train_order = np.flatnonzero(sources == "train")
        test_order = np.flatnonzero(sources == "test")
        train_order = train_order[np.argsort(row_ids[train_order], kind="stable")]
        test_order = test_order[np.argsort(row_ids[test_order], kind="stable")]
        train_columns, test_columns = {}, {}
        for column in list(all_x.columns):
            series = all_x.pop(column)
            train_columns[column] = series.array.take(train_order)
            test_columns[column] = series.array.take(test_order)
            del series
        return pd.DataFrame(train_columns, copy=False), pd.DataFrame(test_columns, copy=False)

    if feature_mode == "impact_liquidity":
        prev_close = grouped["close"].shift(1)
        tr = pd.concat([
            (all_x["high"] - all_x["low"]).rename("a"),
            (all_x["high"] - prev_close).abs().rename("b"),
            (all_x["low"] - prev_close).abs().rename("c"),
        ], axis=1).max(axis=1)
        all_x["true_range_pct"] = tr / prev_close
        all_x["atr_20"] = all_x.groupby("ts_code")["true_range_pct"].transform(
            lambda s: s.rolling(20, min_periods=20).mean()
        )
        all_x["range_shock"] = all_x["true_range_pct"] / all_x["atr_20"]
        ret_1 = grouped["close"].pct_change()
        amihud_1 = ret_1.abs() / all_x["amount"].where(all_x["amount"] > 0)
        all_x["amihud_20"] = amihud_1.groupby(all_x["ts_code"], sort=False).transform(
            lambda s: s.rolling(20, min_periods=20).mean()
        )
        all_x["amihud_20_rank"] = all_x.groupby("trade_date")["amihud_20"].rank(pct=True)
        del grouped, prev_close, tr, ret_1, amihud_1, train_x, test_x
        sources = all_x.pop("__source").to_numpy()
        row_ids = all_x.pop("__row_id").to_numpy()
        train_order = np.flatnonzero(sources == "train")
        test_order = np.flatnonzero(sources == "test")
        train_order = train_order[np.argsort(row_ids[train_order], kind="stable")]
        test_order = test_order[np.argsort(row_ids[test_order], kind="stable")]
        train_columns, test_columns = {}, {}
        for column in list(all_x.columns):
            series = all_x.pop(column)
            train_columns[column] = series.array.take(train_order)
            test_columns[column] = series.array.take(test_order)
            del series
        return pd.DataFrame(train_columns, copy=False), pd.DataFrame(test_columns, copy=False)

    if feature_mode == "information_liquidity":
        # 信息冲击与流动性承接：隔夜重定价、日内压力和价格冲击只使用当前及历史记录。
        prev_close = grouped["close"].shift(1)
        all_x["gap_1"] = all_x["open"] / prev_close - 1
        all_x["oc_return"] = all_x["close"] / all_x["open"] - 1
        tr = pd.concat([
            (all_x["high"] - all_x["low"]).rename("a"),
            (all_x["high"] - prev_close).abs().rename("b"),
            (all_x["low"] - prev_close).abs().rename("c"),
        ], axis=1).max(axis=1)
        all_x["true_range_pct"] = tr / prev_close
        all_x["atr_20"] = all_x.groupby("ts_code")["true_range_pct"].transform(
            lambda s: s.rolling(20, min_periods=20).mean()
        )
        all_x["range_shock"] = all_x["true_range_pct"] / all_x["atr_20"]
        ret_1 = grouped["close"].pct_change()
        amihud_1 = ret_1.abs() / all_x["amount"].where(all_x["amount"] > 0)
        all_x["amihud_20"] = amihud_1.groupby(all_x["ts_code"], sort=False).transform(
            lambda s: s.rolling(20, min_periods=20).mean()
        )
        all_x["amihud_20_rank"] = all_x.groupby("trade_date")["amihud_20"].rank(pct=True)
        del grouped, prev_close, tr, ret_1, amihud_1, train_x, test_x
        sources = all_x.pop("__source").to_numpy()
        row_ids = all_x.pop("__row_id").to_numpy()
        train_order = np.flatnonzero(sources == "train")
        test_order = np.flatnonzero(sources == "test")
        train_order = train_order[np.argsort(row_ids[train_order], kind="stable")]
        test_order = test_order[np.argsort(row_ids[test_order], kind="stable")]
        train_columns, test_columns = {}, {}
        for column in list(all_x.columns):
            series = all_x.pop(column)
            train_columns[column] = series.array.take(train_order)
            test_columns[column] = series.array.take(test_order)
            del series
        return pd.DataFrame(train_columns, copy=False), pd.DataFrame(test_columns, copy=False)

    if feature_mode == "overnight_market_state":
        # 隔夜信息在市场共同状态下的条件效应。
        prev_close = grouped["close"].shift(1)
        ret_1 = grouped["close"].pct_change()
        all_x["gap_1"] = all_x["open"] / prev_close - 1
        all_x["oc_return"] = all_x["close"] / all_x["open"] - 1
        all_x["market_median_return"] = ret_1.groupby(all_x["trade_date"], sort=False).transform("median")
        all_x["market_breadth"] = ret_1.groupby(all_x["trade_date"], sort=False).transform(lambda s: (s > 0).mean())
        all_x["market_dispersion"] = ret_1.groupby(all_x["trade_date"], sort=False).transform("std")
        all_x["residual_return_1"] = ret_1 - all_x["market_median_return"]
        all_x["residual_momentum_20"] = all_x.groupby("ts_code")["residual_return_1"].transform(lambda s: s.rolling(20, min_periods=20).sum())
        stock_vol_20 = ret_1.groupby(all_x["ts_code"], sort=False).transform(lambda s: s.rolling(20, min_periods=20).std())
        all_x["market_relative_volatility"] = stock_vol_20 / all_x["market_dispersion"]
        del grouped, prev_close, ret_1, stock_vol_20, train_x, test_x
        sources = all_x.pop("__source").to_numpy(); row_ids = all_x.pop("__row_id").to_numpy()
        train_order = np.flatnonzero(sources == "train"); test_order = np.flatnonzero(sources == "test")
        train_order = train_order[np.argsort(row_ids[train_order], kind="stable")]; test_order = test_order[np.argsort(row_ids[test_order], kind="stable")]
        train_columns, test_columns = {}, {}
        for column in list(all_x.columns):
            series = all_x.pop(column); train_columns[column] = series.array.take(train_order); test_columns[column] = series.array.take(test_order)
        return pd.DataFrame(train_columns, copy=False), pd.DataFrame(test_columns, copy=False)

    if feature_mode == "downside_liquidity_stress":
        # 下行风险与流动性冲击：检验压力状态而非价格形态。
        prev_close = grouped["close"].shift(1)
        tr = pd.concat([(all_x["high"] - all_x["low"]).rename("a"), (all_x["high"] - prev_close).abs().rename("b"), (all_x["low"] - prev_close).abs().rename("c")], axis=1).max(axis=1)
        all_x["true_range_pct"] = tr / prev_close
        all_x["atr_20"] = all_x.groupby("ts_code")["true_range_pct"].transform(lambda s: s.rolling(20, min_periods=20).mean())
        all_x["range_shock"] = all_x["true_range_pct"] / all_x["atr_20"]
        ret_1 = grouped["close"].pct_change()
        amihud_1 = ret_1.abs() / all_x["amount"].where(all_x["amount"] > 0)
        all_x["amihud_20"] = amihud_1.groupby(all_x["ts_code"], sort=False).transform(lambda s: s.rolling(20, min_periods=20).mean())
        all_x["amihud_20_rank"] = all_x.groupby("trade_date")["amihud_20"].rank(pct=True)
        downside = ret_1.where(ret_1 < 0, 0.0)
        all_x["downside_vol_20"] = downside.pow(2).groupby(all_x["ts_code"], sort=False).transform(lambda s: s.rolling(20, min_periods=20).mean()).pow(0.5)
        all_x["downside_freq_20"] = (ret_1 < 0).astype("float32").groupby(all_x["ts_code"], sort=False).transform(lambda s: s.rolling(20, min_periods=20).mean())
        rolling_max = all_x.groupby("ts_code")["close"].transform(lambda s: s.rolling(20, min_periods=20).max())
        all_x["drawdown_20"] = all_x["close"] / rolling_max - 1
        del grouped, prev_close, tr, ret_1, amihud_1, downside, rolling_max, train_x, test_x
        sources = all_x.pop("__source").to_numpy(); row_ids = all_x.pop("__row_id").to_numpy()
        train_order = np.flatnonzero(sources == "train"); test_order = np.flatnonzero(sources == "test")
        train_order = train_order[np.argsort(row_ids[train_order], kind="stable")]; test_order = test_order[np.argsort(row_ids[test_order], kind="stable")]
        train_columns, test_columns = {}, {}
        for column in list(all_x.columns):
            series = all_x.pop(column); train_columns[column] = series.array.take(train_order); test_columns[column] = series.array.take(test_order)
        return pd.DataFrame(train_columns, copy=False), pd.DataFrame(test_columns, copy=False)

    if feature_mode == "market_liquidity_state":
        # 市场状态与个股流动性冲击的条件组合。
        prev_close = grouped["close"].shift(1); ret_1 = grouped["close"].pct_change()
        all_x["market_median_return"] = ret_1.groupby(all_x["trade_date"], sort=False).transform("median")
        all_x["market_breadth"] = ret_1.groupby(all_x["trade_date"], sort=False).transform(lambda s: (s > 0).mean())
        all_x["market_dispersion"] = ret_1.groupby(all_x["trade_date"], sort=False).transform("std")
        stock_vol_20 = ret_1.groupby(all_x["ts_code"], sort=False).transform(lambda s: s.rolling(20, min_periods=20).std())
        all_x["market_relative_volatility"] = stock_vol_20 / all_x["market_dispersion"]
        tr = pd.concat([(all_x["high"] - all_x["low"]).rename("a"), (all_x["high"] - prev_close).abs().rename("b"), (all_x["low"] - prev_close).abs().rename("c")], axis=1).max(axis=1)
        all_x["true_range_pct"] = tr / prev_close
        all_x["atr_20"] = all_x.groupby("ts_code")["true_range_pct"].transform(lambda s: s.rolling(20, min_periods=20).mean())
        all_x["range_shock"] = all_x["true_range_pct"] / all_x["atr_20"]
        amihud_1 = ret_1.abs() / all_x["amount"].where(all_x["amount"] > 0)
        all_x["amihud_20"] = amihud_1.groupby(all_x["ts_code"], sort=False).transform(lambda s: s.rolling(20, min_periods=20).mean())
        all_x["amihud_20_rank"] = all_x.groupby("trade_date")["amihud_20"].rank(pct=True)
        del grouped, prev_close, ret_1, stock_vol_20, tr, amihud_1, train_x, test_x
        sources = all_x.pop("__source").to_numpy(); row_ids = all_x.pop("__row_id").to_numpy()
        train_order = np.flatnonzero(sources == "train"); test_order = np.flatnonzero(sources == "test")
        train_order = train_order[np.argsort(row_ids[train_order], kind="stable")]; test_order = test_order[np.argsort(row_ids[test_order], kind="stable")]
        train_columns, test_columns = {}, {}
        for column in list(all_x.columns):
            series = all_x.pop(column); train_columns[column] = series.array.take(train_order); test_columns[column] = series.array.take(test_order)
        return pd.DataFrame(train_columns, copy=False), pd.DataFrame(test_columns, copy=False)

    if feature_mode == "conditional_information_state":
        prev_close = grouped["close"].shift(1); ret_1 = grouped["close"].pct_change()
        all_x["gap_1"] = all_x["open"] / prev_close - 1; all_x["oc_return"] = all_x["close"] / all_x["open"] - 1
        all_x["market_median_return"] = ret_1.groupby(all_x["trade_date"], sort=False).transform("median")
        all_x["market_breadth"] = ret_1.groupby(all_x["trade_date"], sort=False).transform(lambda s: (s > 0).mean())
        all_x["market_dispersion"] = ret_1.groupby(all_x["trade_date"], sort=False).transform("std")
        all_x["residual_return_1"] = ret_1 - all_x["market_median_return"]
        all_x["residual_momentum_20"] = all_x.groupby("ts_code")["residual_return_1"].transform(lambda s: s.rolling(20, min_periods=20).sum())
        stock_vol_20 = ret_1.groupby(all_x["ts_code"], sort=False).transform(lambda s: s.rolling(20, min_periods=20).std())
        all_x["market_relative_volatility"] = stock_vol_20 / all_x["market_dispersion"]
        all_x["gap_market_alignment"] = all_x["gap_1"] * (2 * all_x["market_breadth"] - 1)
        prev_tr = pd.concat([(all_x["high"] - all_x["low"]).rename("a"), (all_x["high"] - prev_close).abs().rename("b"), (all_x["low"] - prev_close).abs().rename("c")], axis=1).max(axis=1) / prev_close
        atr = prev_tr.groupby(all_x["ts_code"], sort=False).transform(lambda s: s.rolling(20, min_periods=20).mean())
        all_x["shock_market_alignment"] = (prev_tr / atr) * all_x["market_dispersion"]
        del grouped, prev_close, ret_1, stock_vol_20, prev_tr, atr, train_x, test_x
        sources = all_x.pop("__source").to_numpy(); row_ids = all_x.pop("__row_id").to_numpy(); train_order = np.flatnonzero(sources == "train"); test_order = np.flatnonzero(sources == "test"); train_order = train_order[np.argsort(row_ids[train_order], kind="stable")]; test_order = test_order[np.argsort(row_ids[test_order], kind="stable")]; train_columns, test_columns = {}, {}
        for column in list(all_x.columns):
            series = all_x.pop(column); train_columns[column] = series.array.take(train_order); test_columns[column] = series.array.take(test_order)
        return pd.DataFrame(train_columns, copy=False), pd.DataFrame(test_columns, copy=False)

    if feature_mode == "stress_transmission":
        prev_close = grouped["close"].shift(1); ret_1 = grouped["close"].pct_change()
        all_x["market_median_return"] = ret_1.groupby(all_x["trade_date"], sort=False).transform("median"); all_x["market_breadth"] = ret_1.groupby(all_x["trade_date"], sort=False).transform(lambda s: (s > 0).mean()); all_x["market_dispersion"] = ret_1.groupby(all_x["trade_date"], sort=False).transform("std")
        tr = pd.concat([(all_x["high"] - all_x["low"]).rename("a"), (all_x["high"] - prev_close).abs().rename("b"), (all_x["low"] - prev_close).abs().rename("c")], axis=1).max(axis=1); all_x["true_range_pct"] = tr / prev_close; all_x["atr_20"] = all_x.groupby("ts_code")["true_range_pct"].transform(lambda s: s.rolling(20, min_periods=20).mean()); all_x["range_shock"] = all_x["true_range_pct"] / all_x["atr_20"]
        amihud_1 = ret_1.abs() / all_x["amount"].where(all_x["amount"] > 0); all_x["amihud_20"] = amihud_1.groupby(all_x["ts_code"], sort=False).transform(lambda s: s.rolling(20, min_periods=20).mean()); all_x["amihud_20_rank"] = all_x.groupby("trade_date")["amihud_20"].rank(pct=True)
        downside = ret_1.where(ret_1 < 0, 0.0); all_x["downside_vol_20"] = downside.pow(2).groupby(all_x["ts_code"], sort=False).transform(lambda s: s.rolling(20, min_periods=20).mean()).pow(0.5); all_x["downside_freq_20"] = (ret_1 < 0).astype("float32").groupby(all_x["ts_code"], sort=False).transform(lambda s: s.rolling(20, min_periods=20).mean()); rolling_max = all_x.groupby("ts_code")["close"].transform(lambda s: s.rolling(20, min_periods=20).max()); all_x["drawdown_20"] = all_x["close"] / rolling_max - 1
        all_x["stress_breadth_alignment"] = all_x["downside_freq_20"] * (1 - all_x["market_breadth"]); all_x["liquidity_stress_gap"] = all_x["amihud_20"] * (1 - all_x["market_breadth"])
        del grouped, prev_close, ret_1, tr, amihud_1, downside, rolling_max, train_x, test_x
        sources = all_x.pop("__source").to_numpy(); row_ids = all_x.pop("__row_id").to_numpy(); train_order = np.flatnonzero(sources == "train"); test_order = np.flatnonzero(sources == "test"); train_order = train_order[np.argsort(row_ids[train_order], kind="stable")]; test_order = test_order[np.argsort(row_ids[test_order], kind="stable")]; train_columns, test_columns = {}, {}
        for column in list(all_x.columns):
            series = all_x.pop(column); train_columns[column] = series.array.take(train_order); test_columns[column] = series.array.take(test_order)
        return pd.DataFrame(train_columns, copy=False), pd.DataFrame(test_columns, copy=False)

    if feature_mode in {"stress_core", "stress_recovery", "constraint_stress"}:
        prev_close = grouped["close"].shift(1); ret_1 = grouped["close"].pct_change()
        all_x["market_breadth"] = ret_1.groupby(all_x["trade_date"], sort=False).transform(lambda s: (s > 0).mean())
        all_x["market_dispersion"] = ret_1.groupby(all_x["trade_date"], sort=False).transform("std")
        tr = pd.concat([(all_x["high"] - all_x["low"]).rename("a"), (all_x["high"] - prev_close).abs().rename("b"), (all_x["low"] - prev_close).abs().rename("c")], axis=1).max(axis=1)
        all_x["true_range_pct"] = tr / prev_close
        atr = all_x.groupby("ts_code")["true_range_pct"].transform(lambda s: s.rolling(20, min_periods=20).mean())
        all_x["range_shock"] = all_x["true_range_pct"] / atr
        amihud_1 = ret_1.abs() / all_x["amount"].where(all_x["amount"] > 0)
        all_x["amihud_20"] = amihud_1.groupby(all_x["ts_code"], sort=False).transform(lambda s: s.rolling(20, min_periods=20).mean())
        if feature_mode == "stress_core":
            all_x["amihud_20_rank"] = all_x.groupby("trade_date")["amihud_20"].rank(pct=True)
            downside = ret_1.where(ret_1 < 0, 0.0)
            all_x["downside_freq_20"] = (ret_1 < 0).astype("float32").groupby(all_x["ts_code"], sort=False).transform(lambda s: s.rolling(20, min_periods=20).mean())
            all_x["stress_breadth_alignment"] = all_x["downside_freq_20"] * (1 - all_x["market_breadth"])
            all_x["liquidity_stress_gap"] = all_x["amihud_20"] * (1 - all_x["market_breadth"])
        elif feature_mode == "stress_recovery":
            all_x["market_median_return"] = ret_1.groupby(all_x["trade_date"], sort=False).transform("median")
            downside = ret_1.where(ret_1 < 0, 0.0)
            all_x["downside_vol_20"] = downside.pow(2).groupby(all_x["ts_code"], sort=False).transform(lambda s: s.rolling(20, min_periods=20).mean()).pow(0.5)
            all_x["downside_freq_20"] = (ret_1 < 0).astype("float32").groupby(all_x["ts_code"], sort=False).transform(lambda s: s.rolling(20, min_periods=20).mean())
            rolling_max = all_x.groupby("ts_code")["close"].transform(lambda s: s.rolling(20, min_periods=20).max())
            rolling_min = all_x.groupby("ts_code")["close"].transform(lambda s: s.rolling(20, min_periods=20).min())
            all_x["drawdown_20"] = all_x["close"] / rolling_max - 1; all_x["recovery_from_low_20"] = all_x["close"] / rolling_min - 1
            all_x["pressure_recovery_alignment"] = all_x["recovery_from_low_20"] * (1 - all_x["market_breadth"])
        else:
            all_x["downside_freq_20"] = (ret_1 < 0).astype("float32").groupby(all_x["ts_code"], sort=False).transform(lambda s: s.rolling(20, min_periods=20).mean())
            all_x["limit_stress_alignment"] = (all_x["flag_limit_up"] - all_x["flag_limit_down"]) * (1 - all_x["market_breadth"])
            all_x["stress_dispersion_alignment"] = all_x["range_shock"] * all_x["market_dispersion"]
        del grouped, prev_close, ret_1, tr, atr, amihud_1, train_x, test_x
        sources = all_x.pop("__source").to_numpy(); row_ids = all_x.pop("__row_id").to_numpy(); train_order = np.flatnonzero(sources == "train"); test_order = np.flatnonzero(sources == "test"); train_order = train_order[np.argsort(row_ids[train_order], kind="stable")]; test_order = test_order[np.argsort(row_ids[test_order], kind="stable")]; train_columns, test_columns = {}, {}
        for column in list(all_x.columns):
            series = all_x.pop(column); train_columns[column] = series.array.take(train_order); test_columns[column] = series.array.take(test_order)
        return pd.DataFrame(train_columns, copy=False), pd.DataFrame(test_columns, copy=False)

    if feature_mode == "downside_drawdown":
        ret_1 = grouped["close"].pct_change()
        for window in [20, 60]:
            downside = ret_1.where(ret_1 < 0, 0.0)
            all_x[f"downside_vol_{window}"] = downside.pow(2).groupby(all_x["ts_code"], sort=False).transform(
                lambda s: s.rolling(window, min_periods=window).mean()
            ).pow(0.5)
            all_x[f"downside_freq_{window}"] = (ret_1 < 0).astype("float32").groupby(
                all_x["ts_code"], sort=False
            ).transform(lambda s: s.rolling(window, min_periods=window).mean())
            rolling_max = all_x.groupby("ts_code")["close"].transform(
                lambda s: s.rolling(window, min_periods=window).max()
            )
            all_x[f"drawdown_{window}"] = all_x["close"] / rolling_max - 1
        rank_columns = [
            "downside_vol_20", "downside_freq_20", "drawdown_20",
            "downside_vol_60", "downside_freq_60", "drawdown_60",
        ]
        for column in rank_columns:
            all_x[column + "_rank"] = all_x.groupby("trade_date")[column].rank(pct=True)
        del grouped, ret_1, downside, rolling_max, train_x, test_x
        sources = all_x.pop("__source").to_numpy()
        row_ids = all_x.pop("__row_id").to_numpy()
        train_order = np.flatnonzero(sources == "train")
        test_order = np.flatnonzero(sources == "test")
        train_order = train_order[np.argsort(row_ids[train_order], kind="stable")]
        test_order = test_order[np.argsort(row_ids[test_order], kind="stable")]
        train_columns, test_columns = {}, {}
        for column in list(all_x.columns):
            series = all_x.pop(column)
            train_columns[column] = series.array.take(train_order)
            test_columns[column] = series.array.take(test_order)
            del series
        return pd.DataFrame(train_columns, copy=False), pd.DataFrame(test_columns, copy=False)

    if feature_mode == "market_relative":
        ret_1 = grouped["close"].pct_change()
        all_x["market_median_return"] = ret_1.groupby(all_x["trade_date"], sort=False).transform("median")
        all_x["market_breadth"] = ret_1.groupby(all_x["trade_date"], sort=False).transform(lambda s: (s > 0).mean())
        all_x["market_dispersion"] = ret_1.groupby(all_x["trade_date"], sort=False).transform("std")
        all_x["residual_return_1"] = ret_1 - all_x["market_median_return"]
        all_x["residual_momentum_20"] = all_x.groupby("ts_code")["residual_return_1"].transform(
            lambda s: s.rolling(20, min_periods=20).sum()
        )
        stock_vol_20 = ret_1.groupby(all_x["ts_code"], sort=False).transform(
            lambda s: s.rolling(20, min_periods=20).std()
        )
        all_x["market_relative_volatility"] = stock_vol_20 / all_x["market_dispersion"]
        del grouped, ret_1, stock_vol_20, train_x, test_x
        sources = all_x.pop("__source").to_numpy()
        row_ids = all_x.pop("__row_id").to_numpy()
        train_order = np.flatnonzero(sources == "train")
        test_order = np.flatnonzero(sources == "test")
        train_order = train_order[np.argsort(row_ids[train_order], kind="stable")]
        test_order = test_order[np.argsort(row_ids[test_order], kind="stable")]
        train_columns, test_columns = {}, {}
        for column in list(all_x.columns):
            series = all_x.pop(column)
            train_columns[column] = series.array.take(train_order)
            test_columns[column] = series.array.take(test_order)
            del series
        return pd.DataFrame(train_columns, copy=False), pd.DataFrame(test_columns, copy=False)

    if feature_mode == "intraday_acceptance":
        # 日内价格接受与交易参与：仅使用当日及之前的OHLCV字段。
        all_x["hl_range"] = (all_x["high"] - all_x["low"]) / all_x["close"].replace(0, np.nan)
        all_x["close_position"] = (
            (all_x["close"] - all_x["low"]) /
            (all_x["high"] - all_x["low"]).replace(0, np.nan)
        )
        all_x["amount_per_vol"] = all_x["amount"] / all_x["vol"].replace(0, np.nan)
        vol_ma = all_x.groupby("ts_code")["vol"].transform(
            lambda s: s.rolling(20, min_periods=20).mean()
        )
        amount_ma = all_x.groupby("ts_code")["amount"].transform(
            lambda s: s.rolling(20, min_periods=20).mean()
        )
        all_x["vol_ratio_20"] = all_x["vol"] / vol_ma
        all_x["amount_ratio_20"] = all_x["amount"] / amount_ma
        del grouped, train_x, test_x, vol_ma, amount_ma
        sources = all_x.pop("__source").to_numpy()
        row_ids = all_x.pop("__row_id").to_numpy()
        train_order = np.flatnonzero(sources == "train")
        test_order = np.flatnonzero(sources == "test")
        train_order = train_order[np.argsort(row_ids[train_order], kind="stable")]
        test_order = test_order[np.argsort(row_ids[test_order], kind="stable")]
        train_columns, test_columns = {}, {}
        for column in list(all_x.columns):
            series = all_x.pop(column)
            train_columns[column] = series.array.take(train_order)
            test_columns[column] = series.array.take(test_order)
            del series
        return pd.DataFrame(train_columns, copy=False), pd.DataFrame(test_columns, copy=False)

    if feature_mode == "price_discovery_efficiency":
        # 价格发现效率与买卖拒绝：只使用当日及历史OHLCV字段。
        intraday_range = (all_x["high"] - all_x["low"]).replace(0, np.nan)
        body = (all_x["close"] - all_x["open"]).abs()
        all_x["body_range_ratio"] = body / intraday_range
        all_x["upper_shadow_ratio"] = (all_x["high"] - all_x[["open", "close"]].max(axis=1)) / intraday_range
        all_x["lower_shadow_ratio"] = (all_x[["open", "close"]].min(axis=1) - all_x["low"]) / intraday_range
        ret_1 = grouped["close"].pct_change()
        path = ret_1.abs().groupby(all_x["ts_code"], sort=False).transform(
            lambda s: s.rolling(20, min_periods=20).sum()
        )
        net = grouped["close"].pct_change(20).abs()
        all_x["path_efficiency_20"] = net / path.replace(0, np.nan)
        all_x["return_autocorrelation_20"] = grouped["close"].transform(
            lambda s: s.pct_change().rolling(20, min_periods=20).corr(s.pct_change().shift(1))
        )
        del grouped, train_x, test_x, intraday_range, body, ret_1, path, net
        sources = all_x.pop("__source").to_numpy()
        row_ids = all_x.pop("__row_id").to_numpy()
        train_order = np.flatnonzero(sources == "train")
        test_order = np.flatnonzero(sources == "test")
        train_order = train_order[np.argsort(row_ids[train_order], kind="stable")]
        test_order = test_order[np.argsort(row_ids[test_order], kind="stable")]
        train_columns, test_columns = {}, {}
        for column in list(all_x.columns):
            series = all_x.pop(column)
            train_columns[column] = series.array.take(train_order)
            test_columns[column] = series.array.take(test_order)
            del series
        return pd.DataFrame(train_columns, copy=False), pd.DataFrame(test_columns, copy=False)

    all_x["oc_return"] = all_x["close"] / all_x["open"] - 1
    all_x["hl_range"] = (all_x["high"] - all_x["low"]) / all_x["close"]
    all_x["close_position"] = (
        (all_x["close"] - all_x["low"]) /
        (all_x["high"] - all_x["low"]).replace(0, np.nan)
    )
    all_x["amount_per_vol"] = all_x["amount"] / all_x["vol"].replace(0, np.nan)
    all_x["ret_1"] = grouped["close"].pct_change()

    for window in [3, 5, 10, 20, 60]:
        all_x[f"ret_{window}"] = grouped["close"].pct_change(window)
        all_x[f"volatility_{window}"] = (
            all_x.groupby("ts_code")["ret_1"]
            .transform(lambda s: s.rolling(window, min_periods=window).std())
        )
        close_ma = (
            all_x.groupby("ts_code")["close"]
            .transform(lambda s: s.rolling(window, min_periods=window).mean())
        )
        all_x[f"ma_gap_{window}"] = all_x["close"] / close_ma - 1

    for window in [5, 20]:
        vol_ma = (
            all_x.groupby("ts_code")["vol"]
            .transform(lambda s: s.rolling(window, min_periods=window).mean())
        )
        amount_ma = (
            all_x.groupby("ts_code")["amount"]
            .transform(lambda s: s.rolling(window, min_periods=window).mean())
        )
        all_x[f"vol_ratio_{window}"] = all_x["vol"] / vol_ma
        all_x[f"amount_ratio_{window}"] = all_x["amount"] / amount_ma

    if feature_mode == "technical_momentum_skip":
        # 参考 Jegadeesh--Titman 的中期相对强度信号：
        # 形成期分别取约 1、3、6、12 个月，并跳过最近 5 个交易日。
        # 所有 shift 都在单只股票组内进行，只使用当前行之前的数据。
        for window in [120, 240]:
            all_x[f"ret_{window}"] = grouped["close"].pct_change(window)
        close_lag_5 = grouped["close"].shift(5)
        for window in [20, 60, 120, 240]:
            close_lag_start = grouped["close"].shift(window + 5)
            all_x[f"ret_{window}_skip5"] = close_lag_5 / close_lag_start - 1

    rank_columns = ["ret_1", "ret_5", "ret_20", "volatility_20", "amount_ratio_20"]
    if feature_mode == "technical_momentum_skip":
        rank_columns.extend([
            "ret_60", "ret_120", "ret_240",
            "ret_20_skip5", "ret_60_skip5", "ret_120_skip5", "ret_240_skip5",
        ])
    for column in rank_columns:
        all_x[column + "_rank"] = all_x.groupby("trade_date")[column].rank(pct=True)

    # Memory-only change: gather each column into the original row order once.
    # Keep feature arithmetic and dtypes identical; avoid sorting/deep-copying
    # the entire 50-column panel repeatedly after its calculation.
    del grouped, train_x, test_x, close_ma, vol_ma, amount_ma
    if feature_mode == "technical_momentum_skip":
        del close_lag_5, close_lag_start
    sources = all_x.pop("__source").to_numpy()
    row_ids = all_x.pop("__row_id").to_numpy()
    train_order = np.flatnonzero(sources == "train")
    test_order = np.flatnonzero(sources == "test")
    train_order = train_order[np.argsort(row_ids[train_order], kind="stable")]
    test_order = test_order[np.argsort(row_ids[test_order], kind="stable")]
    train_columns, test_columns = {}, {}
    for column in list(all_x.columns):
        series = all_x.pop(column)
        train_columns[column] = series.array.take(train_order)
        test_columns[column] = series.array.take(test_order)
        del series
    train_features = pd.DataFrame(train_columns, copy=False)
    test_features = pd.DataFrame(test_columns, copy=False)
    return train_features, test_features


def numeric_feature_columns(train_features: pd.DataFrame) -> list[str]:
    excluded = {"ts_code", "trade_date"}
    columns = [c for c in train_features.columns if c not in excluded]
    forbidden = [
        c for c in columns
        if any(token in c.lower() for token in FORBIDDEN_FEATURE_TOKENS)
    ]
    if forbidden:
        raise ValueError(f"禁止的标签/未来信息/旧模型字段混入特征: {forbidden}")
    bad = [c for c in columns if not pd.api.types.is_numeric_dtype(train_features[c])]
    if bad:
        raise TypeError(f"以下特征不是数值型: {bad}")
    return columns


def select_training_features(train_features: pd.DataFrame, config: dict[str, Any]) -> list[str]:
    """Only prospectively reviewed financial hypotheses may enter model inputs."""
    from financial_logic import select_columns
    return select_columns(numeric_feature_columns(train_features), config)


def safe_spearman(x: pd.Series, y: pd.Series) -> float:
    if len(x) < 2 or x.nunique(dropna=True) < 2 or y.nunique(dropna=True) < 2:
        return float("nan")
    value, _ = spearmanr(x, y)
    return float(value) if np.isfinite(value) else float("nan")


def competition_metrics(predictions: pd.DataFrame) -> dict[str, float | int | None]:
    ic_values: list[float] = []
    excess_values: list[float] = []
    for _, group in predictions.groupby("trade_date", sort=True):
        valid = group.dropna(subset=["pred", "y_ret_1d"])
        ic = safe_spearman(valid["pred"], valid["y_ret_1d"])
        if np.isfinite(ic):
            ic_values.append(ic)

        investable = group[(group["flag_limit_up"] == 0) & group["y_ret_1d"].notna()].copy()
        if len(investable) >= 100:
            investable = investable.sort_values("pred", ascending=False).reset_index(drop=True)
            n_top = max(len(investable) // 10, 1)
            excess_values.append(float(investable.iloc[:n_top]["y_ret_1d"].mean() - investable["y_ret_1d"].mean()))

    turnover_values: list[float] = []
    previous: set[str] | None = None
    for _, group in predictions.groupby("trade_date", sort=True):
        investable = group[group["flag_limit_up"] == 0].sort_values("pred", ascending=False)
        if len(investable) < 100:
            previous = None
            continue
        n_top = max(len(investable) // 10, 1)
        current = set(investable.iloc[:n_top]["ts_code"].astype(str))
        if previous:
            union = current | previous
            turnover_values.append(1.0 - len(current & previous) / len(union))
        previous = current

    labeled = predictions.dropna(subset=["y_ret_1d", "pred"])
    rmse = float(np.sqrt(np.mean((labeled["pred"] - labeled["y_ret_1d"]) ** 2))) if len(labeled) else float("nan")
    mae = float(np.mean(np.abs(labeled["pred"] - labeled["y_ret_1d"]))) if len(labeled) else float("nan")
    ic_mean = float(np.nanmean(ic_values)) if ic_values else float("nan")
    ic_std = float(np.nanstd(ic_values, ddof=1)) if len(ic_values) > 1 else float("nan")
    annual_excess = float(np.mean(excess_values) * 252) if excess_values else float("nan")
    mean_turnover = float(np.mean(turnover_values)) if turnover_values else float("nan")
    final_score = (
        ic_mean * 0.4 + annual_excess * 0.3 + (1 - mean_turnover) * 0.3
        if all(np.isfinite(v) for v in [ic_mean, annual_excess, mean_turnover])
        else float("nan")
    )
    return {
        "rmse": rmse,
        "mae": mae,
        "ic_mean": ic_mean,
        "ic_std": ic_std,
        "ic_positive_ratio": float(np.mean(np.array(ic_values) > 0)) if ic_values else float("nan"),
        "annual_excess": annual_excess,
        "mean_turnover": mean_turnover,
        "stability_score": 1 - mean_turnover if np.isfinite(mean_turnover) else float("nan"),
        "final_score": final_score,
        "ic_days": len(ic_values),
        "excess_days": len(excess_values),
        "turnover_days": len(turnover_values),
    }


def write_json(path: Path, value: Any) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(value, f, ensure_ascii=False, indent=2, default=str)


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).absolute()
    config = load_config(config_path)
    train_path, test_path = resolve_data_paths(config)

    # 只展示参数并等待确认；确认前不读取大 CSV，不创建模型。
    print_parameter_confirmation(config, train_path, test_path, preview_only=args.preview)
    if args.preview:
        return
    if config.get("evaluation"):
        from walkforward import run_walk_forward
        run_walk_forward(config, train_path, test_path, WORKSPACE)
        return

    started = time.perf_counter()
    train, test = read_inputs(train_path, test_path)
    audit = audit_keys_and_dates(train, test)
    split = config["split"]
    if int(train["trade_date"].min()) > split["train_start"] or int(train["trade_date"].max()) < split["valid_end"]:
        raise ValueError("训练集日期范围不能覆盖要求的训练/验证区间")
    if int(test["trade_date"].min()) > split["test_start"] or int(test["trade_date"].max()) < split["test_end"]:
        raise ValueError("测试集日期范围不能覆盖要求的测试区间")

    train_features, test_features = make_features(train, test, config["feature_mode"])
    features = select_training_features(train_features, config)
    if "y_ret_1d" in features or "pred" in features:
        raise ValueError("标签或预测值混入特征，已停止")

    train_all = train_features.copy()
    train_all["y_ret_1d"] = train["y_ret_1d"].to_numpy()
    fit_mask = (
        train_all["trade_date"].between(split["train_start"], split["train_end"])
        & train_all["y_ret_1d"].notna()
    )
    valid_mask = train_all["trade_date"].between(split["valid_start"], split["valid_end"])
    fit = train_all.loc[fit_mask]
    valid = train_all.loc[valid_mask]
    if len(fit) == 0 or len(valid) == 0:
        raise ValueError("训练或验证样本为空")

    forbidden_training_params = {"init_model", "old_model", "previous_model", "external_data"}
    supplied_forbidden = sorted(forbidden_training_params.intersection(config["model_params"]))
    if supplied_forbidden:
        raise ValueError(f"禁止使用前代模型或外部数据参数: {supplied_forbidden}")

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_dir = WORKSPACE / "runs" / f"{config['model_name']}_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=False)
    write_json(run_dir / "config_snapshot.json", config)
    from financial_logic import write_selection_snapshot
    write_selection_snapshot(run_dir / "financial_logic_snapshot.json", config)

    print(f"训练样本: {len(fit):,}; 验证样本: {len(valid):,}; 特征数: {len(features)}")
    print("创建全新的 LGBMRegressor；不加载任何旧模型。")
    valid_eval = valid[valid["y_ret_1d"].notna()]
    model = lgb.LGBMRegressor(**config["model_params"])
    model.fit(
        fit[features],
        fit["y_ret_1d"],
        eval_set=[(valid_eval[features], valid_eval["y_ret_1d"])],
        eval_metric="rmse",
        callbacks=[lgb.early_stopping(100), lgb.log_evaluation(100)],
        init_model=None,
    )

    valid_pred = model.predict(valid[features], num_iteration=model.best_iteration_)
    test_pred = model.predict(test_features[features], num_iteration=model.best_iteration_)
    valid_out = valid[["ts_code", "trade_date", "flag_limit_up", "y_ret_1d"]].copy()
    valid_out["pred"] = np.nan_to_num(valid_pred, nan=0.0, posinf=0.0, neginf=0.0)
    test_out = test[["ts_code", "trade_date"]].copy()
    test_out["pred"] = np.nan_to_num(test_pred, nan=0.0, posinf=0.0, neginf=0.0)
    if not test_out[["ts_code", "trade_date"]].reset_index(drop=True).equals(
        test[["ts_code", "trade_date"]].reset_index(drop=True)
    ):
        raise ValueError("submission 键与测试集不一致，已停止保存")
    if test_out[["ts_code", "trade_date"]].duplicated().any():
        raise ValueError("submission 存在重复键，已停止保存")
    if not np.isfinite(test_out["pred"].to_numpy()).all():
        raise ValueError("submission pred 存在 NaN 或无穷大，已停止保存")
    metrics = competition_metrics(valid_out)

    model.booster_.save_model(str(run_dir / "model.txt"))
    valid_out.to_csv(run_dir / "validation_predictions.csv", index=False)
    test_out.to_csv(run_dir / "submission.csv", index=False)
    importance = pd.DataFrame({
        "feature": features,
        "importance_gain": model.booster_.feature_importance(importance_type="gain"),
        "importance_split": model.booster_.feature_importance(importance_type="split"),
    }).sort_values("importance_gain", ascending=False)
    importance.to_csv(run_dir / "feature_importance.csv", index=False)

    save_cfg = config.get("save", {})
    sample_rows = int(save_cfg.get("sample_rows", 1000))
    train_all.head(sample_rows).to_csv(run_dir / "data_train_sample.csv", index=False)
    valid.head(sample_rows).to_csv(run_dir / "data_valid_sample.csv", index=False)
    test_features.head(sample_rows).to_csv(run_dir / "data_test_sample.csv", index=False)
    write_json(run_dir / "feature_columns.json", features)
    write_json(run_dir / "data_manifest.json", {
        "train_source": file_manifest(train_path),
        "test_source": file_manifest(test_path),
        "train_rows": len(train),
        "test_rows": len(test),
        "train_date_min": int(train["trade_date"].min()),
        "train_date_max": int(train["trade_date"].max()),
        "test_date_min": int(test["trade_date"].min()),
        "test_date_max": int(test["trade_date"].max()),
    })

    elapsed = time.perf_counter() - started
    summary = {
        "run_id": run_id,
        "model_name": config["model_name"],
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "training_seconds": round(elapsed, 3),
        "feature_mode": config["feature_mode"],
        "feature_count": len(features),
        "feature_columns": features,
        "optimization_plan": config.get("optimization_plan"),
        "research_references": config.get("research_references", []),
        "model_params": config["model_params"],
        "split": split,
        "rows": {"fit": len(fit), "valid": len(valid), "test": len(test)},
        "missing_y_train": int(train["y_ret_1d"].isna().sum()),
        "audit": audit,
        "constraints": {
            "source_paths_whitelisted": True,
            "source_schema_exact": True,
            "only_provided_columns": True,
            "future_data_policy": "current_and_past_rows_only",
            "fixed_time_split": True,
            "test_labels_used": False,
            "old_model_loaded": False,
            "init_model": None,
            "submission_alignment_checked": True,
            "submission_finite_checked": True,
        },
        "best_iteration": int(model.best_iteration_ or config["model_params"].get("n_estimators", 0)),
        "metrics": metrics,
        "run_dir": str(run_dir),
        "artifacts": {
            "model": str(run_dir / "model.txt"),
            "validation_predictions": str(run_dir / "validation_predictions.csv"),
            "submission": str(run_dir / "submission.csv"),
            "feature_importance": str(run_dir / "feature_importance.csv"),
            "data_manifest": str(run_dir / "data_manifest.json"),
        },
    }
    write_json(run_dir / "summary.json", summary)
    rebuild_report(WORKSPACE)

    print("\n训练完成。")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"模型: {run_dir / 'model.txt'}")
    print(f"提交文件: {run_dir / 'submission.csv'}")
    print(f"总报告: {WORKSPACE / '模型训练总报告.md'}")
    print(f"纯文本总报告: {WORKSPACE / '模型训练总报告.txt'}")
    del model, train, test, train_features, test_features, train_all
    gc.collect()


if __name__ == "__main__":
    main()
