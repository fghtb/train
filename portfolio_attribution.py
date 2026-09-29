"""Post-training portfolio diagnostics using only saved predictions and supplied X.

No training, model loading, downloads, test labels, or modification of old runs.
Positions are determined before any future return is examined.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

PROVIDED_TRAIN = Path(r"D:\competition\赛题五数据\训练集.csv")
VERSION = "top10_long_attribution_v1"
DEFAULTS = {
    "protocol_id": "TOP10_LONG_LABEL_DIAGNOSTIC_V1",
    "top_fraction": 0.1,
    "annual_trading_days": 252,
    "risk_free_annual_assumption": 0.02,
    "one_way_cost_assumption": 0.0003,
    "min_universe": 100,
    "min_metric_observations": 20,
}


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [clean(v) for v in value]
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path, value):
    with Path(path).open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(clean(value), handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def settings(workspace):
    result = dict(DEFAULTS)
    path = Path(workspace) / "configs" / "portfolio_report.json"
    if path.exists():
        result.update(json.loads(path.read_text(encoding="utf-8")))
    for name in ["top_fraction", "annual_trading_days", "risk_free_annual_assumption",
                 "one_way_cost_assumption", "min_universe", "min_metric_observations"]:
        if isinstance(result[name], bool) or not isinstance(result[name], (int, float)) or not math.isfinite(result[name]):
            raise ValueError(f"{name} must be a finite number")
    for name in ["annual_trading_days", "min_universe", "min_metric_observations"]:
        if int(result[name]) != result[name]:
            raise ValueError(f"{name} must be an integer")
    fixed = {
        "position": "long_only_equal_weight",
        "benchmark": "provided_current_eligible_universe_equal_weight_gross",
        "eligibility": "flag_limit_up=0 and current open>0 and close>0 and vol>0",
        "missing_return_policy": "select_before_labels_then_require_complete_top_and_benchmark_for_paired_metrics",
        "cost_method": "one_way_cost_times_L1_target_weight_change_including_initial_purchase",
        "terminal_liquidation": False,
        "execution_interpretation": "provided_next_day_label_diagnostic_not_executable_close_fill_backtest",
    }
    if set(result) - (set(DEFAULTS) | set(fixed) | {"notes"}):
        raise ValueError("Unknown portfolio reporting setting")
    for name, expected in fixed.items():
        if name in result and result[name] != expected:
            raise ValueError(f"Unsupported portfolio methodology: {name}")
    if not 0 < result["top_fraction"] <= 1:
        raise ValueError("top_fraction must be in (0, 1]")
    if result["annual_trading_days"] <= 0 or result["risk_free_annual_assumption"] <= -1:
        raise ValueError("Invalid annualization assumptions")
    if not 0 <= result["one_way_cost_assumption"] < 0.5:
        raise ValueError("Invalid cost assumption")
    if result["min_universe"] < 1 or result["min_metric_observations"] < 2:
        raise ValueError("Invalid observation threshold")
    return result


def load_market_context(specifications):
    """Read only contemporaneous X fields from the one whitelisted training CSV."""
    starts = [item["start"] for item in specifications]
    ends = [item["end"] for item in specifications]
    pieces, dates = [], set()
    columns = ["ts_code", "trade_date", "open", "close", "vol"]
    for chunk in pd.read_csv(PROVIDED_TRAIN, usecols=columns,
                             dtype={"ts_code": str, "trade_date": "int32"},
                             chunksize=500_000):
        dates.update(int(x) for x in chunk.trade_date.unique())
        selected = chunk.trade_date.between(min(starts), max(ends))
        if selected.any():
            pieces.append(chunk.loc[selected].copy())
    if not pieces:
        raise ValueError("Provided training X has no matching assessment dates")
    market = pd.concat(pieces, ignore_index=True)
    if market[["ts_code", "trade_date"]].duplicated().any():
        raise ValueError("Duplicate market keys")
    calendar = sorted(dates)
    next_dates = dict(zip(calendar[:-1], calendar[1:]))
    return market, next_dates


def daily_portfolio(predictions, market, next_dates, start, end, config):
    """Select holdings without labels; value only fully observed paired days."""
    required = ["ts_code", "trade_date", "flag_limit_up", "pred", "y_ret_1d"]
    if not set(required).issubset(predictions.columns):
        raise ValueError("Saved historical predictions lack required columns")
    frame = predictions[required].copy()
    if frame[["ts_code", "trade_date"]].isna().any().any():
        raise ValueError("Missing prediction keys")
    if frame[["ts_code", "trade_date"]].duplicated().any():
        raise ValueError("Duplicate prediction keys")
    if not frame.trade_date.between(start, end).all():
        raise ValueError("Predictions outside the stated assessment interval")
    if not np.isfinite(frame["pred"]).all():
        raise ValueError("Non-finite saved predictions")
    if not frame.flag_limit_up.isin([0, 1]).all():
        raise ValueError("Unknown limit-up flag")
    frame["ts_code"] = frame.ts_code.astype(str)
    context = market[market.trade_date.between(start, end)].copy()
    context["ts_code"] = context.ts_code.astype(str)
    expected_keys = pd.MultiIndex.from_frame(context[["ts_code", "trade_date"]])
    actual_keys = pd.MultiIndex.from_frame(frame[["ts_code", "trade_date"]])
    if len(expected_keys) != len(actual_keys) or not expected_keys.difference(actual_keys).empty:
        raise ValueError("Historical predictions do not cover the full provided assessment universe")
    frame = frame.merge(context, on=["ts_code", "trade_date"], how="left",
                        validate="one_to_one", indicator=True)
    if not frame["_merge"].eq("both").all():
        raise ValueError("Missing contemporaneous X for historical predictions")
    daily, positions, previous = [], [], {}
    for signal_date, group in frame.groupby("trade_date", sort=True):
        # Do not add a label or label-availability filter here.
        mask = (group.flag_limit_up.eq(0) & group.open.gt(0) &
                group.close.gt(0) & group.vol.gt(0) &
                np.isfinite(group[["open", "close", "vol"]]).all(axis=1))
        eligible = group.loc[mask].sort_values(
            ["pred", "ts_code"], ascending=[False, True], kind="mergesort")
        usable = len(eligible) >= config["min_universe"]
        count = max(int(len(eligible) * config["top_fraction"]), 1) if usable else 0
        top = eligible.head(count)
        target = {code: 1.0 / count for code in top.ts_code} if count else {}
        l1 = sum(abs(target.get(code, 0.0) - previous.get(code, 0.0))
                 for code in target.keys() | previous.keys())
        cost = config["one_way_cost_assumption"] * l1
        previous = target
        positions.extend({"trade_date": int(signal_date), "ts_code": code,
                          "target_weight": weight} for code, weight in target.items())
        realization = next_dates.get(int(signal_date))
        within = realization is not None and int(realization) <= end
        top_finite = np.isfinite(top.y_ret_1d)
        bench_finite = np.isfinite(eligible.y_ret_1d)
        top_coverage = float(top_finite.mean()) if count else None
        bench_coverage = float(bench_finite.mean()) if len(eligible) else None
        complete_top = usable and bool(top_finite.all())
        complete_benchmark = usable and bool(bench_finite.all())
        paired = bool(within and complete_top and complete_benchmark)
        gross = float(top.y_ret_1d.mean()) if within and complete_top else np.nan
        benchmark = float(eligible.y_ret_1d.mean()) if within and complete_benchmark else np.nan
        net = gross - cost if math.isfinite(gross) else np.nan
        daily.append({
            "trade_date": int(signal_date), "realization_date": realization,
            "eligible_stocks": len(eligible), "top_stocks": count,
            "top_label_coverage": top_coverage, "benchmark_label_coverage": bench_coverage,
            "top_missing_labels": int((~top_finite).sum()),
            "benchmark_missing_labels": int((~bench_finite).sum()),
            "label_within_period": bool(within), "paired_valid": paired,
            "l1_target_weight_change": l1, "half_l1_turnover": 0.5 * l1,
            "cost_proxy": cost, "gross_return": gross, "net_return": net,
            "benchmark_gross_return": benchmark,
        })
    return pd.DataFrame(daily), pd.DataFrame(
        positions, columns=["trade_date", "ts_code", "target_weight"])


def performance(returns, benchmark, config, complete_interval=False):
    """Arithmetic annual diagnostics; no fabricated NAV across unknown returns."""
    r, b = np.asarray(returns, dtype=float), np.asarray(benchmark, dtype=float)
    if len(r) != len(b) or not np.isfinite(r).all() or not np.isfinite(b).all():
        raise ValueError("Performance inputs must be finite, aligned paired observations")
    result = {key: None for key in [
        "annual_arithmetic_return", "annual_arithmetic_active_return",
        "annual_volatility", "sharpe", "information_ratio", "tracking_error",
        "beta", "alpha_annual_arithmetic", "beta_contribution_annual",
        "risk_free_contribution_annual", "regression_r_squared",
        "cumulative_return", "annual_geometric_return", "maximum_drawdown",
    ]}
    result["observations"] = len(r)
    if not len(r):
        return result
    annual = config["annual_trading_days"]
    rf = (1 + config["risk_free_annual_assumption"]) ** (1 / annual) - 1
    result.update(annual_arithmetic_return=float(r.mean() * annual),
                  annual_arithmetic_active_return=float((r - b).mean() * annual),
                  risk_free_contribution_annual=rf * annual)
    if len(r) >= config["min_metric_observations"]:
        sd, active_sd = float(np.std(r - rf, ddof=1)), float(np.std(r - b, ddof=1))
        result["annual_volatility"] = sd * math.sqrt(annual)
        result["tracking_error"] = active_sd * math.sqrt(annual)
        if sd > 1e-12:
            result["sharpe"] = float(np.mean(r - rf) / sd * math.sqrt(annual))
        if active_sd > 1e-12:
            result["information_ratio"] = float(np.mean(r - b) / active_sd * math.sqrt(annual))
        x, y = b - rf, r - rf
        dx, dy = x - x.mean(), y - y.mean()
        denominator = float(np.dot(dx, dx))
        if denominator > 1e-20:
            beta = float(np.dot(dx, dy) / denominator)
            alpha = float(y.mean() - beta * x.mean())
            residual = y - (alpha + beta * x)
            result.update(beta=beta, alpha_annual_arithmetic=alpha * annual,
                          beta_contribution_annual=float(beta * x.mean() * annual))
            y_squares = float(np.dot(dy, dy))
            if y_squares > 1e-20:
                result["regression_r_squared"] = float(
                    1 - np.dot(residual, residual) / y_squares)
    if complete_interval and np.all(r > -1):
        nav = np.concatenate([[1.0], np.cumprod(1 + r)])
        result["cumulative_return"] = float(nav[-1] - 1)
        result["annual_geometric_return"] = float(nav[-1] ** (annual / len(r)) - 1)
        result["maximum_drawdown"] = float((nav / np.maximum.accumulate(nav) - 1).min())
    return clean(result)


def summarize_daily(daily, config):
    valid = daily.loc[daily.paired_valid].copy()
    bounded = daily.loc[daily.label_within_period]
    complete = len(bounded) > 0 and len(valid) == len(bounded)
    gross = performance(valid.gross_return, valid.benchmark_gross_return, config, complete)
    net = performance(valid.net_return, valid.benchmark_gross_return, config, complete)
    benchmark = performance(valid.benchmark_gross_return, valid.benchmark_gross_return, config, complete)
    coverage = {
        "signal_days": len(daily), "boundary_excluded_days": len(daily) - len(bounded),
        "within_period_days": len(bounded), "paired_days": len(valid),
        "paired_fraction": len(valid) / len(bounded) if len(bounded) else None,
        "missing_or_ineligible_days": len(bounded) - len(valid),
        "complete_interval": complete,
        "top_complete_days": int((bounded.top_label_coverage.eq(1) & bounded.top_stocks.gt(0)).sum()),
        "benchmark_complete_days": int(bounded.benchmark_label_coverage.eq(1).sum()),
        "mean_top_label_coverage": bounded.top_label_coverage.mean(),
        "mean_benchmark_label_coverage": bounded.benchmark_label_coverage.mean(),
    }
    periods = []
    if len(valid):
        dates = pd.to_datetime(valid.realization_date.astype("int64").astype(str), format="%Y%m%d")
        for frequency in ["M", "Q"]:
            for period, group in valid.groupby(dates.dt.to_period(frequency)):
                stats = performance(group.net_return, group.benchmark_gross_return, config, False)
                periods.append({
                    "frequency": frequency, "period": str(period), "paired_days": len(group),
                    "mean_daily_net_return": float(group.net_return.mean()),
                    "mean_daily_active_return": float((group.net_return - group.benchmark_gross_return).mean()),
                    "information_ratio": stats["information_ratio"],
                    "mean_daily_cost_proxy": float(group.cost_proxy.mean()),
                })
    return clean({
        "coverage": coverage, "gross": gross, "net": net, "benchmark": benchmark,
        "mean_half_l1_turnover_all_bounded_days": bounded.half_l1_turnover.mean(),
        "annual_cost_drag_paired_days": valid.cost_proxy.mean() * config["annual_trading_days"],
        "annual_cost_proxy_all_bounded_days": bounded.cost_proxy.mean() * config["annual_trading_days"],
        "periods": periods,
    })


def specifications(workspace, summaries):
    result = []
    for summary in summaries:
        run_dir = Path(summary["_summary_path"]).resolve().parent
        if run_dir.parent != (Path(workspace) / "runs").resolve():
            raise ValueError("Summary is outside the workspace run directory")
        group = summary.get("cross_validation", {}).get("protocol_fingerprint", "legacy_fixed_split_v1")
        common = {"model_name": summary["model_name"], "run": run_dir.name,
                  "feature_mode": summary.get("feature_mode"), "evaluation_group": group,
                  "legacy_training_caveat": not bool(summary.get("cross_validation"))}
        split = summary["split"]
        result.append({**common, "stage": "final_diagnostic",
                       "start": split["valid_start"], "end": split["valid_end"],
                       "path": run_dir / "validation_predictions.csv"})
        for fold in summary.get("cross_validation", {}).get("folds", []):
            if Path(fold["name"]).name != fold["name"]:
                raise ValueError("Unsafe fold name")
            result.append({**common, "stage": fold["name"],
                           "start": fold["split"]["assessment_start"],
                           "end": fold["split"]["assessment_end"],
                           "path": run_dir / "folds" / fold["name"] / "validation_predictions.csv"})
        if any(s["end"] >= split["test_start"] for s in result if s["run"] == run_dir.name):
            raise ValueError("Portfolio diagnostics must not enter the competition test period")
    return result


def ensure_artifacts(workspace, summaries, config):
    """Content-addressed additive reports; original runs are never written."""
    specs = specifications(workspace, summaries)
    if not specs:
        return []
    source_stat = PROVIDED_TRAIN.stat()
    source = {"path": str(PROVIDED_TRAIN.resolve()), "bytes": source_stat.st_size,
              "mtime_ns": source_stat.st_mtime_ns, "sha256": file_hash(PROVIDED_TRAIN)}
    code_sha = file_hash(__file__)
    output = Path(workspace) / "reports" / "portfolio"
    pending, records = [], []
    for spec in specs:
        if not spec["path"].exists():
            records.append({**spec, "status": "missing_predictions", "error": "尚无保存的该期历史预测"})
            continue
        identity = {**spec, "path": str(spec["path"]), "prediction_sha256": file_hash(spec["path"]),
                    "source": source, "config": config, "code_sha256": code_sha, "version": VERSION}
        key = hashlib.sha256(json.dumps(identity, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()
        folder = output / spec["run"] / spec["stage"] / key[:16]
        cached = folder / "attribution.json"
        if cached.exists():
            record = json.loads(cached.read_text(encoding="utf-8"))
            if record.get("identity") == identity and all(
                    (folder / name).exists() and file_hash(folder / name) == digest
                    for name, digest in record["output_hashes"].items()):
                records.append(record)
                continue
        pending.append((spec, identity, folder))
    if pending:
        market, next_dates = load_market_context([p[0] for p in pending])
        for spec, identity, folder in pending:
            predictions = pd.read_csv(spec["path"], dtype={"ts_code": str})
            daily, holdings = daily_portfolio(
                predictions, market, next_dates, spec["start"], spec["end"], config)
            folder.mkdir(parents=True, exist_ok=True)
            daily.to_csv(folder / "daily_portfolio.csv", index=False, encoding="utf-8-sig")
            holdings.to_csv(folder / "holdings.csv", index=False, encoding="utf-8-sig")
            record = {**{k: v for k, v in spec.items() if k != "path"},
                      "status": "complete", "identity": identity, "folder": str(folder),
                      **summarize_daily(daily, config),
                      "output_hashes": {name: file_hash(folder / name)
                                       for name in ["daily_portfolio.csv", "holdings.csv"]}}
            write_json(cached := folder / "attribution.json", record)
            records.append(clean(record))
    order = {(s["run"], s["stage"]): n for n, s in enumerate(specs)}
    records.sort(key=lambda r: order[(r["run"], r["stage"])])
    return records


def number(value, percent=False):
    if value is None or not isinstance(value, (int, float)) or not math.isfinite(value):
        return "无法计算"
    return f"{100 * value:.4f}%" if percent else f"{value:.6f}"


def comparison_lines(records, config):
    groups = {}
    for record in records:
        if record["status"] == "complete":
            key = (record["evaluation_group"], record["stage"], record["start"], record["end"])
            groups.setdefault(key, []).append(record)
    lines = ["", "### 相同口径下的成本与模型比较", ""]
    for key, group in groups.items():
        if len(group) < 2:
            lines.append(f"{key[1]}：同一评估协议目前只有一个完成候选，等待后续候选后比较。")
            continue
        frames = [pd.read_csv(Path(r["folder"]) / "daily_portfolio.csv").set_index("trade_date") for r in group]
        common = set.intersection(*(set(df.index[df.paired_valid]) for df in frames))
        comparison = []
        lines += [f"评估期 {key[2]}至{key[3]}；同一协议；共同有效配对日 {len(common)} 天。",
                  "", "| 模型 | 配对日平均目标权重换手率 | 毛IR | 净IR | 成本后算术年化 | 年化费用拖累 |",
                  "|---|---:|---:|---:|---:|---:|"]
        for record, frame in zip(group, frames):
            frame = frame.loc[sorted(common)]
            gross = performance(frame.gross_return, frame.benchmark_gross_return, config)
            net = performance(frame.net_return, frame.benchmark_gross_return, config)
            turnover = float(frame.half_l1_turnover.mean()) if len(frame) else None
            drag = float(frame.cost_proxy.mean() * config["annual_trading_days"]) if len(frame) else None
            comparison.append((record["model_name"], turnover, gross["information_ratio"], net["information_ratio"]))
            lines.append(f"| {record['model_name']} | {number(turnover, True)} | {number(gross['information_ratio'])} | {number(net['information_ratio'])} | {number(net['annual_arithmetic_return'], True)} | {number(drag, True)} |")
        usable = [r for r in comparison if r[1] is not None and r[3] is not None]
        if len(usable) >= 2:
            low, high = min(usable, key=lambda x: x[1]), max(usable, key=lambda x: x[1])
            relation = "低于" if high[3] < low[3] else ("高于" if high[3] > low[3] else "等于")
            lines += ["", f"在共同可观测日期上，换手最高的 {high[0]} 成本后IR为 {number(high[3])}，{relation}换手最低的 {low[0]} 的 {number(low[3])}。",
                      "这只是已完成模型的诊断对比，特征等因素也存在差异；不能把模型间全部差距归因于手续费。提高min_child_samples的控制变量实验仍须另行展示参数并确认训练。"]
        lines += [""]
    return lines


def portfolio_report_lines(workspace, summaries):
    config = settings(workspace)
    lines = ["", "## 模拟组合业绩归因报告", "",
             "依据用户提供的top组.docx补充。以下是已保存历史预测的标签诊断，不是实盘业绩，也不替代比赛官方综合评分。",
             f"组合为当日预测前 {config['top_fraction']:.0%} 等权多头；筛选只使用当日涨停标记、正开盘价、正收盘价和正成交量；预测并列时按代码排序。",
             "基准为提供样本中通过同样筛选的股票等权毛收益，不是沪深300或完整A股市场。采用纯多头方案，不假设可以借券做空。",
             f"固定情景假设：每年 {config['annual_trading_days']} 个交易日；无风险年利率 {config['risk_free_annual_assumption']:.2%}；单边成本 {config['one_way_cost_assumption']:.2%}。均不读取外部行情或利率。",
             "每日净收益=组合毛收益-单边费率×相邻目标权重差的绝对值之和。买卖双向计费，包含初始买入，不强制期末清仓；未模拟权重价格漂移、税费变化、冲击、滑点及无法卖出的情形。",
             "先确定持仓，再检查收益标签。组合与基准成分收益全部可观测、且兑现日不超出评估区间的日期，才进入配对指标。缺失不补零，交易成本记录在收益缺失日也继续衔接。",
             "缺失日会造成样本选择偏差；有效配对日算术年化收益不是全年实际收益。数据不完整时，完整区间复利收益、连续净值和最大回撤显示无法计算。",
             f"算术年化=日均收益×{config['annual_trading_days']}；波动率与跟踪误差使用样本标准差×√{config['annual_trading_days']}。夏普使用扣除固定日无风险收益的序列；IR使用组合减基准的序列。",
             f"至少 {config['min_metric_observations']} 个有效配对日才输出风险比率和回归；零分母返回无法计算。年化遵循常用日频缩放假设，未声称校正收益自相关。",
             "Alpha/Beta回归：组合日收益减无风险收益=截距+Beta×基准日超额收益+残差。年化截距、基准联动项和无风险项之和等于算术年化组合收益。IR不自动剔除Beta，回归截距也不能证明纯选股能力。",
             "新滚动候选逐折报告，再单列2024诊断；2024曾用于旧代选模。为旧代补充诊断不改变其原训练流程，也不能消除既有选择偏差。",
             ""]
    if not summaries:
        return lines + ["尚无完成模型。生成历史预测后，本节自动补充计算结果。", ""]
    records = ensure_artifacts(Path(workspace), summaries, config)
    for record in records:
        lines += [f"### {record['model_name']} 模拟组合业绩归因 {record['stage']}", ""]
        if record["status"] != "complete":
            lines += [f"状态：{record['error']}；不生成成绩。", ""]
            continue
        coverage = record["coverage"]
        lines += [f"信号日期范围：{record['start']}至{record['end']}；边界排除 {coverage['boundary_excluded_days']} 天；区间内应评价 {coverage['within_period_days']} 天；完整配对 {coverage['paired_days']} 天，覆盖率 {number(coverage['paired_fraction'], True)}。",
                  f"Top持仓平均标签覆盖率 {number(coverage['mean_top_label_coverage'], True)}；基准平均标签覆盖率 {number(coverage['mean_benchmark_label_coverage'], True)}。",
                  "状态：" + ("区间内配对收益完整，可计算诊断复利和回撤。" if coverage["complete_interval"]
                               else "收益序列不完整，以下为有效样本诊断，不输出完整区间复利及最大回撤数值。"),
                  "", "| 指标 | 成本前组合 | 成本后组合 |", "|---|---:|---:|"]
        for field, title, percent in [
            ("annual_arithmetic_return", "有效样本算术年化收益", True),
            ("annual_arithmetic_active_return", "有效样本算术年化超额", True),
            ("annual_volatility", "年化波动率", True), ("sharpe", "夏普比率", False),
            ("information_ratio", "信息比率IR", False), ("tracking_error", "年化跟踪误差", True),
            ("beta", "相对样本基准Beta", False),
            ("alpha_annual_arithmetic", "回归截距算术年化Alpha", True),
            ("beta_contribution_annual", "基准联动项算术年化贡献", True),
            ("risk_free_contribution_annual", "固定无风险项算术年化贡献", True),
            ("regression_r_squared", "回归解释度R平方", False),
            ("annual_geometric_return", "完整区间几何年化收益", True),
            ("maximum_drawdown", "完整区间最大回撤", True),
        ]:
            lines.append(f"| {title} | {number(record['gross'].get(field), percent)} | {number(record['net'].get(field), percent)} |")
        lines += ["", f"配对日基准算术年化收益：{number(record['benchmark']['annual_arithmetic_return'], True)}；配对日年化成本拖累：{number(record['annual_cost_drag_paired_days'], True)}。",
                  f"全部区间内日期平均目标权重换手率：{number(record['mean_half_l1_turnover_all_bounded_days'], True)}。此指标使用权重差，不能与官方持仓集合换手率混同。"]
        net = record["net"]
        if net["beta"] is not None:
            beta_part, alpha_part = net["beta_contribution_annual"], net["alpha_annual_arithmetic"]
            larger = "基准联动项" if abs(beta_part) > abs(alpha_part) else "回归截距项"
            lines += [f"收益来源诊断：成本后Beta为 {number(net['beta'])}；基准联动贡献 {number(beta_part, True)}；截距贡献 {number(alpha_part, True)}，两者中绝对值较大的是{larger}。",
                      "这里只描述已观测样本的单基准关联，未控制行业、规模等其他风险暴露，也未证明统计显著性；不能据此断言真实Alpha已经成立。"]
        lines += ["", "| 期间 | 有效配对日 | 日均净收益 | 日均超额 | 净IR | 日均成本 |",
                  "|---|---:|---:|---:|---:|---:|"]
        for period in record["periods"]:
            lines.append(f"| {period['period']} | {period['paired_days']} | {number(period['mean_daily_net_return'], True)} | {number(period['mean_daily_active_return'], True)} | {number(period['information_ratio'])} | {number(period['mean_daily_cost_proxy'], True)} |")
        lines += ["", f"逐日收益与缺失记录：{record['folder']}\\daily_portfolio.csv", "",
                  f"每日目标持仓：{record['folder']}\\holdings.csv", "",
                  f"完整指标、输入哈希和假设：{record['folder']}\\attribution.json", ""]
    lines += comparison_lines(records, config)
    return lines
