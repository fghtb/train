# -*- coding: utf-8 -*-
"""从 runs/*/summary.json 重建同一份累计训练报告。"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any


def _fmt(value: Any, digits: int = 6) -> str:
    if value is None:
        return "-"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(number):
        return "-"
    return f"{number:.{digits}f}"


def _load_summaries(workspace: Path) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for path in (workspace / "runs").glob("*/summary.json"):
        if path.parent.name.startswith("research_1_"):
            continue
        try:
            with path.open("r", encoding="utf-8") as f:
                item = json.load(f)
            item["_summary_path"] = str(path)
            summaries.append(item)
        except Exception as exc:
            print(f"跳过无法读取的摘要 {path}: {exc}")
    summaries.sort(key=lambda x: (x.get("created_at", ""), x.get("run_id", "")))
    return summaries


def _evaluation_group(summary):
    cv = summary.get("cross_validation", {})
    return cv.get("protocol_fingerprint", "legacy_fixed_split_v1")


def _generation(summary, fallback):
    match = re.match(r"^model_(\d+)", summary.get("model_name", ""))
    return match.group(1) if match else str(fallback)


_CHINESE_MODEL_NAMES = {
    "model_009_overnight_pressure_financial_v1": "隔夜信息与日内价格压力",
    "model_010_impact_liquidity_financial_v1": "真实波幅与流动性价格冲击",
    "model_011_downside_drawdown_financial_v1": "下行风险与回撤状态",
    "model_012_market_relative_financial_v1": "市场状态与个股相对收益",
    "model_013_intraday_acceptance_financial_v1": "日内价格接受与交易参与",
    "model_014_price_discovery_efficiency_financial_v1": "价格发现效率与买卖拒绝",
    "model_015_information_liquidity_financial_v1": "信息冲击与流动性承接",
    "model_016_overnight_market_state_financial_v1": "隔夜信息与市场状态",
    "model_017_downside_liquidity_stress_financial_v1": "下行风险与流动性压力",
    "model_018_market_liquidity_state_financial_v1": "市场状态与流动性冲击",
    "model_019_conditional_information_state_financial_v1": "条件信息与市场状态",
    "model_020_stress_transmission_financial_v1": "压力状态传导",
    "model_021_stress_core_financial_v1": "压力传导核心",
    "model_022_stress_recovery_financial_v1": "压力恢复与反转",
    "model_023_constraint_stress_financial_v1": "制度约束与压力传导",
    "model_024_new_mechanism_financial_v1": "成交参与加权价格压力",
    "model_025_new_mechanism_financial_v1": "市场扩散与分化转移",
    "model_026_new_mechanism_financial_v1": "系统性暴露与个股特有风险",
    "model_027_new_mechanism_financial_v1": "方向性流动性冲击不对称",
    "model_028_new_mechanism_financial_v1": "交易注意力集中与扩散",
    "model_029_high_score_synthesis_financial_v1": "压力状态与方向性流动性冲击",
    "model_030_high_score_synthesis_financial_v1": "风险调整后的方向性流动性冲击",
}


def _display_model_name(summary):
    name = summary.get("model_name", "-")
    return _CHINESE_MODEL_NAMES.get(name, name)


def _icir(metrics):
    if "icir" in metrics:
        return metrics["icir"]
    mean, std = metrics.get("ic_mean"), metrics.get("ic_std")
    if isinstance(mean, (int, float)) and isinstance(std, (int, float)) and std > 0:
        return mean / std
    return None


def _conclusion_lines(summaries):
    result = ["## 当前评估结论", "",
              "保留官方综合评分。旧代固定切分结果与新代扩展窗口结果分组展示，不用不同协议的分数直接宣称进步。",
              "2024 年已参与旧代模型选择，因此新代的 2024 年结果仍称最终诊断，不称全新盲测；新代不使用该年标签早停。", ""]
    groups = {}
    for summary in summaries:
        groups.setdefault(_evaluation_group(summary), []).append(summary)
    for group, candidates in groups.items():
        legacy = group == "legacy_fixed_split_v1"
        def score(s):
            if legacy:
                return s.get("metrics", {}).get("final_score")
            stats = s.get("cross_validation", {}).get("aggregate", {}).get("final_score", {})
            return stats.get("mean") if stats.get("valid_folds") == 3 else None
        usable = [(s, score(s)) for s in candidates]
        usable = [(s, v) for s, v in usable if isinstance(v, (int, float)) and math.isfinite(v)]
        if not usable:
            result.append("此评估组暂无可比较的完整指标。")
            continue
        best, value = max(usable, key=lambda pair: pair[1])
        kind = "旧固定切分组的 2024 综合得分" if legacy else "同一滚动协议组三折平均综合得分"
        result.append(f"{kind}最高的已完成模型：{_display_model_name(best)}，数值 {_fmt(value)}。")
        if not legacy and len(usable) == 1:
            result.append("本组当前只有一个已完成候选，尚不能据此宣称优于旧方案。")
    result.extend(["后续比较同时检查最差折、分期 ICIR 和换手率；三折均值不是未偏的最终比赛成绩。", ""])
    for group, candidates in groups.items():
        if group == "legacy_fixed_split_v1":
            continue
        result += ["### 同协议滚动验证横向比较", "",
                   "| 模型 | 特征数 | 三折平均Rank IC | 三折平均ICIR | 三折平均换手率 | 三折平均综合得分 | 最差折得分 | 2024诊断Rank IC |",
                   "|---|---:|---:|---:|---:|---:|---:|---:|"]
        for summary in sorted(candidates, key=lambda item: item.get("model_name", "")):
            stats = summary["cross_validation"]["aggregate"]
            result.append(
                f"| {_display_model_name(summary)} | {summary['feature_count']} | "
                f"{_fmt(stats['ic_mean'].get('mean'))} | {_fmt(stats['icir'].get('mean'))} | "
                f"{_fmt(stats['mean_turnover'].get('mean'))} | {_fmt(stats['final_score'].get('mean'))} | "
                f"{_fmt(stats['final_score'].get('min'))} | {_fmt(summary['metrics'].get('ic_mean'))} |")
        result += ["", "以上各模型按相同评估协议比较。004金融逻辑修订版仅含涨跌停两个标记，须结合预测并列诊断解释其组合得分，不能仅凭低换手断言其选股能力最强。", ""]
    return result


def _protocol_lines(workspace, summaries):
    completed = {s.get("model_name") for s in summaries}
    result = ["## 新增历史滚动验证协议", "",
              "新方案在各折训练历史中单独划出早停期；选定轮数后创建全新模型，用该折全部合规训练历史重训，再评价下一年。",
              "训练标签必须在下一阶段开始前兑现；评估期末无法在本期内兑现的标签不用于指标，相关剔除数量按折记录。",
              "最终模型在 2018—2023 年合规历史上从零训练，轮数取三折内层最佳轮数的中位数，2024 年不参与早停。测试 X 在历史评估完成后读取，测试 Y 不读取。",
              "每个滚动验证模型包括三次内层选轮、三次按所选轮数重训和一次最终训练，共七次拟合。", ""]
    result.append("")
    return result


def _cv_lines(summary):
    cv = summary.get("cross_validation")
    if not cv:
        return []
    result = ["", "**历史滚动验证结果**", "",
              f"评估协议：{cv['protocol_id']}；实际独立拟合次数：{cv['actual_fits']}。",
              "各折评估期不参与该折早停，最终轮数由历史内层早停结果确定。", "",
              "| 折 | 评估开始 | 评估结束 | 最佳轮数 | Rank IC | ICIR | 换手率 | 综合得分 |",
              "|---|---|---|---:|---:|---:|---:|---:|"]
    for fold in cv["folds"]:
        m, s = fold["metrics"], fold["split"]
        result.append(f"| {fold['name']} | {s['assessment_start']} | {s['assessment_end']} | {fold['best_iteration']} | {_fmt(m.get('ic_mean'))} | {_fmt(m.get('icir'))} | {_fmt(m.get('mean_turnover'))} | {_fmt(m.get('final_score'))} |")
    aggregate = cv["aggregate"]["final_score"]
    result += ["", f"三折综合得分均值：{_fmt(aggregate.get('mean'))}；最差折：{_fmt(aggregate.get('min'))}；折间标准差：{_fmt(aggregate.get('std'))}。",
               f"内层最佳轮数：{cv['inner_best_iterations']}；最终采用轮数：{cv['final_best_iteration']}。",
               f"最终训练边界剔除标签样本数：{cv['final_fit_purged_rows']}；最终比较期边界标签剔除数：{cv['comparison_purged_labels']}。",
               f"完整每折参数、边界剔除数量和期间统计：{summary.get('artifacts', {}).get('cross_validation', '-')}。", ""]
    diagnostic = cv.get("aggregate", {})
    ratio = diagnostic.get("top10_distinct_prediction_ratio_mean")
    ambiguous = diagnostic.get("top10_boundary_ambiguous_days")
    if ratio is not None or ambiguous is not None:
        result += [f"并列诊断：Top10%内不同预测值比例均值 {_fmt(ratio)}，最小值 {_fmt(diagnostic.get('top10_distinct_prediction_ratio_min'))}；边界并列天数 {ambiguous if ambiguous is not None else '-'}，边界并列比例 {_fmt(diagnostic.get('top10_boundary_ambiguous_ratio'))}。该诊断不改变官方换手率或综合得分。", ""]
    return result


def _period_lines(summary):
    periods = summary.get("period_metrics", {})
    result = []
    for key, title in [("monthly", "最终比较期月度结果"), ("quarterly", "最终比较期季度结果")]:
        if not periods.get(key):
            continue
        result += ["", f"**{title}**", "", "| 期间 | Rank IC | ICIR | 每日IC为正比例 | 换手率 | 综合得分 |", "|---|---:|---:|---:|---:|---:|"]
        for item in periods[key]:
            result.append(f"| {item['period']} | {_fmt(item.get('ic_mean'))} | {_fmt(item.get('icir'))} | {_fmt(item.get('ic_positive_ratio'))} | {_fmt(item.get('mean_turnover'))} | {_fmt(item.get('final_score'))} |")
    if result:
        result += ["", "分月分季换手率保留本评估序列内跨月、跨季的上一交易日持仓衔接。", ""]
    return result


def _optimization_record(summary: dict[str, Any], index: int) -> dict[str, Any]:
    """Return a readable optimization record, including legacy runs."""
    explicit = summary.get("optimization_plan")
    if explicit:
        if isinstance(explicit, str):
            return {"plan": explicit, "change": "配置中记录的本代实验", "next": "结合验证集指标决定下一代实验。"}
        return explicit

    mode = summary.get("feature_mode")
    name = summary.get("model_name", "")
    if index == 1 or mode == "raw" or "baseline" in name:
        return {
            "plan": "建立可复现的原始字段基线，固定时间切分、早停和参数，作为后续各代的唯一对照。",
            "change": "只使用题目提供的原始 OHLCV、涨跌停标记等字段，不加入手工技术特征。",
            "next": "在保持模型参数不变的情况下增加少量、可解释的历史技术特征。",
        }
    if mode == "technical":
        return {
            "plan": "检验量价、波动率、均线偏离、成交活跃度和横截面排名等技术特征是否能提高预测能力。",
            "change": "相对上一代只改变特征集合，LightGBM 参数保持一致，以隔离特征工程的影响。",
            "next": "比较信号增益和换手成本；若换手率升高，优先增加正则化并降低树复杂度。",
        }
    if mode == "technical_momentum_skip":
        return {
            "plan": "参考 Jegadeesh-Titman 的中期相对强度研究，测试 120/240 日动量及跳过最近 5 个交易日的动量。",
            "change": "在第二代技术特征上增加中期动量和横截面排名，LightGBM 参数保持一致。",
            "next": "本代预测能力下降，不继续堆叠动量特征；转向技术特征模型的树复杂度和换手率控制。",
        }
    return {
        "plan": "按单变量对照原则验证本代改动。",
        "change": "以配置和特征模式中记录的改动为准。",
        "next": "结合验证集综合得分、Rank IC、超额收益和换手率决定下一代。",
    }


def _research_references(summary: dict[str, Any], index: int) -> list[dict[str, Any]]:
    refs = summary.get("research_references")
    if isinstance(refs, list):
        return refs
    if index == 3 or summary.get("feature_mode") == "technical_momentum_skip":
        return [
            {
                "title": "Empirical Asset Pricing via Machine Learning",
                "path": r"D:\competition\OP-REVF200009.pdf",
                "use": "动量、流动性、波动率、时间顺序验证和换手率比较",
            },
            {
                "title": "Returns to Buying Winners and Selling Losers",
                "path": r"D:\competition\Test _Title_.pdf",
                "use": "中期动量与跳过最近交易日的特征假设",
            },
        ]
    return []


def _clean_text_inline(text: str) -> str:
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
    text = text.replace("`", "")
    return re.sub(r"\s+", " ", text).strip()


def _plain_table_row(line: str) -> list[str]:
    return [_clean_text_inline(part) for part in line.strip().strip("|").split("|")]


def _write_plain_text_report(source: Path, destination: Path) -> None:
    """Write a phone-friendly UTF-8 text copy without Markdown syntax."""
    lines = source.read_text(encoding="utf-8").splitlines()
    output: list[str] = []
    paragraph: list[str] = []
    in_code = False

    def flush_paragraph() -> None:
        if paragraph:
            value = _clean_text_inline(" ".join(paragraph))
            if value:
                output.append(value)
            paragraph.clear()

    i = 0
    while i < len(lines):
        raw = lines[i]
        stripped = raw.strip()
        if not stripped:
            flush_paragraph()
            i += 1
            continue
        if stripped.startswith("```"):
            flush_paragraph()
            in_code = not in_code
            i += 1
            continue
        if in_code:
            value = stripped.strip(",")
            if value not in ("{", "}"):
                match = re.fullmatch(r'"([^\"]+)"\s*:\s*(.*?),?', value)
                if match:
                    key, val = match.groups()
                    val = val.rstrip(",").strip().strip('"')
                    if val.lower() == "true":
                        val = "是"
                    elif val.lower() == "false":
                        val = "否"
                    output.append(f"{key}：{val}")
                elif value:
                    output.append(_clean_text_inline(value))
            i += 1
            continue
        if stripped.startswith("|"):
            flush_paragraph()
            table_lines = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                table_lines.append(lines[i].strip())
                i += 1
            rows = [_plain_table_row(line) for line in table_lines]
            rows = [
                row for row in rows
                if row and not all(re.fullmatch(r":?-{3,}:?", cell.replace(" ", "")) for cell in row)
            ]
            if rows:
                headers = rows[0]
                for row in rows[1:]:
                    fields = [
                        f"{headers[index]}：{value}"
                        for index, value in enumerate(row)
                        if index < len(headers) and value
                    ]
                    output.append("；".join(fields) if fields else "；".join(row))
            continue
        heading = re.match(r"^#{1,6}\s+(.*)$", stripped)
        if heading:
            flush_paragraph()
            output.append(_clean_text_inline(heading.group(1)))
            i += 1
            continue
        if stripped.startswith("> "):
            flush_paragraph()
            output.append(_clean_text_inline(stripped[2:]))
            i += 1
            continue
        if re.fullmatch(r"\*\*[^*]+\*\*", stripped):
            flush_paragraph()
            output.append(_clean_text_inline(stripped))
            i += 1
            continue
        bullet = re.match(r"^\s*[-*]\s+(.*)$", raw)
        numbered = re.match(r"^\s*(\d+)\.\s+(.*)$", raw)
        if bullet or numbered:
            flush_paragraph()
            output.append(
                _clean_text_inline(bullet.group(1))
                if bullet
                else f"{numbered.group(1)}、{_clean_text_inline(numbered.group(2))}"
            )
            i += 1
            continue
        paragraph.append(stripped)
        i += 1

    flush_paragraph()
    text = "\n".join(line for line in output if line.strip()) + "\n"
    if "```" in text or "**" in text or "`" in text:
        raise ValueError("纯文本报告仍包含 Markdown 标记")
    if any(line.startswith(("#", "- ", "|")) for line in text.splitlines()):
        raise ValueError("纯文本报告仍包含 Markdown 行标记")
    with destination.open("w", encoding="utf-8-sig", newline="\n") as handle:
        handle.write(text)


def _append_financial_whitepaper(workspace, summaries, lines):
    from financial_logic_reporting import whitepaper_lines
    financial_lines = whitepaper_lines(workspace, summaries)
    lines.extend(financial_lines)
    folder = workspace / "reports" / "financial_logic"
    folder.mkdir(parents=True, exist_ok=True)
    source = folder / "financial_logic_whitepaper.md"
    with source.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("\n".join(financial_lines) + "\n")
    _write_plain_text_report(source, workspace / "金融逻辑特征白皮书.txt")


def _portfolio_2024_lines(workspace, summaries):
    """Display 2024 portfolio diagnostics without changing training or saved runs."""
    from portfolio_attribution import portfolio_report_lines
    selected = []
    for summary in summaries:
        split = summary.get("split", {})
        start = int(str(split.get("valid_start", 0)).replace("-", ""))
        end = int(str(split.get("valid_end", 0)).replace("-", ""))
        if not (20240101 <= start <= end <= 20241231):
            continue
        display_summary = dict(summary)
        if summary.get("cross_validation"):
            display_summary["cross_validation"] = {
                **summary["cross_validation"], "folds": []
            }
        selected.append(display_summary)
    lines = portfolio_report_lines(workspace, selected)
    return [line.replace(
        "新滚动候选逐折报告，再单列2024诊断；2024曾用于旧代选模。",
        "本节仅展示2024年模拟组合业绩归因诊断及同口径比较。2024曾用于旧代选模。"
    ).replace("模拟组合业绩归因 final_diagnostic", "模拟组合业绩归因 2024诊断")
            for line in lines]


def rebuild_report(workspace: Path | str) -> Path:
    workspace = Path(workspace)
    summaries = _load_summaries(workspace)
    report = workspace / "模型训练总报告.md"
    lines: list[str] = [
        "# LightGBM 多模型训练总报告",
        "",
        "> 本文由 `reporting.py` 根据每一代运行目录中的 `summary.json` 自动重建。每个模型独立创建、独立保存，报告按训练完成时间排序。",
        "",
        "## 最终训练与比较区间",
        "",
        "| 数据段 | 日期范围 | 用途 |",
        "|---|---|---|",
        "| 训练集 | 2018-01-02 至 2023-12-31 | 拟合模型 |",
        "| 验证或诊断集 | 2024-01-01 至 2024-12-31 | 旧代早停比较；新代仅最终诊断 |",
        "| 测试集 | 2025-01-02 至 2026-06-08 | 生成提交预测 |",
        "",
        "## 模型横向对比",
        "",
    ]
    if not summaries:
        lines.append("当前还没有完成的模型运行。训练确认后，运行目录和本表会自动更新。")
        lines += _portfolio_2024_lines(workspace, summaries)
        _append_financial_whitepaper(workspace, summaries, lines)
        with report.open("w", encoding="utf-8", newline="\n") as f:
            f.write("\n".join(lines) + "\n")
        _write_plain_text_report(report, workspace / "模型训练总报告.txt")
        return report

    lines += _protocol_lines(workspace, summaries)
    lines += _conclusion_lines(summaries)
    batch_note = workspace / "reports" / "research_024_028" / "interpretation.md"
    if batch_note.exists():
        lines += ["", batch_note.read_text(encoding="utf-8"), ""]
    synthesis_note = workspace / "reports" / "research_029_030" / "interpretation.md"
    if synthesis_note.exists():
        lines += ["", synthesis_note.read_text(encoding="utf-8"), ""]

    lines.extend([
        "| 代次 | 模型 | 特征模式 | 特征数 | 主要Rank IC | 主要年化超额 | 主要换手率 | 主要综合得分 | 最佳轮数 |",
        "|---:|---|---|---:|---:|---:|---:|---:|---:|",
    ])
    for index, summary in enumerate(summaries, start=1):
        metrics = summary.get("metrics", {})
        if summary.get("cross_validation"):
            aggregate = summary["cross_validation"]["aggregate"]
            primary = {key: value.get("mean") for key, value in aggregate.items() if isinstance(value, dict)}
        else:
            primary = metrics
        lines.append(
            f"| {_generation(summary, index)} {_display_model_name(summary)} | {summary.get('feature_mode', '-')} | "
            f"{summary.get('feature_count', '-')} | {_fmt(primary.get('ic_mean'))} | {_fmt(primary.get('annual_excess'))} | {_fmt(primary.get('mean_turnover'))} | "
            f"{_fmt(primary.get('final_score'))} | {summary.get('best_iteration', '-')} |"
        )

    lines += ["", "## 各代模型详细记录", ""]
    for index, summary in enumerate(summaries, start=1):
        metrics = summary.get("metrics", {})
        previous = summaries[index - 2].get("metrics", {}) if index > 1 else None
        optimization = _optimization_record(summary, index)
        references = _research_references(summary, index)
        lines += [
            f"### 第 {_generation(summary, index)} 代：{_display_model_name(summary)}",
            "",
            f"- 运行时间：`{summary.get('created_at', '-')}`",
            f"- 运行目录：`{summary.get('run_dir', '-')}`",
            f"- 特征模式：`{summary.get('feature_mode', '-')}`，特征数：`{summary.get('feature_count', '-')}`",
            f"- 拟合样本：`{summary.get('rows', {}).get('fit', '-'):,}`；验证样本：`{summary.get('rows', {}).get('valid', '-'):,}`；测试样本：`{summary.get('rows', {}).get('test', '-'):,}`",
            f"- 训练标签缺失数：`{summary.get('missing_y_train', '-'):,}`",
            f"- 最佳迭代轮数：`{summary.get('best_iteration', '-')}`",
            "",
            "**训练方法**",
            "",
            ("使用三折历史扩展窗口；内层早停后每折从零重训。最终模型轮数取历史内层最佳轮数中位数，2024只用于诊断，不用于早停。" if summary.get("cross_validation") else "使用 LightGBM 回归模型，按固定日期切分训练和验证数据；训练过程中使用验证集早停，未加载任何前代模型。原始字段或技术特征由配置中的 feature_mode 决定。"),
            "",
            "**本代优化思路**",
            "",
            f"- 实验目的：{optimization.get('plan', '-')}",
            f"- 本代改动：{optimization.get('change', '-')}",
            f"- 下一步：{optimization.get('next', '-')}",
            "",
        ]
        if references:
            lines += [
                "**参考文献**",
                "",
            ]
            for reference in references:
                lines.append(
                    f"- `{reference.get('title', '-')}`：{reference.get('use', '-')}（`{reference.get('path', '-')}`）"
                )
            lines.append("")
        lines += [
            "",
            "**本代参数**",
            "",
            "```json",
            json.dumps(summary.get("model_params", {}), ensure_ascii=False, indent=2),
            "```",
            "",
            "**验证结果**",
            "",
            "| 指标 | 数值 |",
            "|---|---:|",
            f"| RMSE | {_fmt(metrics.get('rmse'))} |",
            f"| MAE | {_fmt(metrics.get('mae'))} |",
            f"| Rank IC 均值 | {_fmt(metrics.get('ic_mean'))} |",
            f"| ICIR 未年化 | {_fmt(_icir(metrics))} |",
            f"| 月均IC为正的月份比例 | {_fmt(metrics.get('monthly_ic_positive_ratio'))} |",
            f"| Rank IC > 0 占比 | {_fmt(metrics.get('ic_positive_ratio'))} |",
            f"| 年化 Top1 超额收益 | {_fmt(metrics.get('annual_excess'))} |",
            f"| 平均预测换手率 | {_fmt(metrics.get('mean_turnover'))} |",
            f"| 稳定性得分 1 - 换手率 | {_fmt(metrics.get('stability_score'))} |",
            f"| 综合得分 | {_fmt(metrics.get('final_score'))} |",
            "",
        ]
        lines += _cv_lines(summary)
        lines += _period_lines(summary)
        if previous is None:
            lines.append("**与前代区别**：这是当前报告中的首代模型，没有前代基线。")
        elif _evaluation_group(summary) != _evaluation_group(summaries[index - 2]):
            lines.append("**与前代区别**：评估或标签边界口径不同，不计算跨协议的改善幅度；应在同协议候选之间比较。")
        else:
            lines += [
                f"**与上一完成模型 {_display_model_name(summaries[index - 2])} 的区别和变化**",
                "",
                "| 指标 | 相对上一完成模型变化 |",
                "|---|---:|",
            ]
            for key, label in [
                ("ic_mean", "Rank IC 均值"),
                ("annual_excess", "年化超额收益"),
                ("mean_turnover", "平均预测换手率"),
                ("final_score", "综合得分"),
            ]:
                current = metrics.get(key)
                old = previous.get(key)
                if current is None or old is None:
                    delta = "-"
                else:
                    try:
                        delta = _fmt(float(current) - float(old))
                    except (TypeError, ValueError):
                        delta = "-"
                lines.append(f"| {label} | {delta} |")
            lines += [
                "",
                "变化解释：综合得分、Rank IC 和年化超额收益为正表示提高；平均换手率为负表示降低了换手。最终是否保留该模型，应结合三个比赛指标和跨时间稳定性判断。",
            ]
        lines += [
            "",
            "**本代产物**",
            "",
            f"- 模型：`{summary.get('artifacts', {}).get('model', '-')}`",
            f"- 验证预测：`{summary.get('artifacts', {}).get('validation_predictions', '-')}`",
            f"- 提交预测：`{summary.get('artifacts', {}).get('submission', '-')}`",
            f"- 特征重要性：`{summary.get('artifacts', {}).get('feature_importance', '-')}`",
            f"- 数据清单：`{summary.get('artifacts', {}).get('data_manifest', '-')}`",
            "",
        ]

    lines += _portfolio_2024_lines(workspace, summaries)
    _append_financial_whitepaper(workspace, summaries, lines)
    diagnostic = workspace / "reports" / "training_comparison_004_005_006" / "comparison_addendum.txt"
    if diagnostic.exists():
        diagnostic_lines = diagnostic.read_text(encoding="utf-8-sig").splitlines()
        reader_lines = [
            line for line in diagnostic_lines
            if not line.startswith(("第一次006尝试", "内存修复", "核验详情：", "金融逻辑白皮书、逐折"))
        ]
        if reader_lines:
            reader_lines[0] = "004至008代相关候选均已完成训练；以下并列说明同协议结果和重要性诊断。"
        lines += ["", "## 第004至008代滚动验证结果与并列预测诊断", "",
                  "\n".join(reader_lines), ""]
    with report.open("w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines) + "\n")
    _write_plain_text_report(report, workspace / "模型训练总报告.txt")
    from word_report import build_word_report
    build_word_report(workspace)
    return report


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", default=Path(__file__).parent.absolute())
    args = parser.parse_args()
    print(rebuild_report(args.workspace))
