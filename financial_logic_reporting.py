"""Auditable financial whitepaper and plain-text pre-training review."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from financial_logic import (
    POLICY_VERSION, registry, policy_fingerprint, selection_details, validate_policy,
)


def _training_registry(summary):
    path = Path(summary.get("_summary_path", "")).parent / "financial_logic_snapshot.json"
    if path.is_file():
        snapshot = json.loads(path.read_text(encoding="utf-8"))
        if summary.get("model_name") == "model_028_new_mechanism_financial_v1":
            # Display-only erratum; preserve the original training snapshot on disk.
            for row in snapshot["retained"]:
                if row["feature"] == "amount_share_rank":
                    row.update({k: registry()["amount_share_rank"][k] for k in
                                ("formula", "logic", "family", "failure_risk", "expected_turnover")})
                    row["logic"] += "（报告说明已勘误；训练快照原文保留，本次补正不追认为训练前已正确展示。）"
        return {row["feature"]: row for row in snapshot["retained"]}, snapshot
    return registry(), None


def _importance(summary):
    path = Path(summary.get("_summary_path", "")).parent / "feature_importance.csv"
    if not path.is_file():
        return None, "未保存重要性明细，不能补造排名。"
    frame = pd.read_csv(path)
    kind = summary.get("model_params", {}).get("importance_type", "split")
    column = "importance_gain" if kind == "gain" else "importance_split"
    if not {"feature", column}.issubset(frame.columns):
        return None, "重要性文件缺少声明口径对应的列。"
    frame = frame.sort_values([column, "feature"], ascending=[False, True], kind="mergesort")
    return frame, kind


def whitepaper_lines(workspace, summaries):
    workspace = Path(workspace)
    catalog = registry()
    lines = ["", "## 金融逻辑特征白皮书", "",
             "本节说明所用特征的计算方法、金融假设、换手影响和失效风险。",
             "金融解释只是假设，不是已识别的因果关系。缺少注释不等于已经证明特征无经济含义；为旧特征事后补写解释，也不算原训练前的研究证据。",
             "收益预测和重要性不能证明因果；删特征后IC下降不能单独证明原模型依赖伪相关。放量不能识别资金身份或净流入，涨跌停也不能保证下一日收益方向。",
             "文档中关于样本仅含存续公司、具体ST/退市遗漏、财报滞后期限等表述尚未从本题数据说明核实，不作为已证实事实；现有OHLCV也不足以完整控制市值和退市风险。",
             "哑变量K-1和VIF等线性回归规则不能机械套用到树模型；有金融解释的特征仍可能冗余或过拟合，必须经过按时间隔离的验证。",
             ""]
    if summaries:
        latest = summaries[-1]
        current_registry, snapshot = _training_registry(latest)
        importance, kind = _importance(latest)
        lines += ["### 当前已完成模型的前10项重要特征", "",
                  f"最近完成的模型：{latest['model_name']}。"]
        if importance is None:
            lines += [kind, ""]
        else:
            label = "分裂次数split" if kind == "split" else "增益gain"
            lines += [f"按照该模型feature_importances_的口径排序：{label}。默认split不是收益贡献或因果贡献；相关特征间的重要性也可能分摊。",
                      "", "| 排名 | 特征 | split次数 | gain | 金融假设一句话 |",
                      "|---:|---|---:|---:|---|"]
            for rank, row in enumerate(importance.head(10).to_dict("records"), 1):
                entry = current_registry.get(row["feature"], catalog.get(row["feature"], {}))
                count = row.get("importance_split")
                gain = row.get("importance_gain")
                count_text = str(int(count)) if count is not None and pd.notna(count) else "未保存"
                gain_text = f"{gain:.6f}" if gain is not None and pd.notna(gain) else "未保存"
                status = "已在训练前登记并确认" if snapshot else (
                    "隔离，待重新审查" if entry.get("status") == "quarantined"
                    else "可进入下一候选的确认清单")
                lines.append(f"| {rank} | {row['feature']} | {count_text} | {gain_text} | {entry.get('logic', '金融假设待补充')} |")
        if latest["model_name"] == "model_003_momentum_skip":
            lines += ["", "原始特征代码中，Top10的ret_20_skip5有跳过近期动量的研究说明；其余九项没有明确的逐项金融机制说明，本次先隔离。上表给出的可能解释不追认旧模型的事前研究。",
                      "审查依据是保留的原始代码和模型重要性；没有发现文档举例的股票代码尾号或任意平方拼接已进入当前特征集。", ""]
    else:
        lines += ["尚无完成模型，暂不生成重要性排名。", ""]

    candidates = []
    completed_summaries = {s["model_name"]: s for s in summaries}
    for path in sorted((workspace / "configs").glob("model_*.json")):
        config = json.loads(path.read_text(encoding="utf-8"))
        if "financial_logic" not in config:
            continue
        try:
            completed_summary = completed_summaries.get(config["model_name"])
            if completed_summary is not None:
                _, details = _training_registry(completed_summary)
                details = details or selection_details(config)
            else:
                details = selection_details(config)
        except ValueError as exc:
            lines += [f"候选{config.get('model_name', path.stem)}审核失效：{exc}，禁止训练。", ""]
            continue
        candidates.append((path, config, details))
    completed = {s["model_name"] for s in summaries}
    for path, config, details in candidates:
        state = "已有完成记录，按对应运行目录查看成绩" if config["model_name"] in completed else "待明确确认，尚无训练结果"
        lines += [f"### 模型 {config['model_name']} 的金融逻辑", "",
                  f"状态：{state}。保留{details['feature_count']}项，剔除{len(details['excluded'])}项；已完成模型以当次训练快照为准。",
                  "",
                  "| 保留特征 | 计算定义 | 金融假设 | 预期换手影响 | 失效风险 |",
                  "|---|---|---|---|---|"]
        for row in details["retained"]:
            lines.append(f"| {row['feature']} | {row['formula']} | {row['logic']} | {row['expected_turnover']} | {row['failure_risk']} |")
        lines += ["", "### 本轮剔除清单与理由", "",
                  "| 特征 | 剔除原因 |", "|---|---|"]
        for row in details["excluded"]:
            lines.append(f"| {row['feature']} | {row['decision_reason']} |")
        lines += ["", "原始价格和成交量仍可作为合规派生变量的计算原料；中间列ret_1仍可用于历史波动率计算，但这些被隔离的列不传入模型拟合或预测。",
                  "本候选剔除范围以本节逐项清单为准；不同特征模式的原始目录不同。这是本轮保守筛选，不是证明这些变量永远无效。",
                  "", "### 对照方案与结果解读", "",
                  config.get("comparison_plan", {}).get("interpretation", "仅在相同时间协议和指标口径下作可比评价。"),
                  "本候选按本代已确认的LightGBM参数执行三折历史滚动验证加2024最终诊断；每候选7次全新拟合，测试期仅生成预测。",
                  ("本候选已完成训练，Rank IC、ICIR、换手率、综合得分和模拟组合成本后IR见总报告对应运行记录；指标上升或下降都如实保留，不能仅凭有金融解释就声称有效。"
                   if config["model_name"] in completed else
                   "完成后报告Rank IC、ICIR、换手率、综合得分和模拟组合成本后IR；当前尚无本候选完成结果，不能预先宣称本代改动有效。"),
                  ""]
    return lines


def write_confirmation_text(workspace, config):
    workspace = Path(workspace)
    details = selection_details(config)
    lines = [
        "第006代金融逻辑与训练参数待确认",
        f"模型：{config['model_name']}；状态：尚未开始真实训练。",
        f"金融逻辑优先：保留{details['feature_count']}项，剔除{len(details['excluded'])}项；不新增特征。",
        "金融解释是待检验假设，不能作为因果证明，也不能预先保证换手率下降。",
        f"审查版本：{details['version']}；登记指纹：{details['registry_fingerprint']}",
        "保留特征的逐项金融逻辑：",
    ]
    for n, row in enumerate(details["retained"], 1):
        lines += [f"{n}、{row['feature']}：{row['logic']}",
                  f"计算：{row['formula']}；可用时点：{row['available_at']}",
                  f"预期换手：{row['expected_turnover']}；失效风险：{row['failure_risk']}"]
    lines += ["剔除特征与原因："]
    lines += [f"{row['feature']}：{row['decision_reason']}" for row in details["excluded"]]
    lines += ["LightGBM参数（沿用003，未改动）："]
    lines += [f"{key}：{value}" for key, value in config["model_params"].items()]
    evaluation = config["evaluation"]
    lines += [f"早停耐心：{evaluation['early_stopping_rounds']}轮；拟合次数：{evaluation['max_model_fits']}次。",
              "时间划分："]
    for fold in evaluation["folds"]:
        lines.append(f"{fold['name']}：内层训练{fold['fit_start']}至{fold['inner_fit_end']}；早停{fold['early_stop_start']}至{fold['early_stop_end']}；全新重训截至{fold['refit_end']}；评估{fold['assessment_start']}至{fold['assessment_end']}。")
    split = config["split"]
    lines += [f"最终训练：{split['train_start']}至{split['train_end']}；轮数取三折内层最佳轮数中位数。",
              f"2024最终诊断：{split['valid_start']}至{split['valid_end']}，不参与早停，且不是未被查看过的盲测。",
              f"比赛预测：{split['test_start']}至{split['test_end']}，历史评估完成后才读取测试X；不读取测试Y。",
              "所有特征只用提供数据的当日与历史记录；训练标签继续按兑现日期隔离边界。",
              config["comparison_plan"]["interpretation"],
              "新增白皮书和组合归因会随本代训练报告一起更新；旧模型和预测不覆盖。",
              f"确认范围：上述全部金融逻辑、特征清单、参数、日期、选轮规则及本候选7次独立拟合。",
              f"确认口令：{config['confirmation_token']}；其他口令或继续二字不代表本候选的参数确认。"]
    path = workspace / "第006代金融逻辑与训练参数待确认.txt"
    with path.open("w", encoding="utf-8-sig", newline="\n") as handle:
        handle.write("\n".join(lines) + "\n")
    return path
