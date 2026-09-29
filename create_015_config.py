import json
from pathlib import Path

import financial_logic

root = Path(r"D:\competition\train")
base = json.loads((root / "configs" / "model_014_price_discovery_efficiency_financial_v1.json").read_text(encoding="utf-8"))
config = base.copy()
config["model_name"] = "model_015_information_liquidity_financial_v1"
config["confirmation_token"] = "CONFIRM_MODEL_015_NEW_LOGIC"
config["feature_mode"] = "information_liquidity"
config["optimization_plan"] = {
    "plan": "独立研究非交易时段信息到达后的价格冲击与流动性承接，不加入市场相对、下行回撤、日内形态和路径效率特征。",
    "change": "组合gap_1、oc_return与真实波幅、ATR、异常波幅和Amihud流动性特征；保留涨跌停约束标记，剔除隔夜方向的重复横截面排名。",
    "next": "按同一三折滚动协议比较方向信息能力、综合稳定性、换手和2024诊断，不预设一定提升。",
}
config["research_references"] = [{
    "title": "第015代信息冲击与流动性承接研究候选",
    "path": r"D:\competition\train\第015代信息冲击与流动性承接研究候选.txt",
    "use": "隔夜信息、日内压力、价格冲击与流动性承接的事前金融假设。",
}]
config["financial_logic"] = financial_logic.make_policy("information_liquidity")
config["comparison_plan"] = {
    "historical_reference": "基于009和010的互补方向进行独立组合检验，不预设优于任一前代。",
    "candidate_protocol": "WF2021_2023_INNER_H2_PURGE1_V1",
    "selection_years": [2021, 2022, 2023],
    "diagnostic_year": 2024,
    "test_usage": "仅按题目测试集X生成预测，不读取测试标签。",
    "primary_decision_metric": "三折平均综合得分；同时检查最差折、Rank IC、ICIR、年化超额、换手和并列预测。",
}
out = root / "configs" / "model_015_information_liquidity_financial_v1.json"
out.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
current_fp = config["financial_logic"]["registry_fingerprint"]
for path in (root / "configs").glob("model_*.json"):
    try:
        old = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        continue
    if isinstance(old.get("financial_logic"), dict):
        old["financial_logic"]["registry_fingerprint"] = current_fp
        path.write_text(json.dumps(old, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(out)
print(json.dumps({"registry_fingerprint": current_fp, "allowed_features": config["financial_logic"]["allowed_features"], "excluded_features": config["financial_logic"]["excluded_features"]}, ensure_ascii=False, indent=2))
