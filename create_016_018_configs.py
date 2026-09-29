import json
from pathlib import Path

import financial_logic

root = Path(r"D:\competition\train")
base = json.loads((root / "configs" / "model_015_information_liquidity_financial_v1.json").read_text(encoding="utf-8"))
candidates = [
    (
        "016", "overnight_market_state", "model_016_overnight_market_state_financial_v1", "隔夜信息与市场状态",
        "检验隔夜信息和日内压力在市场共同方向、广度、分化度下是否呈现不同的个股相对收益。",
        "保留gap_1、oc_return与市场中位数收益、市场广度、市场分化度、个股残差收益、残差动量和相对波动；不加入流动性冲击，避免把两个机制混在一起。",
        "第016代隔夜信息与市场状态研究候选.txt",
    ),
    (
        "017", "downside_liquidity_stress", "model_017_downside_liquidity_stress_financial_v1", "下行风险与流动性压力",
        "检验负收益风险、回撤状态和流动性冲击共同出现时是否形成可识别的压力状态。",
        "使用20期下行风险与20期价格冲击，放弃60期重复窗口和全部对应排名，减少冗余并保持压力机制清晰。",
        "第017代下行风险与流动性压力研究候选.txt",
    ),
    (
        "018", "market_liquidity_state", "model_018_market_liquidity_state_financial_v1", "市场状态与流动性冲击",
        "检验市场共同状态是否能够调节个股的真实波幅和流动性价格冲击，寻找兼顾IC、收益和换手的状态模型。",
        "组合市场中位数收益、市场广度、市场分化度、相对波动与真实波幅、ATR、异常冲击和Amihud；不使用隔夜方向、下行回撤和价格形态。",
        "第018代市场状态与流动性冲击研究候选.txt",
    ),
]
for number, mode, model_name, cn_name, plan, change, ref_name in candidates:
    config = json.loads(json.dumps(base, ensure_ascii=False))
    config["model_name"] = model_name
    config["confirmation_token"] = f"CONFIRM_MODEL_{number}_NEW_LOGIC"
    config["feature_mode"] = mode
    config["optimization_plan"] = {"plan": plan, "change": change, "next": "采用相同三折滚动协议比较三折平均综合得分、最差折、Rank IC、ICIR、年化超额、换手和2024诊断，不预设方向。"}
    config["research_references"] = [{"title": ref_name[:-4], "path": str(root / ref_name), "use": plan}]
    config["financial_logic"] = financial_logic.make_policy(mode)
    config["comparison_plan"] = {
        "historical_reference": "连续三代独立机制研究；不加载前代模型，不以单一指标预设结果。",
        "candidate_protocol": "WF2021_2023_INNER_H2_PURGE1_V1",
        "selection_years": [2021, 2022, 2023],
        "diagnostic_year": 2024,
        "test_usage": "仅按题目测试集X生成预测，不读取测试标签。",
        "primary_decision_metric": "三折平均综合得分；同时检查最差折、Rank IC、ICIR、年化超额、换手和并列预测。",
    }
    out = root / "configs" / f"{model_name}.json"
    out.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(out)
