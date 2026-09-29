import json
from pathlib import Path
import financial_logic
root=Path(r"D:\competition\train")
base=json.loads((root/"configs"/"model_018_market_liquidity_state_financial_v1.json").read_text(encoding="utf-8"))
items=[("019","conditional_information_state","model_019_conditional_information_state_financial_v1","条件信息与市场状态","检验隔夜信息在市场共同状态中的条件一致性。","组合隔夜信息与市场状态，并加入跳空与市场广度、异常冲击与市场分化的明确交互特征。","第019代条件信息与市场状态研究候选.txt"),("020","stress_transmission","model_020_stress_transmission_financial_v1","压力状态传导","检验市场下跌广度是否调节个股下行风险与流动性冲击。","组合市场状态、真实波幅、Amihud、下行风险，并加入下行频率与市场广度、流动性冲击与市场广度的交互特征。","第020代压力状态传导研究候选.txt")]
for n,mode,name,cn,plan,change,ref in items:
 c=json.loads(json.dumps(base,ensure_ascii=False)); c["model_name"]=name; c["confirmation_token"]=f"CONFIRM_MODEL_{n}_NEW_LOGIC"; c["feature_mode"]=mode; c["optimization_plan"]={"plan":plan,"change":change,"next":"用同一三折滚动协议检查综合得分、最差折、Rank IC、ICIR、年化超额、换手和2024诊断。"}; c["research_references"]=[{"title":ref[:-4],"path":str(root/ref),"use":plan}]; c["financial_logic"]=financial_logic.make_policy(mode); c["comparison_plan"]={"historical_reference":"围绕018优点做机制扩展，不进行参数微调。","candidate_protocol":"WF2021_2023_INNER_H2_PURGE1_V1","selection_years":[2021,2022,2023],"diagnostic_year":2024,"test_usage":"仅按题目测试集X生成预测，不读取测试标签。","primary_decision_metric":"三折平均综合得分，同时检查最差折、Rank IC、ICIR、年化超额、换手和并列预测。"}; (root/"configs"/(name+".json")).write_text(json.dumps(c,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
