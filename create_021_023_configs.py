# -*- coding: utf-8 -*-
import json
from pathlib import Path
import financial_logic
root=Path(r"D:\competition\train")
base=json.loads((root/"configs"/"model_020_stress_transmission_financial_v1.json").read_text(encoding="utf-8"))
items=[("021","stress_core","model_021_stress_core_financial_v1","压力传导核心","把020拆成更精简的市场压力、异常波动、流动性冲击核心，检验020高分是否来自少数机制。","去除市场中位数收益、ATR、下行波动和回撤，只保留市场广度、分化、异常冲击、Amihud和压力交互。","第021代压力传导核心研究候选.txt"),("022","stress_recovery","model_022_stress_recovery_financial_v1","压力恢复与反转","研究个股在市场压力状态下从近期低点恢复的条件收益。","保留市场方向和广度、20期下行风险、负收益频率、回撤，并加入低点恢复和压力恢复交互。","第022代压力恢复与反转研究候选.txt"),("023","constraint_stress","model_023_constraint_stress_financial_v1","制度约束与压力传导","研究涨跌停制度约束是否改变市场压力和流动性冲击的预测作用。","保留制度标记、市场广度和分化、真实波幅、异常冲击、Amihud和两项制度压力交互。","第023代制度约束与压力传导研究候选.txt")]
for n,mode,name,cn,plan,change,ref in items:
 c=json.loads(json.dumps(base,ensure_ascii=False)); c["model_name"]=name; c["confirmation_token"]=f"CONFIRM_MODEL_{n}_NEW_LOGIC"; c["feature_mode"]=mode; c["optimization_plan"]={"plan":plan,"change":change,"next":"采用同一三折滚动协议比较综合得分、最差折、Rank IC、ICIR、年化超额、换手和2024诊断。"}; c["research_references"]=[{"title":ref[:-4],"path":str(root/ref),"use":plan}]; c["financial_logic"]=financial_logic.make_policy(mode); c["comparison_plan"]={"historical_reference":"基于020压力传导结果的机制拆分，不进行参数微调。","candidate_protocol":"WF2021_2023_INNER_H2_PURGE1_V1","selection_years":[2021,2022,2023],"diagnostic_year":2024,"test_usage":"仅按题目测试集X生成预测，不读取测试标签。","primary_decision_metric":"三折平均综合得分，同时检查最差折、Rank IC、ICIR、年化超额、换手和并列预测。"}; (root/"configs"/(name+".json")).write_text(json.dumps(c,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
