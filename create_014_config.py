import json
from pathlib import Path

import financial_logic

root = Path(r'D:\competition\train')
base = json.loads((root / 'configs' / 'model_013_intraday_acceptance_financial_v1.json').read_text(encoding='utf-8'))
config = base.copy()
config['model_name'] = 'model_014_price_discovery_efficiency_financial_v1'
config['confirmation_token'] = 'CONFIRM_MODEL_014_NEW_LOGIC'
config['feature_mode'] = 'price_discovery_efficiency'
config['optimization_plan'] = {
    'plan': '独立研究价格发现效率与买卖拒绝，不继承009至013的隔夜、真实波幅冲击、下行回撤、市场相对收益或相对成交参与逻辑。',
    'change': '使用实体区间比例、上下影线比例、20期路径效率和20期收益自相关，保留涨跌停标记；剔除013代的日内区间和交易参与特征。',
    'next': '按相同三折滚动协议比较跨年份稳定性、Rank IC、ICIR、换手、综合得分和2024诊断；不预设方向。',
}
config['research_references'] = [{
    'title': '第014代价格发现效率研究候选',
    'path': r'D:\competition\train\第014代价格发现效率研究候选.txt',
    'use': '实体接受、影线价格拒绝、路径效率与收益序列依赖的事前金融假设。',
}]
config['financial_logic'] = financial_logic.make_policy('price_discovery_efficiency')
config['comparison_plan'] = {
    'historical_reference': '独立新逻辑研究，不绑定前代优劣。',
    'candidate_protocol': 'WF2021_2023_INNER_H2_PURGE1_V1',
    'selection_years': [2021, 2022, 2023],
    'diagnostic_year': 2024,
    'test_usage': '仅按题目测试集X生成预测，不读取测试标签。',
    'primary_decision_metric': '三折平均综合得分；同时检查最差折、Rank IC、ICIR、年化超额、换手和并列预测。',
}
out = root / 'configs' / 'model_014_price_discovery_efficiency_financial_v1.json'
out.write_text(json.dumps(config, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
current_fp = config['financial_logic']['registry_fingerprint']
for path in (root / 'configs').glob('model_*.json'):
    try:
        old = json.loads(path.read_text(encoding='utf-8'))
    except Exception:
        continue
    if isinstance(old.get('financial_logic'), dict):
        old['financial_logic']['registry_fingerprint'] = current_fp
        path.write_text(json.dumps(old, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
print(out)
print(json.dumps({'registry_fingerprint': current_fp, 'allowed_features': config['financial_logic']['allowed_features'], 'excluded_features': config['financial_logic']['excluded_features']}, ensure_ascii=False, indent=2))
