import json
from pathlib import Path

import financial_logic

root = Path(r'D:\competition\train')
base = json.loads((root / 'configs' / 'model_012_market_relative_financial_v1.json').read_text(encoding='utf-8'))
config = base.copy()
config['model_name'] = 'model_013_intraday_acceptance_financial_v1'
config['confirmation_token'] = 'CONFIRM_MODEL_013_NEW_LOGIC'
config['feature_mode'] = 'intraday_acceptance'
config['optimization_plan'] = {
    'plan': '独立研究日内价格接受与交易参与，不继承009至012的隔夜、真实波幅冲击、下行回撤或市场相对收益组合。',
    'change': '使用题目提供的OHLCV和涨跌停字段，审查日内振幅、收盘区间位置、相对成交量和相对成交额；amount_per_vol保留在目录但不进入训练。',
    'next': '按相同三折滚动协议比较跨年份稳定性、Rank IC、ICIR、换手、综合得分和2024诊断；不预设方向。',
}
config['research_references'] = [{
    'title': '第013代日内价格接受与交易参与研究候选',
    'path': r'D:\competition\train\第013代日内价格接受研究候选.txt',
    'use': '收盘在日内区间的位置、日内波幅和相对交易参与的事前金融假设。',
}]
config['financial_logic'] = financial_logic.make_policy('intraday_acceptance')
config['comparison_plan'] = {
    'historical_reference': '独立新逻辑研究，不绑定前代优劣。',
    'candidate_protocol': 'WF2021_2023_INNER_H2_PURGE1_V1',
    'selection_years': [2021, 2022, 2023],
    'diagnostic_year': 2024,
    'test_usage': '仅按题目测试集X生成预测，不读取测试标签。',
    'primary_decision_metric': '三折平均综合得分；同时检查最差折、Rank IC、ICIR、年化超额、换手和并列预测。',
}
out = root / 'configs' / 'model_013_intraday_acceptance_financial_v1.json'
out.write_text(json.dumps(config, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
print(out)
print(json.dumps({'allowed_features': config['financial_logic']['allowed_features'], 'excluded_features': config['financial_logic']['excluded_features'], 'registry_fingerprint': config['financial_logic']['registry_fingerprint']}, ensure_ascii=False, indent=2))
