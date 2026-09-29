import copy
import json
from pathlib import Path

from financial_logic import feature_catalog, policy_fingerprint, POLICY_VERSION

workspace = Path(__file__).parent
source = json.loads((workspace / 'configs' / 'model_030_high_score_synthesis_financial_v1.json').read_text(encoding='utf-8'))
designs = json.loads((workspace / 'research_designs_1-XXX' / '模型验证设计_1-031至1-038.json').read_text(encoding='utf-8'))['models']
modes = {
    '1-031': 'research_1_031', '1-032': 'research_1_032',
    '1-033': 'research_1_033', '1-034': 'research_1_034',
    '1-035': 'research_1_035', '1-036': 'research_1_036',
    '1-037': 'research_1_037', '1-038': 'research_1_038',
}
out_dir = workspace / 'configs' / 'research_1-XXX'
out_dir.mkdir(parents=True, exist_ok=True)
for d in designs:
    code = d['code']
    cfg = copy.deepcopy(source)
    slug = code.replace('-', '_')
    cfg['model_name'] = f"research_{slug}_financial_v1"
    cfg['confirmation_token'] = f"AUTO_RESEARCH_{slug}"
    cfg['auto_confirm'] = True
    cfg['feature_mode'] = modes[code]
    cfg['optimization_plan'] = {
        'plan': d['purpose'],
        'change': f"研究代号{code}：仅改变金融特征集合，LightGBM参数、时间协议和数据范围固定。",
        'next': '与指定对照模型比较三折平均综合得分、最差折、Rank IC、ICIR、年化超额收益和换手率。',
    }
    catalog = feature_catalog(cfg['feature_mode'])
    allowed = d['features']
    missing = sorted(set(allowed) - set(catalog))
    if missing:
        raise ValueError(f'{code} features not in generated catalog: {missing}')
    cfg['financial_logic'] = {
        'version': POLICY_VERSION,
        'registry_fingerprint': policy_fingerprint(),
        'allowed_features': allowed,
        'excluded_features': [name for name in catalog if name not in allowed],
        'display_logic_before_confirmation': True,
    }
    cfg['comparison_plan'] = {
        'historical_reference': f"研究代号{code}；对照模型：{d['control']}。",
        'candidate_protocol': 'WF2021_2023_INNER_H2_PURGE1_V1',
        'selection_years': [2021, 2022, 2023],
        'diagnostic_year': 2024,
        'test_usage': '仅按题目测试集X生成预测，不读取测试标签。',
        'primary_decision_metric': '三折平均综合得分，同时检查最差折、Rank IC、ICIR、年化超额、换手和并列预测。',
    }
    cfg['research_references'] = [{
        'title': f'研究设计{code} {d["name"]}',
        'path': str(workspace / 'research_designs_1-XXX' / '模型验证设计_1-031至1-038.txt'),
        'use': d['purpose'],
    }]
    out = out_dir / f'{code.replace("-", "_")}.json'
    out.write_text(json.dumps(cfg, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(out)
