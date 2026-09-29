import json
from pathlib import Path

run = Path(r'D:\competition\train\runs\model_013_intraday_acceptance_financial_v1_20260926_160813_863636')
s = json.loads((run / 'summary.json').read_text(encoding='utf-8'))
cv = s['cross_validation']['aggregate']
print('2024_diagnostic', {k: s['metrics'].get(k) for k in ['final_score', 'ic_mean', 'icir', 'annual_excess', 'mean_turnover', 'top1_annual_ret']})
print('cv_mean', {k: v.get('mean') for k, v in cv.items() if isinstance(v, dict) and 'mean' in v})
print('cv_min_score', cv['final_score']['min'])
print('folds', [(x['name'], x['best_iteration'], x['metrics']['final_score']) for x in s['cross_validation']['folds']])
print('rows', s['rows'])
print('features', s['feature_columns'])
print('artifacts', s['artifacts'])
print('importance')
print((run / 'feature_importance.csv').read_text(encoding='utf-8'))
