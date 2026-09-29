import csv
import json
from pathlib import Path

workspace = Path(r'D:\competition\train')
out_dir = workspace / 'reports' / 'research_1-XXX'
out_dir.mkdir(parents=True, exist_ok=True)
labels = {
    '031': '方向性冲击去制度标记',
    '032': '方向性冲击市场状态',
    '033': '方向性冲击风险调整',
    '034': '市场压力传导精简',
    '035': '广度分化与方向性冲击',
    '036': '精简压力与方向性冲击',
    '037': '风险调整方向性精简',
    '038': '低换手风险冲击组合',
}

latest = {}
for summary_path in workspace.glob('runs/research_1_*_*/summary.json'):
    data = json.loads(summary_path.read_text(encoding='utf-8'))
    code = data['model_name'].split('_')[2]
    status_path = summary_path.parent / 'run_status.json'
    status = json.loads(status_path.read_text(encoding='utf-8')).get('status') if status_path.exists() else 'unknown'
    if status != 'complete':
        continue
    previous = latest.get(code)
    if previous is None or data['run_id'] > previous['run_id']:
        latest[code] = data

rows = []
for code, name in labels.items():
    data = latest.get(code)
    if data is None:
        rows.append({'code': f'1-{code}', 'name': name, 'status': '未完成'})
        continue
    agg = data['cross_validation']['aggregate']
    score = agg['final_score']
    valid = score.get('mean') is not None and data['metrics'].get('final_score') is not None
    rows.append({
        'code': f'1-{code}', 'name': name, 'status': '有效' if valid else '无效预测',
        'features': data['feature_count'], 'cv_score_mean': score.get('mean'), 'cv_score_worst': score.get('min'),
        'cv_score_std': score.get('std'), 'rank_ic': agg['ic_mean'].get('mean'), 'icir': agg['icir'].get('mean'),
        'annual_excess': agg['annual_excess'].get('mean'), 'turnover': agg['mean_turnover'].get('mean'),
        'diagnostic_2024_score': data['metrics'].get('final_score'), 'best_iteration': data['best_iteration'],
        'test_rows': data['rows']['test'], 'run_dir': data['run_dir'],
    })

fields = ['code', 'name', 'status', 'features', 'cv_score_mean', 'cv_score_worst', 'cv_score_std', 'rank_ic', 'icir', 'annual_excess', 'turnover', 'diagnostic_2024_score', 'best_iteration', 'test_rows', 'run_dir']
with (out_dir / '重要数据对比.csv').open('w', encoding='utf-8-sig', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)

def fmt(v, pct=False):
    if v is None:
        return '无'
    return f'{v:.2%}' if pct else f'{v:.6f}'

valid_rows = [r for r in rows if r['status'] == '有效']
valid_rows.sort(key=lambda r: r['cv_score_mean'], reverse=True)
lines = [
    '1-XXX验证模型重要数据对比报告', '',
    '报告性质：独立研究报告，不写入累计模型训练总报告。',
    '数据：仅使用题目提供的训练集.csv和测试集_X.csv。',
    '验证协议：WF2021_2023_INNER_H2_PURGE1_V1；2024年仅作诊断；测试集仅生成预测。',
    '每个候选：7次独立拟合；不加载旧模型；测试预测行数应为1,599,600。', '',
    '综合得分公式：0.4×Rank IC + 0.3×年化Top1超额收益 + 0.3×(1-平均换手率)。',
    '主要排序：三折平均综合得分；同时检查最差折、标准差、Rank IC、ICIR和换手率。', '',
    '有效候选排名',
    '排名  代号   名称                         三折均值  最差折   标准差  Rank IC    ICIR   年化超额  换手率  2024诊断分',
]
for i, r in enumerate(valid_rows, 1):
    lines.append(
        f"{i:>2}    {r['code']:<5} {r['name']:<28} {fmt(r['cv_score_mean']):>8} "
        f"{fmt(r['cv_score_worst']):>8} {fmt(r['cv_score_std']):>8} {fmt(r['rank_ic']):>8} "
        f"{fmt(r['icir']):>8} {fmt(r['annual_excess'], True):>8} {fmt(r['turnover'], True):>8} "
        f"{fmt(r['diagnostic_2024_score']):>8}"
    )
lines += ['', '全部候选状态']
for r in rows:
    if r['status'] == '有效':
        lines.append(f"{r['code']} {r['name']}：有效；特征数{r['features']}；最终轮数{r['best_iteration']}；测试预测{r['test_rows']}行。")
    else:
        lines.append(f"{r['code']} {r['name']}：无效预测；综合得分和Rank IC不可用，不能参与排名。")
lines += ['', '初步结论']
if valid_rows:
    best = valid_rows[0]
    lines.append(f"当前有效候选中，{best['code']} {best['name']}的三折平均综合得分最高，为{fmt(best['cv_score_mean'])}，最差折{fmt(best['cv_score_worst'])}。")
    lines.append('无效预测候选不按0分处理；这表明固定参数下预测发生常数化或关键指标不可计算，应先排查特征可辨识性。')
lines.append('详细运行目录保留在D:\\competition\\train\\runs；本报告仅保留重要比较数据。')
(out_dir / '1-XXX验证模型重要数据对比报告.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')
print(out_dir / '1-XXX验证模型重要数据对比报告.txt')
print(out_dir / '重要数据对比.csv')
