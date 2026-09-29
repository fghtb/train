import json
from pathlib import Path

from docx import Document
from docx.enum.section import WD_ORIENT
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Inches, Pt, RGBColor

root = Path(r'D:\competition\train')
report_dir = root / 'reports' / 'research_1-XXX'
report_dir.mkdir(parents=True, exist_ok=True)
out = report_dir / '1-XXX验证模型重要数据对比报告.docx'
labels = {
    '031': '方向性冲击去制度标记', '032': '方向性冲击市场状态',
    '033': '方向性冲击风险调整', '034': '市场压力传导精简',
    '035': '广度分化与方向性冲击', '036': '精简压力与方向性冲击',
    '037': '风险调整方向性精简', '038': '低换手风险冲击组合',
}
latest = {}
for path in root.glob('runs/research_1_*_*/summary.json'):
    d = json.loads(path.read_text(encoding='utf-8'))
    code = d['model_name'].split('_')[2]
    status_path = path.parent / 'run_status.json'
    status = json.loads(status_path.read_text(encoding='utf-8')).get('status') if status_path.exists() else ''
    if status == 'complete' and (code not in latest or d['run_id'] > latest[code]['run_id']):
        latest[code] = d

def val(x, digits=4):
    return '无效' if x is None else f'{x:.{digits}f}'

def pct(x):
    return '无效' if x is None else f'{x:.2%}'

def shade(cell, fill):
    tc_pr = cell._tc.get_or_add_tcPr()
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn
    shd = OxmlElement('w:shd')
    shd.set(qn('w:fill'), fill)
    tc_pr.append(shd)

def set_cell(cell, text, bold=False, color='000000'):
    cell.text = ''
    p = cell.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run(str(text))
    run.bold = bold
    run.font.name = 'Microsoft YaHei'
    run.font.size = Pt(8.5)
    run.font.color.rgb = RGBColor.from_string(color)
    cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER

def table(doc, headers, rows, widths=None):
    t = doc.add_table(rows=1, cols=len(headers))
    t.style = 'Table Grid'
    t.autofit = False
    for i, h in enumerate(headers):
        set_cell(t.rows[0].cells[i], h, True, 'FFFFFF')
        shade(t.rows[0].cells[i], '263B53')
    for ridx, row in enumerate(rows):
        cells = t.add_row().cells
        for i, value in enumerate(row):
            set_cell(cells[i], value)
            if ridx % 2 == 0:
                shade(cells[i], 'EAF0F5')
    if widths:
        for row in t.rows:
            for i, width in enumerate(widths):
                row.cells[i].width = Inches(width)
    doc.add_paragraph().paragraph_format.space_after = Pt(2)
    return t

doc = Document()
sec = doc.sections[0]
sec.orientation = WD_ORIENT.LANDSCAPE
sec.page_width, sec.page_height = Inches(11), Inches(8.5)
sec.left_margin = sec.right_margin = Inches(0.55)
sec.top_margin = sec.bottom_margin = Inches(0.55)
normal = doc.styles['Normal']
normal.font.name = 'Microsoft YaHei'
normal.font.size = Pt(9)
normal.font.color.rgb = RGBColor(0, 0, 0)
for style_name in ['Title', 'Heading 1', 'Heading 2']:
    style = doc.styles[style_name]
    style.font.name = 'Microsoft YaHei'
    style.font.color.rgb = RGBColor(0, 0, 0)

p = doc.add_paragraph()
title_run = p.add_run('1-XXX 验证模型重要数据对比报告')
title_run.bold = False
title_run.font.name = 'Microsoft YaHei'
title_run.font.size = Pt(22)
title_run.font.color.rgb = RGBColor(0, 0, 0)
p.paragraph_format.space_after = Pt(4)
p = doc.add_paragraph('本报告独立于累计模型训练总报告，仅汇总1-031至1-038的关键对比数据。所有模型使用相同三折滚动协议和固定参数；每代7次独立拟合。2024年仅作诊断，测试集只生成预测。')
p.paragraph_format.space_after = Pt(5)

doc.add_heading('主要对比', level=1)
rows = []
for code in labels:
    d = latest.get(code)
    if not d:
        rows.append([f'1-{code}', labels[code], '未完成', '-', '-', '-', '-', '-'])
        continue
    agg = d['cross_validation']['aggregate']
    score = agg['final_score'].get('mean')
    status = '有效' if score is not None and d['metrics'].get('final_score') is not None else '无效预测'
    rows.append([
        f'1-{code}', labels[code], status, str(d['feature_count']), val(score),
        val(agg['final_score'].get('min')), val(agg['ic_mean'].get('mean')),
        pct(agg['mean_turnover'].get('mean')),
    ])
table(doc, ['代号', '模型方向', '状态', '特征数', '三折均分', '最差折', 'Rank IC', '换手率'], rows,
      [0.65, 2.25, 0.85, 0.65, 0.95, 0.9, 0.9, 0.85])

doc.add_heading('关键诊断', level=1)
diag_rows = []
for code in labels:
    d = latest.get(code)
    if not d:
        continue
    agg = d['cross_validation']['aggregate']
    m = d['metrics']
    folds = agg['final_score']
    tie = m.get('top10_distinct_prediction_ratio_mean')
    diag_rows.append([
        f'1-{code}', val(agg['icir'].get('mean')), pct(agg['annual_excess'].get('mean')),
        val(folds.get('std')), val(m.get('final_score')),
        '无效' if tie is None else f'{tie:.2%}', str(d['best_iteration']),
    ])
table(doc, ['代号', 'ICIR', '三折年化超额', '得分标准差', '2024诊断分', 'Top10预测区分度', '轮数'], diag_rows,
      [0.65, 0.8, 1.15, 1.0, 1.0, 1.35, 0.7])

doc.add_heading('历史参照', level=1)
anchors = [
    ['020', '压力状态传导', '0.296137', '0.284022', '0.065362', '13.23%'],
    ['025', '市场扩散与分化转移', '0.295017', '0.281661', '0.048292', '8.91%'],
    ['027', '方向性流动性冲击不对称', '0.308346', '0.287439', '0.040991', '23.38%'],
    ['030', '风险调整后的方向性流动性冲击', '0.289448', '0.259014', '0.068000', '17.98%'],
]
table(doc, ['代号', '历史模型', '三折均分', '最差折', 'Rank IC', '换手率'], anchors,
      [0.65, 2.8, 1.0, 1.0, 1.0, 1.0])

doc.add_heading('结果解读', level=1)
valid = []
invalid = []
for code in labels:
    d = latest.get(code)
    if not d:
        continue
    agg = d['cross_validation']['aggregate']
    if agg['final_score'].get('mean') is not None and d['metrics'].get('final_score') is not None:
        valid.append((code, d))
    else:
        invalid.append((code, d))
valid.sort(key=lambda item: item[1]['cross_validation']['aggregate']['final_score']['mean'], reverse=True)
if valid:
    code, best = valid[0]
    best_score = best['cross_validation']['aggregate']['final_score']['mean']
    best_worst = best['cross_validation']['aggregate']['final_score']['min']
    doc.add_paragraph(f"本组有效候选中，1-{code}“{labels[code]}”表现最好：三折平均综合得分{best_score:.6f}，最差折{best_worst:.6f}。")
if invalid:
    codes = '、'.join(f'1-{code}' for code, _ in invalid)
    doc.add_paragraph(f'{codes}的Rank IC和综合得分无法计算，不能按零分参与排名。它们的Top10预测区分度极低或为0，提示预测出现严重并列/常数化；尽管部分模型换手率较低，也不代表排序能力有效。')
doc.add_paragraph('综合得分公式：0.4×Rank IC + 0.3×年化Top1超额收益 + 0.3×(1−平均换手率)。模型选择以三折平均综合得分为主，同时检查最差折、ICIR、年化超额、换手和预测区分度。2024年结果只作诊断；测试集没有标签成绩。')
doc.add_paragraph('所有运行目录、模型和预测文件均保留在各自runs目录；本报告没有写入累计模型训练总报告。')

doc.save(out)
print(out)
