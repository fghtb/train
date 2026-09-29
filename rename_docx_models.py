from pathlib import Path
from docx import Document

src = Path(r'D:\competition\train\_transaction\word_cn_names_20260926\BASELINE_original.docx')
dst = Path(r'D:\competition\train\_transaction\word_cn_names_20260926\MODIFIED_FILE.docx')
names = {
    'model_009_overnight_pressure_financial_v1': '隔夜信息与日内价格压力',
    'model_010_impact_liquidity_financial_v1': '真实波幅与流动性价格冲击',
    'model_011_downside_drawdown_financial_v1': '下行风险与回撤状态',
    'model_012_market_relative_financial_v1': '市场状态与个股相对收益',
}
table_names = {
    'model_009_overnight_pressure_financial_v1': '009 隔夜信息与日内价格压力',
    'model_010_impact_liquidity_financial_v1': '010 真实波幅与流动性价格冲击',
    'model_011_downside_drawdown_financial_v1': '011 下行风险与回撤状态',
    'model_012_market_relative_financial_v1': '012 市场状态与个股相对收益',
}
doc = Document(src)
for p in doc.paragraphs:
    for old, new in names.items():
        if old in p.text:
            for run in p.runs:
                if old in run.text:
                    run.text = run.text.replace(old, new)
for table in doc.tables:
    for row in table.rows:
        for cell in row.cells:
            for p in cell.paragraphs:
                for old, new in table_names.items():
                    for run in p.runs:
                        if old in run.text:
                            run.text = run.text.replace(old, new)
doc.save(dst)
print(dst)
