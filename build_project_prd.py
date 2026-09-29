from pathlib import Path
from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
from docx.shared import Inches, Pt, RGBColor
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

OUT = Path(r"D:\competition\train\项目PRD_LightGBM竞赛训练系统.docx")

def set_cell_shading(cell, fill):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = tcPr.find(qn('w:shd'))
    if shd is None:
        shd = OxmlElement('w:shd')
        tcPr.append(shd)
    shd.set(qn('w:fill'), fill)

def set_cell_border(cell, color='D9D9D9', sz='6'):
    tc = cell._tc
    tcPr = tc.get_or_add_tcPr()
    borders = tcPr.first_child_found_in('w:tcBorders')
    if borders is None:
        borders = OxmlElement('w:tcBorders')
        tcPr.append(borders)
    for edge in ('top', 'left', 'bottom', 'right', 'insideH', 'insideV'):
        tag = 'w:' + edge
        element = borders.find(qn(tag))
        if element is None:
            element = OxmlElement(tag)
            borders.append(element)
        element.set(qn('w:val'), 'single')
        element.set(qn('w:sz'), sz)
        element.set(qn('w:space'), '0')
        element.set(qn('w:color'), color)

def set_cell_text(cell, text, bold=False, color='000000', size=9.5):
    cell.text = ''
    p = cell.paragraphs[0]
    p.paragraph_format.space_after = Pt(2)
    p.paragraph_format.line_spacing = 1.05
    r = p.add_run(str(text))
    r.bold = bold
    r.font.size = Pt(size)
    r.font.color.rgb = RGBColor.from_string(color)
    r.font.name = 'Microsoft YaHei'
    r._element.rPr.rFonts.set(qn('w:eastAsia'), 'Microsoft YaHei')
    cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER

def add_table(doc, headers, rows, widths=None):
    table = doc.add_table(rows=1, cols=len(headers))
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.style = 'Table Grid'
    header_pr = table.rows[0]._tr.get_or_add_trPr()
    header_repeat = OxmlElement('w:tblHeader')
    header_repeat.set(qn('w:val'), 'true')
    header_pr.append(header_repeat)
    for i, h in enumerate(headers):
        set_cell_text(table.rows[0].cells[i], h, bold=True, color='FFFFFF', size=9)
        set_cell_shading(table.rows[0].cells[i], '234E70')
    for ridx, row in enumerate(rows):
        cells = table.add_row().cells
        for i, val in enumerate(row):
            set_cell_text(cells[i], val, size=9)
            set_cell_shading(cells[i], 'F4F7FA' if ridx % 2 else 'FFFFFF')
    if widths:
        for row in table.rows:
            for idx, width in enumerate(widths):
                row.cells[idx].width = Inches(width)
    for row in table.rows:
        row_pr = row._tr.get_or_add_trPr()
        cant_split = OxmlElement('w:cantSplit')
        row_pr.append(cant_split)
        for cell in row.cells:
            set_cell_border(cell)
    doc.add_paragraph().paragraph_format.space_after = Pt(1)
    return table

def add_bullet(doc, text, level=0):
    p = doc.add_paragraph(style='List Bullet' if level == 0 else 'List Bullet 2')
    p.paragraph_format.space_after = Pt(3)
    p.paragraph_format.line_spacing = 1.08
    p.add_run(text)
    return p

def add_num(doc, text):
    p = doc.add_paragraph(style='List Number')
    p.paragraph_format.space_after = Pt(3)
    p.paragraph_format.line_spacing = 1.08
    p.add_run(text)
    return p

def add_step(doc, number, text):
    p = doc.add_paragraph(style='Normal')
    p.paragraph_format.left_indent = Inches(0.18)
    p.paragraph_format.first_line_indent = Inches(-0.18)
    p.paragraph_format.space_after = Pt(3)
    p.paragraph_format.line_spacing = 1.08
    p.add_run(f'{number}. {text}')
    return p

def add_heading(doc, text, level=1):
    p = doc.add_paragraph(style=f'Heading {level}')
    p.paragraph_format.keep_with_next = True
    p.add_run(text)
    return p

def add_para(doc, text, bold_lead=None):
    p = doc.add_paragraph(style='Normal')
    p.paragraph_format.space_after = Pt(6)
    p.paragraph_format.line_spacing = 1.18
    if bold_lead and text.startswith(bold_lead):
        p.add_run(bold_lead).bold = True
        p.add_run(text[len(bold_lead):])
    else:
        p.add_run(text)
    return p

def configure(doc):
    sec = doc.sections[0]
    sec.top_margin = Inches(0.7)
    sec.bottom_margin = Inches(0.65)
    sec.left_margin = Inches(0.8)
    sec.right_margin = Inches(0.8)
    styles = doc.styles
    normal = styles['Normal']
    normal.font.name = 'Microsoft YaHei'
    normal._element.rPr.rFonts.set(qn('w:eastAsia'), 'Microsoft YaHei')
    normal.font.size = Pt(10.5)
    for name, size in [('Title', 22), ('Heading 1', 15), ('Heading 2', 12), ('Heading 3', 10.5)]:
        st = styles[name]
        st.font.name = 'Microsoft YaHei'
        st._element.rPr.rFonts.set(qn('w:eastAsia'), 'Microsoft YaHei')
        st.font.color.rgb = RGBColor(0, 0, 0)
        st.font.size = Pt(size)
        st.font.bold = True
    # Remove the built-in Title style's decorative border in Word.
    title_ppr = styles['Title']._element.get_or_add_pPr()
    title_borders = title_ppr.find(qn('w:pBdr'))
    if title_borders is not None:
        title_ppr.remove(title_borders)
    styles['Title'].paragraph_format.space_after = Pt(8)
    styles['Heading 1'].paragraph_format.space_before = Pt(12)
    styles['Heading 1'].paragraph_format.space_after = Pt(5)
    styles['Heading 2'].paragraph_format.space_before = Pt(8)
    styles['Heading 2'].paragraph_format.space_after = Pt(3)
    header = sec.header.paragraphs[0]
    header.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    r = header.add_run('LightGBM 竞赛训练系统  |  项目PRD')
    r.font.name = 'Microsoft YaHei'; r.font.size = Pt(8); r.font.color.rgb = RGBColor(100, 100, 100)
    footer = sec.footer.paragraphs[0]
    footer.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = footer.add_run('内部项目文档  |  版本 1.0  |  2026-09-25')
    r.font.name = 'Microsoft YaHei'; r.font.size = Pt(8); r.font.color.rgb = RGBColor(100, 100, 100)

def build():
    doc = Document()
    configure(doc)

    title = doc.add_paragraph(style='Title')
    title.alignment = WD_ALIGN_PARAGRAPH.LEFT
    title.add_run('LightGBM 竞赛训练系统项目需求文档')
    sub = doc.add_paragraph()
    sub.paragraph_format.space_after = Pt(14)
    r = sub.add_run('版本 1.0  |  面向赛题五的可复现多模型研究与提交系统')
    r.font.size = Pt(11); r.font.color.rgb = RGBColor(80, 80, 80)
    add_para(doc, '本项目将题目提供的训练集和测试集转换为可重复运行的 LightGBM 多模型训练流程。系统要求每代模型独立、时间顺序清晰、特征具有金融解释、结果可比较，并持续生成训练报告、模型文件和提交预测。本文定义产品目标、使用流程、数据协议、模型迭代规则、产物和验收标准，作为后续开发与评审依据。')

    add_heading(doc, '一 项目概述', 1)
    add_table(doc, ['项目项', '定义'], [
        ('项目名称', '赛题五 LightGBM 多模型训练与比较系统'),
        ('项目目标', '在仅使用题目数据的前提下，训练多个可比较的股票横截面预测模型，并输出可提交预测与累计研究报告'),
        ('核心用户', '模型研究者、竞赛参赛者、复核训练过程的协作者'),
        ('交付形态', '每代独立运行目录、配置快照、LightGBM模型、预测文件、特征重要性、累计Markdown/TXT/Word报告'),
        ('当前基准', '第006代金融逻辑模型；第007、008代为参数改进实验'),
        ('当前最新结果', '第008代三折综合得分均值0.184754，Rank IC均值0.059347')
    ], [1.5, 5.7])

    add_heading(doc, '二 产品目标与非目标', 1)
    add_heading(doc, '2.1 产品目标', 2)
    for x in [
        '建立一条从参数确认、特征审查、滚动训练、测试预测到报告更新的完整流水线。',
        '让每次模型迭代都能回答三个问题：改了什么、为什么改、结果是否改善。',
        '使用时间序列交叉验证评估跨年份稳定性，避免把单一年份表现当作模型能力。',
        '把特征的金融逻辑、可用时点、预期风险和失效情形记录为模型的一部分。',
        '保存足够的运行快照，使协作者可以复查配置、代码版本、输入路径和输出文件。'
    ]: add_bullet(doc, x)
    add_heading(doc, '2.2 非目标', 2)
    for x in [
        '不构建实时交易系统，不连接券商或交易接口。',
        '不下载行情、利率、指数、行业分类或其他外部资料。',
        '没有测试集标签时，不宣称测试集真实成绩，只生成预测文件。',
        '模拟组合归因用于研究诊断，不等同于实盘可执行收益。'
    ]: add_bullet(doc, x)

    add_heading(doc, '三 用户场景与核心流程', 1)
    add_heading(doc, '3.1 主要用户场景', 2)
    add_table(doc, ['场景', '用户动作', '系统结果'], [
        ('开始一代训练', '查看金融逻辑与全部参数，输入确认口令', '创建全新运行目录并执行7次独立拟合'),
        ('比较候选模型', '查看三折指标、最差折、换手和诊断结果', '在累计报告中记录本代与历史代差异'),
        ('调整一个参数', '复制配置，只修改一组参数', '形成可解释的单变量实验'),
        ('复核历史运行', '打开运行目录和快照文件', '还原当时配置、特征、数据清单和模型产物'),
        ('准备提交', '读取最新模型的submission.csv', '得到与测试集主键顺序一致的预测文件')
    ], [1.3, 2.9, 3.0])
    add_heading(doc, '3.2 标准用户流程', 2)
    for x in [
        '创建或复制一份代际配置，并登记本代唯一名称、确认口令和修改目的。',
        '运行预览：系统展示数据路径、时间协议、特征白名单、金融逻辑、LightGBM参数和保存位置。',
        '人工确认：只有确认口令完全匹配，系统才读取大文件并开始训练。',
        '滚动验证：每个年份折先确定早停轮数，再用历史数据从零重训并评估外层年份。',
        '最终训练：取三折最佳轮数的中位数，在2018至2023年合规历史上从零训练最终模型。',
        '诊断与预测：2024年只做模拟组合诊断；测试集X只生成预测，不读取测试标签。',
        '交付与记录：保存模型和提交文件，重建累计报告，更新TXT和Word版本。'
    ]: add_num(doc, x)

    add_heading(doc, '四 数据产品与时间协议', 1)
    add_heading(doc, '4.1 数据来源', 2)
    add_table(doc, ['数据集', '路径', '用途'], [
        ('训练集', r'D:\competition\赛题五数据\训练集.csv', '特征计算、滚动训练、历史评估与2024诊断'),
        ('测试集X', r'D:\competition\赛题五数据\测试集_X.csv', '最终模型预测和提交文件生成')
    ], [1.2, 3.7, 2.3])
    add_para(doc, '系统只接受上述题目数据文件。输入数据清单、字段清单、文件哈希和样本快照保存在每代运行目录中，便于复核本次运行实际使用的来源。')
    add_heading(doc, '4.2 时间区间', 2)
    add_table(doc, ['阶段', '日期范围', '用途'], [
        ('历史训练总范围', '2018-01-02 至 2023-12-31', '最终模型可使用的历史数据'),
        ('滚动外层评估', '2021、2022、2023全年', '评估跨年份稳定性并计算综合得分'),
        ('2024诊断', '2024-01-01 至 2024-12-31', '模拟组合及附加诊断；不作为本轮参数选择依据'),
        ('测试预测', '2025-01-02 至 2026-06-08', '只生成预测，不读取测试标签')
    ], [1.5, 2.7, 3.0])
    add_heading(doc, '4.3 滚动验证协议', 2)
    add_table(doc, ['折次', '内层训练', '早停窗口', '外层评估'], [
        ('2021折', '2018-01-02 至 2020-06-30', '2020-07-01 至 2020-12-31', '2021全年'),
        ('2022折', '2018-01-02 至 2021-06-30', '2021-07-01 至 2021-12-31', '2022全年'),
        ('2023折', '2018-01-02 至 2022-06-30', '2022-07-01 至 2022-12-31', '2023全年')
    ], [1.1, 2.3, 2.3, 1.5])
    add_para(doc, '每折先在内层早停窗口确定最佳轮数，再用截至该折历史末日的数据新建模型重训。最终轮数采用三折内层最佳轮数的中位数。标签兑现日跨越阶段边界时按协议处理，避免使用尚未在当时可获得的信息。')

    add_heading(doc, '五 特征与金融逻辑要求', 1)
    add_heading(doc, '5.1 金融逻辑优先原则', 2)
    add_para(doc, '每个进入训练的特征必须在训练前登记金融含义、计算口径、可用时点、预期换手影响和失效风险。重要性高并不等于应保留；如果特征无法说明它反映的市场机制，系统应将其隔离，重新训练逻辑更清晰的版本。')
    add_heading(doc, '5.2 第008代实际特征分组', 2)
    add_table(doc, ['逻辑组', '代表特征', '金融解释'], [
        ('交易状态', 'flag_limit_up、flag_limit_down', '反映涨跌停约束、流动性受限和次日交易可达性'),
        ('风险状态', 'volatility_20、volatility_60及截面排名', '反映近期风险、波动聚集和风险溢价状态'),
        ('趋势与偏离', 'ret_60、ret_120、ret_240及均线偏离', '反映中期趋势延续或价格相对历史水平的偏离'),
        ('动量去近期化', 'ret_20/60/120/240_skip5及排名', '减少最近几日短期反转干扰，观察中长期趋势'),
        ('资金参与', 'vol_ratio_20、amount_ratio_20及排名', '反映成交活跃度、资金参与度和拥挤变化')
    ], [1.2, 3.0, 3.0])
    add_heading(doc, '5.3 特征验收标准', 2)
    for x in [
        '特征只能由当前行和历史行计算；滚动统计按股票和日期顺序生成。',
        '特征登记表、实际输入列和训练快照必须一致。',
        '删除或替换特征时，应在报告中说明逻辑原因和可比性影响。',
        '特征重要性报告至少同时保存split次数和gain，排名不能只依赖单一指标。'
    ]: add_bullet(doc, x)

    training_heading = add_heading(doc, '六 模型训练与迭代设计', 1)
    training_heading.paragraph_format.page_break_before = True
    add_heading(doc, '6.1 基础模型', 2)
    add_table(doc, ['参数', '当前基线值', '作用'], [
        ('objective', 'regression', '以连续标签回归为训练目标'),
        ('n_estimators', '3000上限', '给早停提供足够迭代空间，实际轮数由验证确定'),
        ('learning_rate', '0.03', '控制每棵树的步长，降低单轮更新幅度'),
        ('num_leaves', '第006代63；第008代31', '控制树的叶节点复杂度'),
        ('max_depth', '-1', '不设置额外深度上限，由叶节点数和叶样本约束复杂度'),
        ('min_child_samples', '200', '限制叶节点最小样本量，抑制过拟合'),
        ('min_split_gain', '0.0', '分裂增益的最低要求'),
        ('reg_alpha', '0.0', 'L1正则化强度'),
        ('reg_lambda', '10', 'L2正则化，降低模型参数波动'),
        ('colsample_bytree / subsample', '0.8 / 0.8', '增加树之间差异，改善泛化'),
        ('subsample_freq', '1', '每轮启用行采样'),
        ('n_jobs', '-1', '使用可用CPU并行线程'),
        ('force_col_wise', 'true', '固定列优先直方图构建方式'),
        ('random_state', '42', '保证同配置下结果可复现'),
        ('verbosity', '-1', '关闭非必要训练日志')
    ], [1.8, 1.8, 3.6])
    add_heading(doc, '6.2 代际实验规则', 2)
    for x in [
        '每代必须有唯一model_name和独立运行目录，禁止覆盖旧模型。',
        '默认一次只修改一组参数或一组特征，以便解释变化来源。',
        '新模型始终从全新的LGBMRegressor开始，init_model固定为空，不加载前代权重或预测。',
        '比较优先使用同一滚动协议的三折综合得分均值和最差折，再结合Rank IC、ICIR、年化超额、换手率判断。',
        '如果综合得分提高但2024成本后诊断明显变差，报告必须同时呈现，不能只报告有利指标。'
    ]: add_bullet(doc, x)
    add_heading(doc, '6.3 当前代际记录', 2)
    add_table(doc, ['代次', '实验主题', '关键变化', '三折综合得分均值'], [
        ('004', '原始字段滚动基准', '仅保留两个涨跌停标记', '历史对照'),
        ('005', '技术特征滚动基准', '11项金融逻辑技术特征', '历史对照'),
        ('006', '金融逻辑基线', '24项金融逻辑特征', '0.183617'),
        ('007', '叶节点正则化', '调整树复杂度控制参数', '0.180309'),
        ('008', '叶片数实验', 'num_leaves由63改为31', '0.184754')
    ], [0.7, 1.8, 3.1, 1.6])

    add_heading(doc, '七 评估指标与决策规则', 1)
    add_heading(doc, '7.1 核心指标', 2)
    add_table(doc, ['指标', '方向', '用途'], [
        ('Rank IC', '越高越好', '衡量每日横截面排序能力'),
        ('ICIR', '越高越好', '衡量Rank IC相对波动的稳定性'),
        ('年化Top1超额收益', '越高越好', '观察高分组相对基准的收益差'),
        ('平均预测换手率', '越低越好', '反映信号变化和交易成本压力'),
        ('综合得分', '越高越好', '按既定权重综合排序能力、收益和稳定性'),
        ('RMSE、MAE', '越低越好', '只作辅助回归误差指标')
    ], [1.9, 1.1, 4.2])
    add_heading(doc, '7.2 模型选择建议', 2)
    add_para(doc, '综合得分 = 0.4 × Rank IC + 0.3 × 年化Top1超额收益 + 0.3 × (1 − 平均预测换手率)。优先保留三折综合得分均值更高且最差折不明显恶化的候选。若两个候选接近，应优先考虑Rank IC与ICIR更稳定、换手率更低、金融逻辑更清楚的版本。2024模拟组合只用于诊断和风险观察，不能替代滚动外层评估。')
    add_heading(doc, '7.3 第008代结果解读', 2)
    add_para(doc, '第008代相对第006代综合得分均值提高0.001137，最差折、Rank IC和ICIR有所改善，换手率略降；但年化超额均值下降约0.37个百分点，2024成本后模拟组合IR也低于第006代。因此第008代适合作为更低叶片数的候选实验，不应被描述为所有指标全面优于第006代。')

    add_heading(doc, '八 报告与交付物', 1)
    add_table(doc, ['产物', '位置或命名', '内容'], [
        ('配置快照', 'configs/model_XXX.json及运行目录config_snapshot.json', '本代参数、时间协议、特征与确认信息'),
        ('模型文件', 'runs/模型名_时间戳/model.txt', '最终LightGBM模型'),
        ('验证预测', 'validation_predictions.csv', '滚动外层评估预测及标签'),
        ('提交预测', 'submission.csv', '测试集X主键顺序对应的预测值'),
        ('特征重要性', 'feature_importance.csv', 'gain和split两种重要性'),
        ('金融逻辑快照', 'financial_logic_snapshot.json', '当次特征逻辑和审查记录'),
        ('累计报告', '模型训练总报告.md/.txt/易读版.docx', '各代方法、参数、结果、差异和白皮书'),
        ('归因报告', 'reports/portfolio/运行名/', '2024模拟组合、成本及归因诊断')
    ], [1.4, 2.8, 3.0])
    add_para(doc, 'Word报告用于共享和打印，TXT报告采用纯文本格式，便于通过微信文件传输助手在手机上阅读。每次新增模型完成后，累计报告应在同一份文件中追加本代记录，而不是另起孤立报告。')

    add_heading(doc, '九 验收标准', 1)
    add_table(doc, ['验收项', '通过条件'], [
        ('训练流程', '预览先展示参数和金融逻辑；确认后才启动；实际拟合次数与协议一致'),
        ('模型隔离', '新运行目录独立存在；旧模型文件未被删除或覆盖；init_model为空'),
        ('数据使用', '输入仅来自题目训练集.csv和测试集_X.csv；测试标签未读取'),
        ('时间协议', '滚动折边界、标签边界和最终训练区间与配置一致'),
        ('提交文件', '行数与测试集一致；ts_code和trade_date顺序一致；pred均为有限数'),
        ('报告内容', '包含本代方法、参数、指标、前代差异、金融逻辑白皮书和2024诊断'),
        ('可复现性', '运行目录含配置、摘要、数据清单、代码快照和重要性文件'),
        ('文档质量', 'Word无截断、溢出、重叠或乱码，TXT可在手机阅读')
    ], [1.6, 5.6])

    add_heading(doc, '十 风险与应对', 1)
    add_table(doc, ['风险', '表现', '应对'], [
        ('跨年份失效', '某一折明显低于其他年份', '关注最差折和ICIR，保留稳定性更好的候选'),
        ('换手过高', '预测排序频繁变化，成本后表现下降', '调节树复杂度、正则化和特征窗口，并单独报告成本拖累'),
        ('特征失去逻辑', '重要性高但无法解释金融机制', '隔离该特征，登记清理原因，再重跑对比实验'),
        ('标签边界问题', '阶段末样本标签兑现日跨入下一阶段', '按协议剔除边界标签并记录数量'),
        ('结果不可复核', '只有最终模型，没有配置或输入记录', '强制保存配置、数据清单、代码快照和哈希'),
        ('把诊断当成绩', '将2024模拟收益当作盲测或官方分数', '报告中明确诊断用途和可观测样本范围')
    ], [1.5, 2.7, 3.0])

    add_heading(doc, '十一 后续迭代路线', 1)
    add_step(doc, 1, '先以第006代和第008代为主对照，确认下一代只改变一个明确因素，例如min_child_samples、reg_lambda或特征窗口。')
    add_step(doc, 2, '在训练前完成特征金融逻辑登记，输出待确认文本，确认后再进行7次拟合。')
    add_step(doc, 3, '把模型选择从单一均值扩展为均值、最差折、波动、换手和成本后诊断的联合判断。')
    add_step(doc, 4, '在不改变题目数据范围的前提下，完善报告中的稳定性图表、特征重要性排名和代际差异摘要。')
    add_step(doc, 5, '最终提交前锁定候选模型、保存提交文件哈希，并对交付目录做一次完整复核。')

    # Remove the empty spacer after the final table so Word does not create a blank page.
    if doc.paragraphs and not doc.paragraphs[-1].text.strip():
        final_p = doc.paragraphs[-1]._element
        final_p.getparent().remove(final_p)
    doc.core_properties.title = 'LightGBM 竞赛训练系统项目需求文档'
    doc.core_properties.subject = '赛题五多模型训练与比较系统产品需求文档'
    doc.core_properties.author = '项目组'
    doc.core_properties.comments = 'Version 1.0'
    OUT.parent.mkdir(parents=True, exist_ok=True)
    doc.save(OUT)
    print(OUT)

if __name__ == '__main__':
    build()
