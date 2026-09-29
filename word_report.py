"""Readable Word companion to the cumulative report; no model execution."""
from pathlib import Path
import json
import re
from datetime import datetime

from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.opc.constants import RELATIONSHIP_TYPE as RT

ALIASES = {
    "model_001_raw_baseline": "001 原始字段基线",
    "model_002_technical_core": "002 技术特征模型",
    "model_003_momentum_skip": "003 动量扩展模型",
    "model_004_raw_walkforward_financial_v1": "004 制度特征基准",
    "model_005_technical_walkforward_financial_v1": "005 技术特征修订",
    "model_006_financial_logic_v1": "006 金融逻辑模型",
    "model_007_leaf_regularization_financial_v1": "007 叶节点约束实验",
    "model_008_leaf_count_financial_v1": "008 叶片数实验",
    "model_009_overnight_pressure_financial_v1": "009 隔夜信息与日内价格压力",
    "model_010_impact_liquidity_financial_v1": "010 真实波幅与流动性价格冲击",
    "model_011_downside_drawdown_financial_v1": "011 下行风险与回撤状态",
    "model_012_market_relative_financial_v1": "012 市场状态与个股相对收益",
    "model_013_intraday_acceptance_financial_v1": "013 日内价格接受与交易参与",
    "model_014_price_discovery_efficiency_financial_v1": "014 价格发现效率与买卖拒绝",
    "model_015_information_liquidity_financial_v1": "015 信息冲击与流动性承接",
    "model_016_overnight_market_state_financial_v1": "016 隔夜信息与市场状态",
    "model_017_downside_liquidity_stress_financial_v1": "017 下行风险与流动性压力",
    "model_018_market_liquidity_state_financial_v1": "018 市场状态与流动性冲击",
    "model_019_conditional_information_state_financial_v1": "019 条件信息与市场状态",
    "model_020_stress_transmission_financial_v1": "020 压力状态传导",
    "model_021_stress_core_financial_v1": "021 压力传导核心",
    "model_022_stress_recovery_financial_v1": "022 压力恢复与反转",
    "model_023_constraint_stress_financial_v1": "023 制度约束与压力传导",
    "model_024_new_mechanism_financial_v1": "024 成交参与加权价格压力",
    "model_025_new_mechanism_financial_v1": "025 市场扩散与分化转移",
    "model_026_new_mechanism_financial_v1": "026 系统性暴露与个股特有风险",
    "model_027_new_mechanism_financial_v1": "027 方向性流动性冲击不对称",
    "model_028_new_mechanism_financial_v1": "028 交易注意力集中与扩散",
    "model_029_high_score_synthesis_financial_v1": "029 压力状态与方向性流动性冲击",
    "model_030_high_score_synthesis_financial_v1": "030 风险调整后的方向性流动性冲击",
}
NAVY = "253A52"
BLUE = "EEF3F8"
GRAY = "D9D9D9"
WIDTH = 6.9


def readable(text):
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", str(text)).replace("`", "")
    if not re.search(r"[A-Za-z]:[\\/]", text):
        for source, target in sorted(ALIASES.items(), key=lambda item: -len(item[0])):
            text = text.replace(source, target)
        text = text.replace("technical_momentum_skip", "技术与动量").replace("technical", "技术特征")
        text = re.sub(r"\braw\b", "原始字段", text)
        text = text.replace("final_diagnostic", "2024诊断")
        text = re.sub(r"fold_(20\d\d)", r"\1年滚动评估", text)
    return text


def font(run, size=11, bold=False, color="000000"):
    run.font.name = "Calibri"
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.color.rgb = RGBColor.from_string(color)
    rpr = run._element.get_or_add_rPr()
    fonts = rpr.find(qn("w:rFonts"))
    if fonts is None:
        fonts = OxmlElement("w:rFonts")
        rpr.insert(0, fonts)
    fonts.set(qn("w:eastAsia"), "微软雅黑")


def paragraph(document, text, style=None):
    p = document.add_paragraph(style=style)
    run = p.add_run(readable(text))
    font(run)
    p.paragraph_format.space_after = Pt(6)
    p.paragraph_format.line_spacing = 1.2
    p.paragraph_format.widow_control = True
    return p


def heading(document, text, level=1):
    p = document.add_paragraph(style="Heading "+str(level))
    p.add_run(readable(text).replace("：", " ").replace("、", " "))
    p.paragraph_format.keep_with_next = True
    p.paragraph_format.keep_together = True
    p.paragraph_format.space_before = Pt(14 if level < 3 else 8)
    p.paragraph_format.space_after = Pt(6)
    for run in p.runs:
        font(run, {1:16, 2:13, 3:11}[level], True)
    return p


def hyperlink_paragraph(document, text, links, existing=None):
    match = re.search(r"([A-Za-z]:[\\/].+)", text)
    if not match:
        return paragraph(document, text)
    target = match.group(1).strip("`。 ）)")
    prefix = readable(text[:match.start()].strip(" -`： "))
    p = existing if existing is not None else document.add_paragraph()
    if existing is not None:
        font(p.add_run("   ·   "), 10)
    relation = p.part.relate_to(Path(target).as_uri(), RT.HYPERLINK, is_external=True)
    link = OxmlElement("w:hyperlink")
    link.set(qn("r:id"), relation)
    run = OxmlElement("w:r")
    props = OxmlElement("w:rPr")
    color = OxmlElement("w:color")
    color.set(qn("w:val"), "245A81")
    props.append(color)
    underline = OxmlElement("w:u")
    underline.set(qn("w:val"), "single")
    props.append(underline)
    run.append(props)
    content = OxmlElement("w:t")
    content.text = prefix or "相关文件"
    run.append(content)
    link.append(run)
    p._p.append(link)
    p.paragraph_format.space_after = Pt(4)
    links.append({"label": prefix or "相关文件", "path": target})
    return p


def table(document, headers, rows, statistics):
    # Split very wide numeric tables without omitting a column.
    if len(headers) > 7:
        for start in range(1, len(headers), 5):
            indices = [0] + list(range(start, min(start+5, len(headers))))
            if start > 1:
                paragraph(document, "同组指标续表")
            table(document, [headers[i] for i in indices],
                  [[row[i] if i < len(row) else "" for i in indices] for row in rows], statistics)
        return
    # Long explanatory feature tables read better as ordinary labelled paragraphs.
    if headers and headers[0] == "保留特征":
        reused = []
        definitions = statistics.setdefault("feature_definitions", {})
        for row in rows:
            if definitions.get(row[0]) == row[1:]:
                reused.append(row[0])
                continue
            definitions[row[0]] = row[1:]
            heading(document, row[0], 3)
            for key, value in zip(headers[1:], row[1:]):
                paragraph(document, key+"："+value)
        if reused:
            paragraph(document, "沿用本章已说明的特征定义与风险："+ "、".join(reused)+"。")
        statistics["feature_entries"] += len(rows)
        return
    if headers == ["特征", "剔除原因"]:
        grouped = {}
        for row in rows:
            grouped.setdefault(row[1], []).append(row[0])
        headers = ["剔除特征", "原因"]
        rows = [["、".join(names), reason] for reason, names in grouped.items()]
    if "金融假设一句话" in headers:
        index = headers.index("金融假设一句话")
        keep = [i for i in range(len(headers)) if i != index]
        table(document, [headers[i] for i in keep], [[row[i] for i in keep] for row in rows], statistics)
        for row in rows:
            paragraph(document, row[1]+"："+row[index])
        return
    n = len(headers)
    item = document.add_table(rows=1, cols=n)
    item.alignment = WD_TABLE_ALIGNMENT.CENTER
    item.autofit = False
    properties = item._tbl.tblPr
    borders = OxmlElement("w:tblBorders")
    for side in ["top", "left", "bottom", "right", "insideH", "insideV"]:
        edge = OxmlElement("w:"+side)
        for name, value in [("val","single"),("sz","4"),("color",GRAY)]:
            edge.set(qn("w:"+name), value)
        borders.append(edge)
    properties.append(borders)
    margins = OxmlElement("w:tblCellMar")
    for side in ["top", "left", "bottom", "right"]:
        element = OxmlElement("w:"+side)
        element.set(qn("w:w"), "75" if side in ["top", "bottom"] else "95")
        element.set(qn("w:type"), "dxa")
        margins.append(element)
    properties.append(margins)
    if n == 2:
        widths = [2.3, WIDTH-2.3]
    elif n == 3:
        widths = [2.65, 2.0, WIDTH-4.65]
    else:
        first = 1.65 if headers[0] in ["模型","特征","约束"] else 1.0
        widths = [first] + [(WIDTH-first)/(n-1)]*(n-1)
    for column, width in zip(item.columns, widths):
        column.width = Inches(width)
    for row_index, values in enumerate([headers]+rows):
        row = item.rows[0] if row_index == 0 else item.add_row()
        row_properties = row._tr.get_or_add_trPr()
        row_properties.append(OxmlElement("w:cantSplit"))
        if row_index == 0:
            repeat = OxmlElement("w:tblHeader")
            repeat.set(qn("w:val"), "true")
            row_properties.append(repeat)
        for col_index, (cell, value) in enumerate(zip(row.cells, values)):
            cell.width = Inches(widths[col_index])
            cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
            fill = OxmlElement("w:shd")
            fill.set(qn("w:fill"), NAVY if row_index == 0 else (BLUE if row_index % 2 else "FFFFFF"))
            cell._tc.get_or_add_tcPr().append(fill)
            p = cell.paragraphs[0]
            p.paragraph_format.space_after = Pt(2)
            p.paragraph_format.space_before = Pt(2)
            p.paragraph_format.line_spacing = 1.08
            p.paragraph_format.keep_with_next = row_index == 0
            value = readable(value)
            numeric = bool(re.fullmatch(r"[-+]?\d[\d,.%]*|-|无法计算", value))
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER if row_index == 0 or numeric else WD_ALIGN_PARAGRAPH.LEFT
            font(p.add_run(value), 10.5 if n <= 3 else 10, row_index == 0,
                 "FFFFFF" if row_index == 0 else "000000")
    spacer = document.add_paragraph()
    spacer.paragraph_format.space_after = Pt(2)
    spacer.paragraph_format.space_before = Pt(0)
    spacer.paragraph_format.line_spacing = .3
    statistics["tables"] += 1
    statistics["table_rows"] += len(rows)


def configure(document):
    section = document.sections[0]
    section.page_width, section.page_height = Inches(8.5), Inches(11)
    section.left_margin = section.right_margin = Inches(.8)
    section.top_margin = section.bottom_margin = Inches(.7)
    section.header_distance = section.footer_distance = Inches(.3)
    for name in ["Normal", "Title", "Subtitle", "Heading 1", "Heading 2", "Heading 3"]:
        style = document.styles[name]
        style.font.name = "Calibri"
        style.font.color.rgb = RGBColor(0,0,0)
        style.font.underline = False
        style._element.get_or_add_rPr().rFonts.set(qn("w:eastAsia"), "微软雅黑")
        # Remove inherited decorative rules, including Word's Title bottom rule.
        properties = style._element.get_or_add_pPr()
        for element in list(properties):
            if element.tag == qn("w:pBdr"):
                properties.remove(element)
        borders = OxmlElement("w:pBdr")
        for side in ["top","left","bottom","right","between","bar"]:
            border = OxmlElement("w:"+side)
            border.set(qn("w:val"), "nil")
            borders.append(border)
        properties.append(borders)
    document.styles["Normal"].font.size = Pt(11)
    header = section.header.paragraphs[0]
    font(header.add_run("LightGBM 模型训练与比较"), 9)
    footer = section.footer.paragraphs[0]
    footer.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    font(footer.add_run("第 "), 9)
    field = OxmlElement("w:fldSimple")
    field.set(qn("w:instr"), "PAGE")
    footer._p.append(field)
    font(footer.add_run(" 页"), 9)
    update = OxmlElement("w:updateFields")
    update.set(qn("w:val"), "true")
    document.settings._element.append(update)


def overview(document, workspace, statistics):
    title = document.add_paragraph(style="Title")
    font(title.add_run("LightGBM 多模型训练总报告"), 23, False)
    title.paragraph_format.space_after = Pt(10)
    paragraph(document, "阅读版  |  更新于 "+datetime.now().strftime("%Y年%m月%d日"))
    paragraph(document, "本报告汇总各代模型的训练方法、参数、评估结果及金融逻辑，并提供模拟组合业绩归因。先看本页结论和模型对比，再通过目录进入各代明细。")
    summaries = []
    for path in (workspace/"runs").glob("*/summary.json"):
        if path.parent.name.startswith("research_1_"):
            continue
        item = json.loads(path.read_text(encoding="utf-8"))
        if item.get("cross_validation"):
            summaries.append(item)
    summaries.sort(key=lambda item:item["model_name"])
    if summaries:
        rows = []
        for item in summaries:
            stats = item["cross_validation"]["aggregate"]
            def value(key, percent=False):
                x = stats.get(key,{}).get("mean")
                return ("无法计算" if x is None else (f"{x:.2%}" if percent else f"{x:.6f}"))
            rows.append([readable(item["model_name"]), str(item["feature_count"]),
                         value("ic_mean"), value("final_score"), value("mean_turnover",True)])
        table(document, ["模型","特征数","三折Rank IC","三折综合得分","平均换手率"], rows, statistics)
    names = {item["model_name"] for item in summaries}
    if summaries:
        comparable = [s for s in summaries
                      if s["cross_validation"]["aggregate"]["final_score"].get("mean") is not None]
        if not comparable:
            paragraph(document, "当前没有可计算的三折综合得分模型。")
            return
        best = max(comparable, key=lambda s: s["cross_validation"]["aggregate"]["final_score"]["mean"])
        score = best["cross_validation"]["aggregate"]["final_score"]["mean"]
        paragraph(document, f"当前三折平均综合得分最高的是{readable(best['model_name'])}，得分{score:.6f}。综合分领先不等于各项指标全面领先，仍需检查排序能力、跨年稳定性和并列预测。")
        if "model_028_new_mechanism_financial_v1" in names:
            paragraph(document, "第024至028代研究结论：优先保留027方向性流动性研究，026作为收益与个股风险方向的备选。025与020均有严重并列预测，低换手高分不能直接证明选股能力强；028暂列探索候选，其特征说明勘误见本轮结论。")
    paragraph(document, "阅读口径：旧固定切分与新滚动验证分别比较。2024年结果属于诊断；组合业绩须结合成本假设与有效样本覆盖率理解。")
    document.add_page_break()
    p = document.add_paragraph(style="Title")
    font(p.add_run("阅读目录"), 18, False)
    toc = document.add_paragraph()
    begin = OxmlElement("w:fldChar"); begin.set(qn("w:fldCharType"),"begin")
    instruction = OxmlElement("w:instrText")
    instruction.set(qn("xml:space"),"preserve")
    instruction.text = ' TOC \\o "1-1" \\h \\z \\u '
    separate = OxmlElement("w:fldChar"); separate.set(qn("w:fldCharType"),"separate")
    end = OxmlElement("w:fldChar"); end.set(qn("w:fldCharType"),"end")
    for element in [begin,instruction,separate]:
        run = OxmlElement("w:r"); run.append(element); toc._p.append(run)
    font(toc.add_run("训练与评估区间  ·  滚动验证方法  ·  模型对比与结论  ·  各代训练记录  ·  模拟组合归因  ·  金融逻辑白皮书  ·  并列预测诊断"),11)
    run = OxmlElement("w:r");run.append(end);toc._p.append(run)
    document.add_page_break()


def build_word_report(workspace, output_name=None):
    workspace = Path(workspace)
    source = workspace/"模型训练总报告.md"
    if not source.exists():
        return None
    lines = source.read_text(encoding="utf-8").splitlines()
    document = Document()
    configure(document)
    statistics = {"tables":0,"table_rows":0,"feature_entries":0,"source_lines":len(lines)}
    links = []
    overview(document, workspace, statistics)
    replacements = {
        "最终训练与比较区间":"训练与评估区间",
        "新增历史滚动验证协议":"滚动验证方法",
        "当前评估结论":"模型对比与结论",
        "各代模型详细记录":"各代训练记录",
        "004、005、006代结果对比与并列预测诊断":"004至006代对比与并列预测诊断",
    }
    i = 0
    while i < len(lines):
        text = lines[i].strip()
        if not text or text.startswith("# ") or text.startswith("> 本文由"):
            i += 1; continue
        if text == "## 模型横向对比":
            i += 1; continue
        if text.startswith("```"):
            code = []
            i += 1
            while i < len(lines) and not lines[i].strip().startswith("```"):
                code.append(lines[i]);i += 1
            try:
                values = json.loads("\n".join(code))
                table(document, ["参数","设置"], [[str(k),str(v)] for k,v in values.items()],statistics)
            except (ValueError, AttributeError):
                for value in code: paragraph(document,value)
            i += 1;continue
        if text.startswith("|"):
            raw = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                row = [cell.strip() for cell in lines[i].strip().strip("|").split("|")]
                if not all(re.fullmatch(r":?-{3,}:?",cell) for cell in row):
                    raw.append(row)
                i += 1
            if raw:
                table(document,raw[0],raw[1:],statistics)
            continue
        match = re.match(r"^(#{2,4})\s+(.+)$",text)
        if match:
            level = min(len(match.group(1))-1,3)
            name = replacements.get(match.group(2),match.group(2))
            p = heading(document,name,level)
            if level == 1 and name in ["各代训练记录","模拟组合业绩归因报告","金融逻辑特征白皮书","004至006代对比与并列预测诊断"]:
                p.paragraph_format.page_break_before = True
        elif re.fullmatch(r"\*\*[^*]+\*\*",text):
            heading(document,text.strip("*"),3)
        else:
            text = re.sub(r"^[-*]\s+","",text)
            if re.search(r"[A-Za-z]:[\\/]",text):
                link_paragraph = hyperlink_paragraph(document,text,links)
                # Adjacent file references share one compact line instead of
                # generating an otherwise almost empty final page.
                while True:
                    cursor = i+1
                    while cursor < len(lines) and not lines[cursor].strip():
                        cursor += 1
                    if cursor >= len(lines):
                        break
                    following = re.sub(r"^[-*]\s+","",lines[cursor].strip())
                    if following.startswith(("#","|","**")) or not re.search(r"[A-Za-z]:[\\/]",following):
                        break
                    hyperlink_paragraph(document,following,links,link_paragraph)
                    i = cursor
            else:
                paragraph(document,text)
        i += 1
    output = workspace/(output_name or "模型训练总报告_易读版.docx")
    document.core_properties.title = "LightGBM 多模型训练总报告"
    document.core_properties.subject = "训练结果 模型比较 金融逻辑 模拟组合归因"
    document.save(output)
    reopened = Document(output)
    content = "\n".join([p.text for p in reopened.paragraphs]+
                        [cell.text for t in reopened.tables for row in t.rows for cell in row.cells])
    for forbidden in ["约束检查","source_paths_whitelisted","submission_alignment_checked","后续迭代约束","登记指纹"]:
        if forbidden in content:
            raise ValueError("Reader report contains constraint-check content: "+forbidden)
    metadata = workspace/"reports/word_readability"
    metadata.mkdir(parents=True,exist_ok=True)
    with (metadata/"layout_manifest.json").open("w",encoding="utf-8",newline="\n") as handle:
        json.dump({**statistics,"output":str(output),"source":str(source),"hyperlinks":links,
                   "sections":[p.text for p in reopened.paragraphs if p.style.name=="Heading 1"]},
                  handle,ensure_ascii=False,indent=2)
    return output


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace",default=Path(__file__).parent)
    args = parser.parse_args()
    print(build_word_report(args.workspace))
