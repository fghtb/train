from pathlib import Path

root = Path(r'D:\competition\train')
fp = root / 'financial_logic.py'
s = fp.read_text(encoding='utf-8')
old_mode = '    if mode == "intraday_acceptance":\n        return names + ["hl_range", "close_position", "amount_per_vol", "vol_ratio_20", "amount_ratio_20"]\n'
new_mode = old_mode + '    if mode == "price_discovery_efficiency":\n        return names + ["body_range_ratio", "upper_shadow_ratio", "lower_shadow_ratio", "path_efficiency_20", "return_autocorrelation_20"]\n'
s = s.replace(old_mode, new_mode)
old_approved = '    "hl_range", "close_position", "vol_ratio_20", "amount_ratio_20",\n'
new_approved = old_approved + '    "body_range_ratio", "upper_shadow_ratio", "lower_shadow_ratio", "path_efficiency_20", "return_autocorrelation_20",\n'
s = s.replace(old_approved, new_approved)
needle = '    elif base == "oc_return":\n'
insert = (
    '    elif base == "body_range_ratio":\n'
    '        formula, logic = "abs(close-open)/(high-low)，区间为零时缺失", "实体占日内区间的比例衡量价格方向的接受程度与多空共识。"\n'
    '        family, turnover, risk = "价格接受与共识", "形态变化可能提高换手，需实测", "极窄区间、涨跌停和跳跃行情会使比例不稳定；不保证方向收益。"\n'
    '    elif base == "upper_shadow_ratio":\n'
    '        formula, logic = "(high-max(open,close))/(high-low)，区间为零时缺失", "上影线比例衡量高位供给或买盘未能维持造成的价格拒绝。"\n'
    '        family, turnover, risk = "高位价格拒绝", "事件变化可能提高换手，需实测", "高点可能来自瞬时撮合或异常行情；价格拒绝不等于下一日必跌。"\n'
    '    elif base == "lower_shadow_ratio":\n'
    '        formula, logic = "(min(open,close)-low)/(high-low)，区间为零时缺失", "下影线比例衡量低位承接或卖压被吸收造成的价格拒绝。"\n'
    '        family, turnover, risk = "低位价格拒绝", "事件变化可能提高换手，需实测", "低点可能来自极端冲击或停牌恢复；价格承接不等于下一日必涨。"\n'
    '    elif base == "path_efficiency_20":\n'
    '        formula, logic = "abs(close[t]/close[t-20]-1)/sum(abs(ret_1),20)", "20期净位移相对总路径波动的比例衡量趋势路径的方向效率与价格发现噪声。"\n'
    '        family, turnover, risk = "价格路径效率", "较长窗口可能平滑换手，需实测", "窗口选择、缺失记录和极端行情会影响比例；高效率不等于未来延续。"\n'
    '    elif base == "return_autocorrelation_20":\n'
    '        formula, logic = "过去20期ret_1与一期滞后ret_1的滚动相关", "滚动收益自相关检验短期价格延续或反转的序列依赖，方向不预设。"\n'
    '        family, turnover, risk = "序列依赖", "状态切换时可能改变换手，需实测", "样本窗口短、异常收益和非平稳性会使相关系数不稳定。"\n'
)
if 'elif base == "body_range_ratio"' not in s:
    s = s.replace(needle, insert + needle)
s = s.replace(
    'for mode in ("technical_momentum_skip", "overnight_pressure", "impact_liquidity", "downside_drawdown", "market_relative", "intraday_acceptance"):',
    'for mode in ("technical_momentum_skip", "overnight_pressure", "impact_liquidity", "downside_drawdown", "market_relative", "intraday_acceptance", "price_discovery_efficiency"):'
)
with fp.open('w', encoding='utf-8', newline='\n') as handle:
    handle.write(s)
print('patched financial_logic.py')
