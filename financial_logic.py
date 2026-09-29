"""Prospective financial-hypothesis registry and mandatory training feature gate.

An economic hypothesis is not a causal finding or a promise of lower turnover.
The original model-003 code is preserved in the audit transaction; explanations
added here do not retroactively count as pre-training evidence for old models.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

POLICY_VERSION = "FINANCIAL_LOGIC_FIRST_V1"
RAW = ["open", "high", "low", "close", "vol", "amount", "flag_limit_up", "flag_limit_down"]
HIGH_UNDOCUMENTED = {
    "ret_1", "ret_1_rank", "oc_return", "ma_gap_3",
    "ret_3", "ret_20", "ret_10",
}
APPROVED = [
    "flag_limit_up", "flag_limit_down",
    "volatility_20", "ma_gap_20", "ret_60", "volatility_60", "ma_gap_60",
    "vol_ratio_20", "amount_ratio_20", "ret_120", "ret_240",
    "ret_20_skip5", "ret_60_skip5", "ret_120_skip5", "ret_240_skip5",
    "volatility_20_rank", "amount_ratio_20_rank",
    "ret_60_rank", "ret_120_rank", "ret_240_rank",
    "ret_20_skip5_rank", "ret_60_skip5_rank", "ret_120_skip5_rank", "ret_240_skip5_rank",
    "gap_1", "oc_return", "gap_1_rank", "oc_return_rank",
    "true_range_pct", "atr_20", "range_shock", "amihud_20", "amihud_20_rank",
    "downside_vol_20", "downside_freq_20", "drawdown_20",
    "downside_vol_60", "downside_freq_60", "drawdown_60",
    "downside_vol_20_rank", "downside_freq_20_rank", "drawdown_20_rank",
    "downside_vol_60_rank", "downside_freq_60_rank", "drawdown_60_rank",
    "market_median_return", "market_breadth", "market_dispersion",
    "residual_return_1", "residual_momentum_20", "market_relative_volatility",
    "hl_range", "close_position", "vol_ratio_20", "amount_ratio_20",
    "body_range_ratio", "upper_shadow_ratio", "lower_shadow_ratio", "path_efficiency_20", "return_autocorrelation_20",
    "gap_market_alignment", "shock_market_alignment", "stress_breadth_alignment", "liquidity_stress_gap",
    "recovery_from_low_20", "pressure_recovery_alignment", "limit_stress_alignment", "stress_dispersion_alignment",
    "amount_weighted_return_5", "amount_weighted_return_20", "price_volume_concordance_20", "pressure_horizon_spread",
    "market_breadth_mean_5", "market_breadth_mean_20", "market_breadth_change_5", "market_dispersion_mean_20", "breadth_dispersion_alignment",
    "market_beta_60", "market_beta_change_20", "idiosyncratic_volatility_20", "idiosyncratic_momentum_20",
    "upside_amihud_20", "downside_amihud_20", "impact_asymmetry_20", "downside_impact_share_20",
    "amount_share_rank", "market_attention_hhi", "attention_activity_breadth", "attention_return_alignment",
    "stress_breadth_alignment", "liquidity_stress_gap", "market_beta_60", "market_beta_change_20",
    "idiosyncratic_volatility_20", "idiosyncratic_momentum_20", "downside_amihud_20", "impact_asymmetry_20",
]


def feature_catalog(mode):
    research_bases = {
        "research_1_031": ["asymmetric_impact"],
        "research_1_032": ["stress_asymmetric_impact", "asymmetric_impact", "market_relative"],
        "research_1_033": ["risk_adjusted_impact", "asymmetric_impact"],
        "research_1_034": ["stress_transmission"],
        "research_1_035": ["breadth_transition", "asymmetric_impact"],
        "research_1_036": ["stress_asymmetric_impact", "asymmetric_impact"],
        "research_1_037": ["risk_adjusted_impact", "asymmetric_impact"],
        "research_1_038": ["breadth_transition", "risk_adjusted_impact", "asymmetric_impact"],
    }
    if mode in research_bases:
        merged = []
        for base_mode in research_bases[mode]:
            for name in feature_catalog(base_mode):
                if name not in merged:
                    merged.append(name)
        return merged
    names = list(RAW)
    if mode == "raw":
        return names
    if mode == "overnight_pressure":
        return names + ["gap_1", "oc_return", "gap_1_rank", "oc_return_rank"]
    if mode == "impact_liquidity":
        return names + ["true_range_pct", "atr_20", "range_shock", "amihud_20", "amihud_20_rank"]
    if mode == "information_liquidity":
        return names + ["gap_1", "oc_return", "true_range_pct", "atr_20", "range_shock", "amihud_20", "amihud_20_rank"]
    if mode == "overnight_market_state":
        return names + ["gap_1", "oc_return", "market_median_return", "market_breadth", "market_dispersion", "residual_return_1", "residual_momentum_20", "market_relative_volatility"]
    if mode == "downside_liquidity_stress":
        return names + ["true_range_pct", "atr_20", "range_shock", "amihud_20", "amihud_20_rank", "downside_vol_20", "downside_freq_20", "drawdown_20"]
    if mode == "market_liquidity_state":
        return names + ["market_median_return", "market_breadth", "market_dispersion", "market_relative_volatility", "true_range_pct", "atr_20", "range_shock", "amihud_20", "amihud_20_rank"]
    if mode == "conditional_information_state":
        return names + ["gap_1", "oc_return", "market_median_return", "market_breadth", "market_dispersion", "residual_return_1", "residual_momentum_20", "market_relative_volatility", "gap_market_alignment", "shock_market_alignment"]
    if mode == "stress_transmission":
        return names + ["market_median_return", "market_breadth", "market_dispersion", "true_range_pct", "atr_20", "range_shock", "amihud_20", "amihud_20_rank", "downside_vol_20", "downside_freq_20", "drawdown_20", "stress_breadth_alignment", "liquidity_stress_gap"]
    if mode == "stress_core":
        return names + ["market_breadth", "market_dispersion", "true_range_pct", "range_shock", "amihud_20", "amihud_20_rank", "downside_freq_20", "stress_breadth_alignment", "liquidity_stress_gap"]
    if mode == "stress_recovery":
        return names + ["market_median_return", "market_breadth", "market_dispersion", "true_range_pct", "range_shock", "amihud_20", "downside_vol_20", "downside_freq_20", "drawdown_20", "recovery_from_low_20", "pressure_recovery_alignment"]
    if mode == "constraint_stress":
        return names + ["market_breadth", "market_dispersion", "true_range_pct", "range_shock", "amihud_20", "downside_freq_20", "limit_stress_alignment", "stress_dispersion_alignment"]
    if mode == "value_pressure":
        return names + ["amount_weighted_return_5", "amount_weighted_return_20", "price_volume_concordance_20", "pressure_horizon_spread"]
    if mode == "breadth_transition":
        return names + ["market_breadth_mean_5", "market_breadth_mean_20", "market_breadth_change_5", "market_dispersion_mean_20", "breadth_dispersion_alignment"]
    if mode == "beta_idiosyncratic":
        return names + ["market_beta_60", "market_beta_change_20", "idiosyncratic_volatility_20", "idiosyncratic_momentum_20"]
    if mode == "asymmetric_impact":
        return names + ["upside_amihud_20", "downside_amihud_20", "impact_asymmetry_20", "downside_impact_share_20"]
    if mode == "attention_concentration":
        return names + ["amount_share_rank", "market_attention_hhi", "attention_activity_breadth", "attention_return_alignment"]
    if mode == "stress_asymmetric_impact":
        return names + ["market_breadth", "market_dispersion", "true_range_pct", "range_shock", "amihud_20", "downside_freq_20", "downside_amihud_20", "impact_asymmetry_20", "stress_breadth_alignment", "liquidity_stress_gap"]
    if mode == "risk_adjusted_impact":
        return names + ["market_median_return", "market_breadth", "market_dispersion", "true_range_pct", "range_shock", "amihud_20", "market_beta_60", "market_beta_change_20", "idiosyncratic_volatility_20", "idiosyncratic_momentum_20", "downside_amihud_20", "impact_asymmetry_20"]
    if mode == "downside_drawdown":
        base = []
        for window in [20, 60]:
            base += [f"downside_vol_{window}", f"downside_freq_{window}", f"drawdown_{window}"]
        return names + base + [name + "_rank" for name in base]
    if mode == "market_relative":
        return names + ["market_median_return", "market_breadth", "market_dispersion", "residual_return_1", "residual_momentum_20", "market_relative_volatility"]
    if mode == "intraday_acceptance":
        return names + ["hl_range", "close_position", "amount_per_vol", "vol_ratio_20", "amount_ratio_20"]
    if mode == "price_discovery_efficiency":
        return names + ["body_range_ratio", "upper_shadow_ratio", "lower_shadow_ratio", "path_efficiency_20", "return_autocorrelation_20"]
    if mode not in {"technical", "technical_momentum_skip"}:
        raise ValueError("Unknown feature mode")
    names += ["oc_return", "hl_range", "close_position", "amount_per_vol", "ret_1"]
    for window in [3, 5, 10, 20, 60]:
        names += [f"ret_{window}", f"volatility_{window}", f"ma_gap_{window}"]
    for window in [5, 20]:
        names += [f"vol_ratio_{window}", f"amount_ratio_{window}"]
    if mode == "technical_momentum_skip":
        names += ["ret_120", "ret_240"]
        names += [f"ret_{window}_skip5" for window in [20, 60, 120, 240]]
    ranks = ["ret_1", "ret_5", "ret_20", "volatility_20", "amount_ratio_20"]
    if mode == "technical_momentum_skip":
        ranks += ["ret_60", "ret_120", "ret_240",
                  "ret_20_skip5", "ret_60_skip5", "ret_120_skip5", "ret_240_skip5"]
    return names + [name + "_rank" for name in ranks]


def _definition(name):
    base = name.removesuffix("_rank")
    ranked = name != base
    if base in ["flag_limit_up", "flag_limit_down"]:
        # 金融逻辑：涨跌停制度可能导致订单失衡和价格发现延迟；不预设次日必涨/必跌。
        direction = "涨停" if base.endswith("_up") else "跌停"
        formula = f"题目提供的当日{direction}标记"
        logic = f"{direction}标记刻画交易约束与订单失衡，可能影响下一日价格发现和反转风险。"
        family, turnover = "交易制度与流动性约束", "方向不确定，事件跳变可能增加交易"
        risk = "不是次日溢价保证；事件延续和反转并存，不能据此假设能按涨跌停价成交。"
    elif base.startswith("ret_") and base.endswith("_skip5"):
        # 金融逻辑：中期相对强弱可能反映信息吸收缓慢；跳过5个记录交易日以隔离近期反转。
        window = int(base.split("_")[1])
        formula = f"close[t-5]/close[t-{window + 5}]-1（按股票的历史记录顺序）"
        logic = f"{window}期且跳过最近5期的历史收益检验中期趋势延续，同时隔离近期价格反转。"
        family, turnover = "跳过近期的动量假设", "相对短期收益可能更平滑，仍需实测"
        risk = "反转行情或趋势拥挤时可能失效；5期是本项目假设，不等于原论文方案的精确复现。"
    elif base.startswith("ret_"):
        window = int(base.split("_")[1])
        formula = f"close[t]/close[t-{window}]-1（按股票的历史记录顺序）"
        if window <= 10:
            logic = f"{window}期收益可能反映短期订单冲击后的反转，也可能反映消息延续，方向不能先验确定。"
            family, turnover = "短期反转或延续的待检验假设", "可能增加短期交易，不能仅凭名称下结论"
        else:
            # 金融逻辑：较长历史相对强弱用于检验信息扩散和趋势延续，不作为因果证据。
            logic = f"{window}期历史收益用于检验信息逐步吸收引起的趋势延续，也需要防范趋势反转。"
            family, turnover = "趋势延续假设", "相对短周期可能较低，方向需实测"
        risk = "复权与缺失价格口径、趋势反转和观察窗口选择都可能影响结果。"
    elif base.startswith("volatility_"):
        # 金融逻辑：历史波动率刻画风险状态；次日风险补偿或风险回避方向不作保证。
        window = int(base.split("_")[1])
        formula = f"过去{window}期ret_1的样本标准差，含当日，最少{window}期"
        logic = f"{window}期历史波动率刻画收益不确定性，用于检验风险补偿或风险回避下的定价差异。"
        family, turnover = "风险状态", "较长窗口可能更稳定，冲击时仍会跳变"
        risk = "高波动不必然带来高收益；短窗口估计噪声较大，不能等同风险溢价已成立。"
    elif base.startswith("ma_gap_"):
        # 金融逻辑：价格相对自身均线的偏离表示趋势强度或暂时超调，两种机制需竞争检验。
        window = int(base.split("_")[2])
        formula = f"close[t]/过去{window}期close均值-1，含当日"
        logic = f"{window}期均线偏离衡量价格相对自身历史水平的趋势强度或过度偏离。"
        family, turnover = "趋势与均值回归的竞争假设", "较长窗口可能减小噪声，换手方向不确定"
        risk = "趋势与反转对收益方向的预测相反，不能事后选择最有利的解释。"
    elif base.startswith(("vol_ratio_", "amount_ratio_")):
        # 金融逻辑：自身历史归一化的成交活跃度衡量参与变化，不等同净流入或特定资金身份。
        window = int(base.split("_")[-1])
        field = "vol" if base.startswith("vol_") else "amount"
        label = "成交量" if field == "vol" else "成交额"
        formula = f"{field}[t]/过去{window}期{field}均值，含当日"
        logic = f"{window}期相对{label}反映相对自身常态的交易参与变化，可检验信息到达与流动性状态。"
        family, turnover = "交易参与和流动性状态", "异常成交可能增加交易，不能预先保证降低"
        risk = "放量不等于净流入，也不能确认主力进场；同时可能是恐慌交易。"
    elif base == "gap_1":
        formula, logic = "open[t]/close[t-1]-1（按股票历史记录顺序）", "隔夜跳空反映非交易时段信息到达和开盘重新定价，方向可能延续也可能反转。"
        family, turnover, risk = "隔夜信息与开盘重定价", "单日信号可能增加换手，需实测", "开盘撮合、停牌和公告集中发布会造成极端值；不能解释为资金净流入。"
    elif base == "true_range_pct":
        formula, logic = "max(high-low, abs(high-prev_close), abs(low-prev_close))/prev_close", "真实波幅同时纳入隔夜跳跃和日内价格区间，刻画单日价格冲击与风险状态。"
        family, turnover, risk = "真实波幅与价格冲击", "异常冲击可能提高换手，需实测", "极端行情、停牌和缺失前收盘会放大噪声；高波幅不保证次日收益方向。"
    elif base == "atr_20":
        formula, logic = "过去20期true_range_pct均值（含当日）", "20期平均真实波幅刻画个股常态风险和流动性压力，区别于只看收盘收益的波动率。"
        family, turnover, risk = "常态风险与流动性状态", "较单日冲击更平滑，换手方向不确定", "均值仍会受极端日影响；风险状态不等于风险溢价已成立。"
    elif base == "range_shock":
        formula, logic = "true_range_pct/atr_20", "当前真实波幅相对自身常态的倍数识别异常价格冲击和流动性状态切换。"
        family, turnover, risk = "异常冲击状态", "状态跳变可能增加换手，需实测", "分母接近零或极端行情会产生异常值；不能解释为资金方向。"
    elif base == "amihud_20":
        formula, logic = "过去20期mean(abs(ret_1)/amount)（amount为零时缺失）", "Amihud近似单位成交额对应的价格变化，刻画持续的价格冲击和流动性恶化。"
        family, turnover, risk = "价格冲击与流动性", "20期均值相对平滑，方向需验证", "成交额量纲、零成交和极端日会影响数值；不能据此识别买卖方资金身份。"
    elif base.startswith("downside_vol_"):
        window = int(base.split("_")[-1])
        formula, logic = f"sqrt(mean(min(ret_1,0)^2,{window}))", f"{window}期下行半方差只统计负收益，区分损失方向风险和对称波动。"
        family, turnover, risk = "下行风险状态", "较平滑但风险切换时可能跳变", "高下行风险不保证次日反弹或高收益；极端下跌日会主导窗口。"
    elif base.startswith("downside_freq_"):
        window = int(base.split("_")[-1])
        formula, logic = f"mean(ret_1<0,{window})", f"{window}期负收益频率刻画持续卖压和损失出现的密度。"
        family, turnover, risk = "持续卖压状态", "状态变化时可能提高换手", "负收益频率受市场环境和停牌缺失影响；不等于未来收益方向。"
    elif base.startswith("drawdown_"):
        window = int(base.split("_")[-1])
        formula, logic = f"close/rolling_max(close,{window})-1", f"{window}期滚动回撤刻画相对近期高点的亏损状态，检验被套、去杠杆或风险预算收缩。"
        family, turnover, risk = "回撤与被套状态", "回撤修复或扩大时可能换手", "回撤深度受窗口和历史高点定义影响；深回撤不保证反弹。"
    elif base == "market_median_return":
        formula, logic = "同一交易日所有可用股票ret_1的横截面中位数", "市场中位数收益刻画当日共同市场方向，避免把个股信号误当成市场行情。"
        family, turnover, risk = "市场状态", "共同状态较平滑，换手方向不确定", "样本股票组成变化会改变统计量；不是正式市场指数。"
    elif base == "market_breadth":
        formula, logic = "同一交易日ret_1>0的股票比例", "上涨广度刻画市场参与范围和行情扩散程度。"
        family, turnover, risk = "市场状态", "状态变化时可能提高换手", "缺失价格和停牌会影响分母；广度不代表未来方向。"
    elif base == "market_dispersion":
        formula, logic = "同一交易日ret_1的横截面标准差", "市场分化度刻画个股差异和选股空间。"
        family, turnover, risk = "市场分化状态", "分化扩大时排序可能变化更快", "极端个股会放大统计量；不等于可交易的风险溢价。"
    elif base == "residual_return_1":
        formula, logic = "个股ret_1-同日market_median_return", "剔除共同市场波动后观察个股相对表现。"
        family, turnover, risk = "相对收益", "相对表现变化可能提高换手", "市场代理误差和样本组成变化会影响残差。"
    elif base == "residual_momentum_20":
        formula, logic = "个股20期累计收益-市场中位数收益20期滚动和", "相对市场的中期趋势检验个股信息扩散，而不是简单追踪大盘。"
        family, turnover, risk = "相对趋势", "20期信号相对平滑，仍需实测", "共同收益估计误差、趋势反转和窗口选择可能导致失效。"
    elif base == "market_relative_volatility":
        formula, logic = "个股20期收益标准差/当日市场分化度", "个股风险相对市场分化度衡量风险暴露变化。"
        family, turnover, risk = "相对风险状态", "风险状态跳变可能提高换手", "分母接近零会放大数值；不是完整的CAPM beta。"
    elif base == "body_range_ratio":
        formula, logic = "abs(close-open)/(high-low)，区间为零时缺失", "实体占日内区间的比例衡量价格方向的接受程度与多空共识。"
        family, turnover, risk = "价格接受与共识", "形态变化可能提高换手，需实测", "极窄区间、涨跌停和跳跃行情会使比例不稳定；不保证方向收益。"
    elif base == "upper_shadow_ratio":
        formula, logic = "(high-max(open,close))/(high-low)，区间为零时缺失", "上影线比例衡量高位供给或买盘未能维持造成的价格拒绝。"
        family, turnover, risk = "高位价格拒绝", "事件变化可能提高换手，需实测", "高点可能来自瞬时撮合或异常行情；价格拒绝不等于下一日必跌。"
    elif base == "lower_shadow_ratio":
        formula, logic = "(min(open,close)-low)/(high-low)，区间为零时缺失", "下影线比例衡量低位承接或卖压被吸收造成的价格拒绝。"
        family, turnover, risk = "低位价格拒绝", "事件变化可能提高换手，需实测", "低点可能来自极端冲击或停牌恢复；价格承接不等于下一日必涨。"
    elif base == "gap_market_alignment":
        formula, logic = "gap_1×(2×market_breadth-1)", "隔夜跳空方向与市场上涨广度的一致程度，检验个股信息冲击在广泛行情中的条件效应。"
        family, turnover, risk = "条件信息冲击", "条件信号可能减少无效交易，需实测", "市场广度是样本代理；乘积放大极端跳空，不能解释为因果交互。"
    elif base == "shock_market_alignment":
        formula, logic = "range_shock×market_dispersion", "个股异常价格冲击与市场分化程度的乘积，检验冲击在选股空间扩大时的定价作用。"
        family, turnover, risk = "条件流动性冲击", "状态跳变可能提高换手，需实测", "分化度和冲击均为估计量，极端日会放大乘积。"
    elif base == "stress_breadth_alignment":
        formula, logic = "downside_freq_20×(1-market_breadth)", "个股持续负收益与市场下跌广度的共同程度，刻画系统性压力传导。"
        family, turnover, risk = "压力传导", "压力状态切换可能提高换手，需实测", "市场广度的横截面组成会变化，乘积不代表真实资金流。"
    elif base == "liquidity_stress_gap":
        formula, logic = "amihud_20×(1-market_breadth)", "个股流动性价格冲击在市场下跌广度扩大时的条件状态，检验流动性压力传导。"
        family, turnover, risk = "流动性压力传导", "压力状态可能提高换手，需实测", "Amihud量纲和市场广度误差会共同影响数值。"
    elif base == "recovery_from_low_20":
        formula, logic = "close/rolling_min(close,20)-1", "价格相对20期低点的恢复程度，刻画压力后的承接和反转修复。"
        family, turnover, risk = "压力恢复", "恢复状态变化可能提高换手，需实测", "窗口高低点和停牌记录会影响恢复比例；不保证反转成功。"
    elif base == "pressure_recovery_alignment":
        formula, logic = "recovery_from_low_20×(1-market_breadth)", "个股从低点恢复与市场下跌广度的共同状态，检验逆势承接或压力中的局部修复。"
        family, turnover, risk = "条件恢复", "条件状态变化可能提高换手，需实测", "交互项放大极端行情，不代表因果承接。"
    elif base == "limit_stress_alignment":
        formula, logic = "(flag_limit_up-flag_limit_down)×(1-market_breadth)", "涨跌停约束方向与市场下跌广度的共同程度，刻画制度约束下的压力传导。"
        family, turnover, risk = "制度压力传导", "事件状态跳变可能提高换手", "涨跌停标记只表示制度状态，不保证可成交或未来方向。"
    elif base == "stress_dispersion_alignment":
        formula, logic = "range_shock×market_dispersion", "异常价格冲击与市场分化的共同状态，刻画压力是否伴随选股空间扩大。"
        family, turnover, risk = "分化压力状态", "状态跳变可能提高换手", "分化度和波幅估计误差会放大交互项。"
    elif base.startswith("amount_weighted_return_"):
        window = int(base.rsplit("_", 1)[1])
        formula = f"sum(ret_1×amount,{window})/sum(amount,{window})"
        logic = f"{window}期成交额加权收益刻画高成交额日对近期价格变化的贡献；它是交易活跃度加权，不代表净买入资金。"
        family, turnover, risk = "成交参与与價格压力", "成交结构变化可能带来换手，方向需实测", "成交额权重受规模和事件日影响，不识别买卖方，也不保证未来延续。"
    elif base == "price_volume_concordance_20":
        formula, logic = "20期mean(sign(ret_1)×sign(Δlog(1+amount)))", "价格方向与成交额变化方向的一致频率，检验交易参与增强时价格信息是否被确认。"
        family, turnover, risk = "价格与参与确认", "状态变化时可能增加交易", "成交额变化不等于订单方向；横盘噪声会降低稳定性。"
    elif base == "pressure_horizon_spread":
        formula, logic = "amount_weighted_return_5-amount_weighted_return_20", "短期与中期成交额加权价格压力之差，检验交易参与信号的加速或衰减。"
        family, turnover, risk = "交易参与变化", "变化信号可能更敏感，换手需实测", "两个重叠窗口相关，差值易受短期极端成交影响。"
    elif base.startswith("market_breadth_mean_"):
        window = int(base.rsplit("_", 1)[1])
        formula, logic = f"过去{window}个交易日market_breadth均值，含当日", f"{window}日上涨广度均值衡量市场参与扩散的持续程度。"
        family, turnover, risk = "市场参与扩散", "平滑状态可能降低噪声，方向需检验", "横截面股票构成、停牌和上市变化会影响广度。"
    elif base == "market_breadth_change_5":
        formula, logic = "market_breadth[t]-market_breadth[t-5]", "五个交易日上涨广度变化刻画市场参与扩散的加速或收缩。"
        family, turnover, risk = "市场状态转移", "转折可能提高组合调整", "广度突变也可能是短期冲击，不能当作市场择时因果证据。"
    elif base == "market_dispersion_mean_20":
        formula, logic = "过去20个交易日market_dispersion均值，含当日", "市场横截面收益分化的常态水平反映个股收益差异和选股空间。"
        family, turnover, risk = "市场分化状态", "分化变化可能影响选股排序", "极端股票和样本组成会影响分化统计。"
    elif base == "breadth_dispersion_alignment":
        formula, logic = "market_breadth_change_5×market_dispersion_mean_20", "市场参与扩散变化与常态分化的交互，检验市场转折是否伴随更大的横截面选股差异。"
        family, turnover, risk = "市场扩散与选股空间", "状态转移可能带来换手", "两个市场代理变量的估计误差会在乘积中叠加。"
    elif base == "market_beta_60":
        formula, logic = "60期cov(stock_ret,market_median_return)/var(market_median_return)", "滚动市场敏感度刻画个股共同市场风险暴露，用于区分系统性与个股特有收益。"
        family, turnover, risk = "系统性风险暴露", "暴露变化较慢，换手需实测", "市场中位数只是题目样本代理，不是正式指数；估计窗口和结构变化影响beta。"
    elif base == "market_beta_change_20":
        formula, logic = "market_beta_60[t]-market_beta_60[t-20]", "市场敏感度的近期变化检验个股系统性风险暴露是否正在迁移。"
        family, turnover, risk = "风险暴露迁移", "暴露转变时可能增加換手", "滚动估计误差会叠加，不能解释为稳定结构性beta变化。"
    elif base == "idiosyncratic_volatility_20":
        formula, logic = "20期std(stock_ret-market_beta_60×market_median_return)", "beta调整后的残差波动刻画无法由共同市场变化解释的个股风险。"
        family, turnover, risk = "个股特有风险", "风险重估可能改变排序", "beta和市场代理误差都进入残差；高特有波动不保证高收益。"
    elif base == "idiosyncratic_momentum_20":
        formula, logic = "20期sum(stock_ret-market_beta_60×market_median_return)", "beta调整后的个股残差累计变化检验个股特有信息的扩散或反转。"
        family, turnover, risk = "个股特有信息", "信号方向变化可能提高换手", "市场beta估计误差会污染残差，趋势延续与反转方向需实证区分。"
    elif base.startswith("upside_amihud_") or base.startswith("downside_amihud_"):
        direction = "正收益" if base.startswith("upside") else "负收益绝对值"
        formula = f"过去20期mean({direction}/amount)，仅统计对应方向日"
        logic = f"上涨和下跌时分别估算单位成交额价格冲击，区分买盘推动与卖压冲击的流动性不对称。"
        family, turnover, risk = "方向性流动性价格冲击", "流动性状态转换可能改变换手", "成交额不识别交易对手；稀疏方向样本和极端日会使估计不稳定。"
    elif base == "impact_asymmetry_20":
        formula, logic = "(downside_amihud-upside_amihud)/(downside_amihud+upside_amihud)", "方向性价格冲击的有界差异刻画卖压与买盘推动的相对流动性脆弱度。"
        family, turnover, risk = "流动性不对称", "失衡变化可能触发调整", "分母很小时仍可能受微小冲击主导；不能将冲击方向等同未来收益方向。"
    elif base == "downside_impact_share_20":
        formula, logic = "downside_amihud/(downside_amihud+upside_amihud)", "下行价格冲击占双向冲击总量的比重衡量流动性恶化是否偏向卖出方向。"
        family, turnover, risk = "下行流动性脆弱度", "卖压结构变化可能提高换手", "仅统计观测到的方向样本，成交额和极端行情会影响比值。"
    elif base == "amount_share":
        formula, logic = "amount/同日题目样本内正成交额总和", "相对成交额份额衡量股票在当日样本交易注意力中的位置，不把成交额解释为净买入或价格水平信号。"
        family, turnover, risk = "相对注意力与拥挤", "注意力迁移可能提高换手", "横截面范围和极端大额成交会改变排名；注意力不是净资金流。"
    elif base == "market_attention_hhi":
        formula, logic = "同日sum((stock_amount/market_total_amount)^2)", "成交额份额赫芬达尔指数刻画市场交易注意力集中或分散程度。"
        family, turnover, risk = "市场注意力集中度", "集中度变化可能改变排序环境", "样本覆盖和大额异常成交会明显影响指数；不等同真实市场全量集中度。"
    elif base == "attention_activity_breadth":
        formula, logic = "同日个股amount_ratio_20>1的比例", "高于自身成交额常态的股票占比刻画交易关注是否广泛扩散。"
        family, turnover, risk = "交易注意力扩散", "广泛活动变化可能改变换手", "滚动均值和样本组成影响分母；放量不等于资金净流入。"
    elif base == "attention_return_alignment":
        formula, logic = "ret_1×amount_share_rank", "将个股当日价格方向与相对成交额注意力结合，检验高参与度下的价格变化是否更具信息含量。"
        family, turnover, risk = "注意力与价格确认", "高注意力信号可能更易跳变", "成交额排名不代表主动买卖方向，价格变化可由多种交易需求驱动。"
    elif base == "path_efficiency_20":
        formula, logic = "abs(close[t]/close[t-20]-1)/sum(abs(ret_1),20)", "20期净位移相对总路径波动的比例衡量趋势路径的方向效率与价格发现噪声。"
        family, turnover, risk = "价格路径效率", "较长窗口可能平滑换手，需实测", "窗口选择、缺失记录和极端行情会影响比例；高效率不等于未来延续。"
    elif base == "return_autocorrelation_20":
        formula, logic = "过去20期ret_1与一期滞后ret_1的滚动相关", "滚动收益自相关检验短期价格延续或反转的序列依赖，方向不预设。"
        family, turnover, risk = "序列依赖", "状态切换时可能改变换手，需实测", "样本窗口短、异常收益和非平稳性会使相关系数不稳定。"
    elif base == "oc_return":
        formula, logic = "close/open-1", "开盘到收盘收益刻画日内价格压力，可能延续，也可能在次日反转。"
        family, turnover, risk = "日内价格压力假设", "可能偏高，需验证", "信号只能在收盘后获得；缺少原始事前金融说明。"
    elif base == "hl_range":
        formula, logic = "(high-low)/close", "日内高低价幅度刻画当日风险与信息冲击强度。"
        family, turnover, risk = "日内风险状态", "可能偏高，需验证", "不能把振幅直接认定为风险补偿；缺少原始事前金融说明。"
    elif base == "close_position":
        formula, logic = "(close-low)/(high-low)", "收盘价在日内区间的位置可能刻画收盘前买卖压力及其后续延续或反转。"
        family, turnover, risk = "日内价格压力假设", "可能偏高，需验证", "极窄区间和跳跃行情会使该比值不稳定；缺少原始事前金融说明。"
    elif base == "amount_per_vol":
        formula, logic = "amount/vol", "成交额除成交量在单位一致时近似成交均价，本身不能构成明确的跨股票收益预测机制。"
        family, turnover, risk = "价格水平代理待审查", "无法可靠预判", "单位与缩放需要核实，可能重复价格水平信息。"
    else:
        formula = f"题目当日原始字段{base}"
        logic = ("原始成交规模可以描述活跃程度，但未归一化时混合股票规模和量纲差异。"
                 if base in {"vol", "amount"} else
                 "绝对价格水平可描述报价，但当前没有预先记录的跨股票下一日收益定价机制。")
        family, turnover, risk = "原始水平变量待审查", "无法可靠预判", "保留为派生特征的源字段；本版本不作为模型输入。"
    if ranked:
        formula = f"同日提供样本内对[{formula}]作百分位排名"
        logic += "横截面排名只表达相对位置，不额外创造因果机制。"
        risk += "样本组成变化会改变排名，源变量被隔离时其排名版本也不得绕过限制。"
    return {"formula": formula, "logic": logic, "family": family,
            "expected_turnover": turnover, "failure_risk": risk,
            "available_at": "当日收盘数据形成后；仅用当日和历史记录"}


def registry():
    result = {}
    all_names = []
    for mode in ("technical_momentum_skip", "overnight_pressure", "impact_liquidity", "information_liquidity", "overnight_market_state", "downside_liquidity_stress", "market_liquidity_state", "conditional_information_state", "stress_transmission", "stress_core", "stress_recovery", "constraint_stress", "value_pressure", "breadth_transition", "beta_idiosyncratic", "asymmetric_impact", "attention_concentration", "stress_asymmetric_impact", "risk_adjusted_impact", "downside_drawdown", "market_relative", "intraday_acceptance", "price_discovery_efficiency"):
        for name in feature_catalog(mode):
            if name not in all_names:
                all_names.append(name)
    for name in all_names:
        item = _definition(name)
        approved = name in APPROVED
        documented = (name in ["ret_120", "ret_240"] or "_skip5" in name)
        if name in HIGH_UNDOCUMENTED:
            reason = "Top10高重要性且原始特征代码缺少明确金融假设说明；先隔离，不用事后补写解释追认。"
        elif name == "ret_20_rank":
            reason = "源变量ret_20已隔离，其横截面排名不得绕过隔离。"
        elif name in RAW[:6] or name == "amount_per_vol":
            reason = "当前缺少足够明确的跨股票定价机制或归一化口径；本候选移除水平变量。"
        elif not approved:
            reason = "本轮采用保守的中期风险、参与度和动量特征集；短窗口近似替代项暂不进入候选，未宣称其必然无效。"
        else:
            reason = "已为下一次训练事先登记可检验假设、可用时点和失效风险；仍须用户确认与时间序列验证。"
        result[name] = {**item, "status": "approved_for_confirmation" if approved else "quarantined",
                        "decision_reason": reason, "original_research_comment": documented,
                        "causal_evidence": "未经因果识别验证；金融解释仅为研究假设"}
    return result


def policy_fingerprint():
    payload = {"version": POLICY_VERSION, "registry": registry()}
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def make_policy(feature_mode="technical_momentum_skip"):
    names = feature_catalog(feature_mode)
    return {"version": POLICY_VERSION, "registry_fingerprint": policy_fingerprint(),
            "allowed_features": [name for name in names if name in APPROVED],
            "excluded_features": [name for name in names if name not in APPROVED],
            "display_logic_before_confirmation": True}


def validate_policy(config):
    policy = config.get("financial_logic")
    if not isinstance(policy, dict):
        raise ValueError("未通过金融逻辑审查：旧配置仅供查阅，请使用带financial_logic的已审查新候选。")
    if policy.get("version") != POLICY_VERSION or policy.get("registry_fingerprint") != policy_fingerprint():
        raise ValueError("金融逻辑登记版本或指纹已变化，必须重新审查并展示后再确认。")
    if policy.get("display_logic_before_confirmation") is not True:
        raise ValueError("训练前必须展示逐项金融逻辑。")
    names = feature_catalog(config["feature_mode"])
    allowed = policy.get("allowed_features")
    if not isinstance(allowed, list) or not allowed or len(set(allowed)) != len(allowed):
        raise ValueError("金融逻辑允许列表为空或有重复。")
    if not set(allowed).issubset(set(names) & set(APPROVED)):
        raise ValueError("未知、未审查或已隔离特征不得进入训练。")
    if policy.get("excluded_features") != [name for name in names if name not in allowed]:
        raise ValueError("剔除列表与实际允许列表不一致。")
    if not config.get("evaluation"):
        raise ValueError("新候选须使用已有标签边界隔离的滚动协议；旧固定切分仅保留历史记录。")
    return policy


def selection_details(config):
    policy = validate_policy(config)
    catalog = registry()
    return {**policy, "feature_count": len(policy["allowed_features"]),
            "retained": [{"feature": name, **catalog[name]} for name in policy["allowed_features"]],
            "excluded": [{"feature": name, **catalog[name]} for name in policy["excluded_features"]],
            "hypothesis_limit": "金融假设不是因果证明；换手变化必须在确认后的训练中检验。"}


def select_columns(actual_columns, config):
    policy = validate_policy(config)
    expected = feature_catalog(config["feature_mode"])
    if set(actual_columns) != set(expected):
        missing = sorted(set(expected) - set(actual_columns))
        extra = sorted(set(actual_columns) - set(expected))
        raise ValueError(f"实际生成的特征与已审查特征目录不一致，停止训练。missing={missing}; extra={extra}")
    return list(policy["allowed_features"])


def assert_approved_features(features):
    if not features or len(set(features)) != len(features) or not set(features).issubset(APPROVED):
        raise ValueError("拟合入口发现未经金融逻辑审核的特征。")


def write_selection_snapshot(path, config):
    with Path(path).open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(selection_details(config), handle, ensure_ascii=False, indent=2)
        handle.write("\n")
