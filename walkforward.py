"""Purged expanding-window protocol and official daily diagnostics."""
from __future__ import annotations
import hashlib
import json
import math
from datetime import datetime
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

METHOD = 'expanding_walk_forward'
METRIC_VERSION = 'official_daily_v1_label_boundary_purged'

def validate_protocol(config):
    p, s = config['evaluation'], config['split']
    if p.get('method') != METHOD:
        raise ValueError('不支持的评估协议')
    if set(s) != {'train_start','train_end','valid_start','valid_end','test_start','test_end'}:
        raise ValueError('split字段不完整或有多余字段')
    def check_date(d):
        if isinstance(d,bool) or not isinstance(d,int):
            raise ValueError('日期必须为YYYYMMDD整数')
        datetime.strptime(str(d),'%Y%m%d')
    for d in s.values(): check_date(d)
    if not (20180102 <= s['train_start'] <= s['train_end'] < s['valid_start'] <= s['valid_end'] <= 20241231):
        raise ValueError('训练与最终比较日期必须先后排列且位于已提供的有标签历史内')
    if (s['test_start'],s['test_end']) != (20250102,20260608):
        raise ValueError('比赛测试日期不得改变')
    if p.get('final_comparison_year') != s['valid_start']//10000 or s['valid_start']//10000 != s['valid_end']//10000:
        raise ValueError('最终比较年份与起止日期不一致')
    if p.get('label_horizon_trading_days') != 1 or p.get('boundary_policy') != 'label_date_before_next_stage':
        raise ValueError('必须按下一交易日标签兑现时间隔离边界')
    if p.get('final_rounds_rule') != 'median_inner_best_iteration':
        raise ValueError('最终轮数只能使用历史折内层早停轮数中位数')
    if p.get('test_usage') != 'prediction_only_after_cv_and_2024_assessment':
        raise ValueError('测试集仅可在历史评价完成后用于预测')
    patience=p.get('early_stopping_rounds')
    if isinstance(patience,bool) or not isinstance(patience,int) or patience <= 0:
        raise ValueError('早停耐心必须是正整数')
    if not p.get('protocol_id') or not isinstance(p.get('folds'),list) or len(p['folds']) != 3:
        raise ValueError('须指定协议名称和三折历史验证')
    names=set(); previous=None
    keys={'name','fit_start','inner_fit_end','early_stop_start','early_stop_end','refit_end','assessment_start','assessment_end'}
    for f in p['folds']:
        if set(f)!=keys: raise ValueError('折配置字段不完整')
        name=f['name']
        if not isinstance(name,str) or not name.replace('_','').isascii() or not name.replace('_','').isalnum() or name in names:
            raise ValueError('折名须为唯一的英文数字下划线')
        names.add(name)
        for k,v in f.items():
            if k!='name': check_date(v)
        if not (f['fit_start']==s['train_start'] <= f['inner_fit_end'] < f['early_stop_start'] <= f['early_stop_end'] <= f['refit_end'] < f['assessment_start'] <= f['assessment_end'] <= s['train_end']):
            raise ValueError('每折须满足内层训练 < 早停 <= 重训截止 < 评估；评估不得越过最终训练截止')
        if previous is not None and f['assessment_start'] <= previous:
            raise ValueError('外层评估日期不可重叠')
        previous=f['assessment_end']
    if p.get('max_model_fits') != 7: raise ValueError('须明确确认七次独立模型拟合')
    if not isinstance(config.get('model_params'),dict): raise ValueError('model_params必须为对象')
    if {'init_model','old_model','previous_model','external_data'} & set(config['model_params']):
        raise ValueError('禁止旧模型或外部训练数据参数')
    if config['model_params'].get('objective')!='regression': raise ValueError('本次保持regression目标')
    if not isinstance(config['model_params'].get('n_estimators'),int) or config['model_params']['n_estimators']<=0:
        raise ValueError('须明确正整数最大训练轮数')
    return p

def protocol_fingerprint(config):
    spec={'evaluation':config['evaluation'],'split':config['split'],'metric_version':METRIC_VERSION}
    return hashlib.sha256(json.dumps(spec,sort_keys=True).encode()).hexdigest()

def next_label_dates(dates):
    calendar=sorted(int(d) for d in pd.unique(dates))
    return dates.map(dict(zip(calendar[:-1],calendar[1:])))

def labeled_mask(frame,label_dates,start,end,known_before):
    eligible=frame['trade_date'].between(start,end) & frame['y_ret_1d'].notna()
    known=label_dates.notna() & label_dates.lt(known_before)
    return eligible & known, int((eligible & ~known).sum())

def assessment_frame(frame,label_dates,start,end):
    selected=frame['trade_date'].between(start,end)
    out=frame.loc[selected,['ts_code','trade_date','flag_limit_up','y_ret_1d']].copy()
    unknown=label_dates.loc[selected].isna() | label_dates.loc[selected].gt(end)
    purged=int((unknown & out['y_ret_1d'].notna()).sum())
    out.loc[unknown,'y_ret_1d']=np.nan
    return selected,out,purged

def _mean(values):
    return float(np.mean(values)) if len(values) else float('nan')

def _tie_diagnostics(tradable, n_top):
    """Measure whether the Top10% boundary is determined by prediction ties."""
    top = tradable.iloc[:n_top]
    unique_count = int(top['pred'].nunique(dropna=False))
    cutoff = top['pred'].iloc[-1]
    tied_candidates = int((tradable['pred'] == cutoff).sum())
    strictly_above = int((tradable['pred'] > cutoff).sum())
    boundary_slots = int(n_top - strictly_above)
    return {
        'top10_distinct_prediction_ratio': unique_count / n_top,
        'top10_boundary_tied_candidates': tied_candidates,
        'top10_boundary_slots': boundary_slots,
        'top10_boundary_ambiguous': tied_candidates > boundary_slots,
    }

def metric_bundle(predictions):
    if not np.isfinite(predictions['pred'].to_numpy()).all():
        raise ValueError('预测含非有限值，禁止替换为零')
    records=[]; previous=None
    for date,g in predictions.groupby('trade_date',sort=True):
        valid=g.dropna(subset=['y_ret_1d'])
        ic=float(spearmanr(valid['pred'],valid['y_ret_1d'])[0]) if len(valid)>=30 else float('nan')
        eligible=g[(g['flag_limit_up']==0)&g['y_ret_1d'].notna()].sort_values('pred',ascending=False)
        excess=top=float('nan')
        if len(eligible)>=100:
            top=float(eligible.iloc[:max(len(eligible)//10,1)]['y_ret_1d'].mean())
            excess=top-float(eligible['y_ret_1d'].mean())
        tradable=g[g['flag_limit_up']==0].sort_values('pred',ascending=False)
        turnover=float('nan')
        tie_info={'top10_distinct_prediction_ratio': float('nan'),
                  'top10_boundary_tied_candidates': 0,
                  'top10_boundary_slots': 0,
                  'top10_boundary_ambiguous': False}
        if len(tradable)>=100:
            n_top=max(len(tradable)//10,1)
            tie_info=_tie_diagnostics(tradable,n_top)
            current=set(tradable.iloc[:n_top]['ts_code'].astype(str))
            if previous: turnover=1-len(current&previous)/len(current|previous)
            previous=current
        else: previous=None
        records.append({'trade_date':int(date),'ic':ic,'ic_eligible':len(valid)>=30,'labeled_rows':len(valid),'excess':excess,'top1_return':top,'turnover':turnover,**tie_info})
    daily=pd.DataFrame(records)
    if daily.empty: raise ValueError('评估期无样本')
    def aggregate(part):
        ics=part.loc[part['ic_eligible'],'ic'].to_numpy()
        mean=_mean(ics); std=float(np.std(ics,ddof=1)) if len(ics)>1 else float('nan')
        annual=_mean(part['excess'].dropna().to_numpy())*252
        turnover=_mean(part['turnover'].dropna().to_numpy())
        tie_ratios=part['top10_distinct_prediction_ratio'].dropna().to_numpy()
        ambiguous=part['top10_boundary_ambiguous'].to_numpy(dtype=bool)
        return {'ic_mean':mean,'ic_std':std,'icir':mean/std if std>0 else 0.0,
                'ic_positive_ratio':float(np.mean(ics>0)) if len(ics) else float('nan'),
                'annual_excess':annual,'top1_annual_ret':_mean(part['top1_return'].dropna().to_numpy())*252,
                'mean_turnover':turnover,'stability_score':1-turnover,
                'final_score':.4*mean+.3*annual+.3*(1-turnover),
                'ic_days':len(ics),'excess_days':int(part['excess'].notna().sum()),'turnover_days':int(part['turnover'].notna().sum()),
                'top10_distinct_prediction_ratio_mean':_mean(tie_ratios),
                'top10_distinct_prediction_ratio_min':float(np.min(tie_ratios)) if len(tie_ratios) else float('nan'),
                'top10_boundary_ambiguous_days':int(np.sum(ambiguous)),
                'top10_boundary_ambiguous_ratio':float(np.mean(ambiguous)) if len(ambiguous) else float('nan'),
                'top10_boundary_tie_size_max':int(part['top10_boundary_tied_candidates'].max()) if len(part) else 0}
    metrics=aggregate(daily)
    labeled=predictions.dropna(subset=['y_ret_1d']); residual=labeled['pred']-labeled['y_ret_1d']
    metrics.update(rmse=float(np.sqrt(np.mean(residual**2))) if len(residual) else float('nan'),mae=float(np.mean(np.abs(residual))) if len(residual) else float('nan'))
    dates=pd.to_datetime(daily['trade_date'].astype(str),format='%Y%m%d')
    months=[{'period':str(k),**aggregate(g)} for k,g in daily.groupby(dates.dt.to_period('M'))]
    quarters=[{'period':str(k),**aggregate(g)} for k,g in daily.groupby(dates.dt.to_period('Q'))]
    known=[m['ic_mean'] for m in months if math.isfinite(m['ic_mean'])]
    metrics['monthly_ic_positive_ratio']=_mean(np.array(known)>0)
    metrics['monthly_ic_valid_months']=len(known)
    return metrics,daily,{'monthly':months,'quarterly':quarters,'turnover_period_boundary':'retain previous-day transition within the evaluated series'}

def clean_json(value):
    if isinstance(value,dict): return {str(k):clean_json(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)): return [clean_json(v) for v in value]
    if isinstance(value,np.generic): return clean_json(value.item())
    if isinstance(value,float) and not math.isfinite(value): return None
    return value

def write_json(path,value):
    path.write_text(json.dumps(clean_json(value),ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')

def fold_aggregate(folds):
    out={}
    for key in ['final_score','ic_mean','icir','annual_excess','mean_turnover','ic_positive_ratio',
                'top10_distinct_prediction_ratio_mean','top10_distinct_prediction_ratio_min',
                'top10_boundary_ambiguous_days','top10_boundary_ambiguous_ratio','top10_boundary_tie_size_max']:
        vals=[f['metrics'].get(key) for f in folds if f['metrics'].get(key) is not None and math.isfinite(f['metrics'].get(key))]
        out[key]={'mean':_mean(vals),'std':float(np.std(vals,ddof=1)) if len(vals)>1 else float('nan'),'min':min(vals) if vals else None,'max':max(vals) if vals else None,'valid_folds':len(vals)}
    return out

def run_walk_forward(config,train_path,test_path,workspace):
    from walkforward_run import run
    return run(config,train_path,test_path,workspace)
