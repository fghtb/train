"""Execution of confirmed walk-forward candidates; never invoked by --preview."""
from __future__ import annotations
import gc
import json
import shutil
import time
from datetime import datetime
from pathlib import Path
import lightgbm as lgb
import numpy as np
import pandas as pd
from walkforward import (METRIC_VERSION, validate_protocol, protocol_fingerprint, next_label_dates,
                         labeled_mask, assessment_frame, metric_bundle, fold_aggregate, write_json)


def _fit(params, data, features, stop=None, patience=100, stage="unspecified"):
    from financial_logic import assert_approved_features, registry, POLICY_VERSION
    assert_approved_features(features)
    feature_logic = registry()
    print(json.dumps({'training_stage':stage,'effective_model_params':params,
                      'fit_date_min':int(data['trade_date'].min()),'fit_date_max':int(data['trade_date'].max()),
                      'fit_rows':len(data),'early_stop_used':stop is not None,'init_model':None,
                      'financial_logic_version':POLICY_VERSION,
                      'feature_financial_logic':[{'feature':name,'logic':feature_logic[name]['logic']} for name in features]},
                     ensure_ascii=False,indent=2),flush=True)
    model=lgb.LGBMRegressor(**params)
    options={'init_model':None}
    if stop is not None:
        options.update(eval_set=[(stop[features],stop['y_ret_1d'])],eval_metric='rmse',
                       callbacks=[lgb.early_stopping(patience),lgb.log_evaluation(100)])
    model.fit(data[features],data['y_ret_1d'],**options)
    return model


def run(config, train_path, test_path, workspace):
    import train_models as base
    from reporting import rebuild_report
    from financial_logic import validate_policy, write_selection_snapshot
    validate_policy(config)
    if (train_path, test_path) != base.resolve_data_paths(config):
        raise ValueError('执行路径必须为配置验证过的两份题目数据')
    protocol=validate_protocol(config); split=config['split']; started=time.perf_counter()
    run_id=datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    run_dir=workspace/'runs'/f"{config['model_name']}_{run_id}"
    run_dir.mkdir(parents=True,exist_ok=False)
    write_json(run_dir/'config_snapshot.json',config)
    write_selection_snapshot(run_dir/'financial_logic_snapshot.json',config)
    write_json(run_dir/'run_status.json',{'status':'running','confirmation':'received_by_parent','planned_fits':7})
    snapshot=run_dir/'source_snapshot'; snapshot.mkdir()
    source_root=Path(__file__).resolve().parent
    for name in ['train_models.py','walkforward.py','walkforward_run.py','reporting.py',
                 'financial_logic.py','financial_logic_reporting.py','portfolio_attribution.py']:
        shutil.copyfile(source_root/name,snapshot/name)
    try:
        # Only provided training data is opened while selecting rounds and assessing history.
        base.audit_source_schema(train_path,base.TRAIN_DTYPES)
        train=pd.read_csv(train_path,dtype=base.TRAIN_DTYPES,usecols=list(base.TRAIN_DTYPES))
        if train[['ts_code','trade_date']].isna().any().any() or train[['ts_code','trade_date']].duplicated().any():
            raise ValueError('训练主键缺失或重复')
        if int(train['trade_date'].min())>split['train_start'] or int(train['trade_date'].max())<split['valid_end']:
            raise ValueError('训练文件未覆盖协议日期')
        label_dates=next_label_dates(train['trade_date'])
        empty=train.drop(columns=['y_ret_1d']).iloc[:0].copy()
        history,_=base.make_features(train,empty,config['feature_mode'])
        features=base.select_training_features(history,config)
        # Catalog is validated before discarding excluded columns; fits retain
        # only the already confirmed input columns, with unchanged values.
        # Keep the supplied limit-up flag for portfolio diagnostics even when
        # the research candidate intentionally excludes it from model inputs.
        history=history[['ts_code','trade_date','flag_limit_up']+features]
        history['y_ret_1d']=train['y_ret_1d'].to_numpy()
        fold_records=[]; best_iterations=[]; fit_count=0
        for fold in protocol['folds']:
            folder=run_dir/'folds'/fold['name']; folder.mkdir(parents=True)
            fit_mask,fit_purge=labeled_mask(history,label_dates,fold['fit_start'],fold['inner_fit_end'],fold['early_stop_start'])
            stop_mask,stop_purge=labeled_mask(history,label_dates,fold['early_stop_start'],fold['early_stop_end'],fold['assessment_start'])
            if not fit_mask.any() or not stop_mask.any(): raise ValueError('内层训练或早停期没有有效样本')
            print(f"{fold['name']}: 内层选轮数，之后全新重训评估下一年",flush=True)
            calibration=_fit(dict(config['model_params']),history.loc[fit_mask],features,history.loc[stop_mask],protocol['early_stopping_rounds'],stage=fold['name']+'_inner_early_stop')
            fit_count+=1
            best=int(calibration.best_iteration_ or config['model_params']['n_estimators'])
            calibration.booster_.save_model(str(folder/'early_stop_model.txt'))
            del calibration; gc.collect()
            refit_mask,refit_purge=labeled_mask(history,label_dates,fold['fit_start'],fold['refit_end'],fold['assessment_start'])
            refit_params={**config['model_params'],'n_estimators':best}
            model=_fit(refit_params,history.loc[refit_mask],features,stage=fold['name']+'_outer_refit'); fit_count+=1
            selected,predictions,label_purge=assessment_frame(history,label_dates,fold['assessment_start'],fold['assessment_end'])
            predictions['pred']=model.predict(history.loc[selected,features],num_iteration=best)
            metrics,daily,periods=metric_bundle(predictions)
            model.booster_.save_model(str(folder/'model.txt'))
            predictions.to_csv(folder/'validation_predictions.csv',index=False)
            daily.to_csv(folder/'daily_metrics.csv',index=False)
            write_json(folder/'period_metrics.json',periods)
            record={'name':fold['name'],'split':fold,'best_iteration':best,'metrics':metrics,'period_metrics':periods,'refit_model_params':refit_params,
                    'rows':{'inner_fit':int(fit_mask.sum()),'early_stop':int(stop_mask.sum()),'refit':int(refit_mask.sum()),'assessment':int(selected.sum())},
                    'label_boundary_purged_rows':{'inner_fit':fit_purge,'early_stop':stop_purge,'refit':refit_purge,'assessment_labels':label_purge}}
            write_json(folder/'fold_summary.json',record)
            fold_records.append(record); best_iterations.append(best)
            write_json(run_dir/'cv_progress.json',{'completed_folds':len(fold_records),'actual_fits':fit_count,'folds':fold_records})
            del model,predictions; gc.collect()

        # Final rounds are frozen before the final-comparison labels or test X can influence them.
        final_rounds=int(np.median(best_iterations))
        final_params={**config['model_params'],'n_estimators':final_rounds}
        fit_mask,final_purge=labeled_mask(history,label_dates,split['train_start'],split['train_end'],split['valid_start'])
        final_model=_fit(final_params,history.loc[fit_mask],features,stage='final_frozen_rounds_refit'); fit_count+=1
        selected,valid_out,comparison_purge=assessment_frame(history,label_dates,split['valid_start'],split['valid_end'])
        valid_out['pred']=final_model.predict(history.loc[selected,features],num_iteration=final_rounds)
        metrics,daily,periods=metric_bundle(valid_out)
        final_model.booster_.save_model(str(run_dir/'model.txt'))
        valid_out.to_csv(run_dir/'validation_predictions.csv',index=False)
        daily.to_csv(run_dir/'daily_metrics_final_comparison.csv',index=False)
        write_json(run_dir/'period_metrics.json',periods)
        cross_validation={'protocol_id':protocol['protocol_id'],'protocol_fingerprint':protocol_fingerprint(config),'metric_version':METRIC_VERSION,
                          'folds':fold_records,'aggregate':fold_aggregate(fold_records),'final_rounds_rule':protocol['final_rounds_rule'],
                          'inner_best_iterations':best_iterations,'final_best_iteration':final_rounds,'final_fit_params':final_params,'actual_fits':fit_count,
                          'comparison_used_for_early_stopping':False,'comparison_is_unseen_holdout':False,'final_training_uses_comparison_labels':False,
                          'final_fit_purged_rows':final_purge,'comparison_purged_labels':comparison_purge}
        assert fit_count==protocol['max_model_fits']
        write_json(run_dir/'cv_summary.json',cross_validation)
        del history; gc.collect()

        # Test X is first read after all historical evaluation and parameter freezing.
        base.audit_source_schema(test_path,base.TEST_DTYPES)
        test=pd.read_csv(test_path,dtype=base.TEST_DTYPES,usecols=list(base.TEST_DTYPES))
        audit=base.audit_keys_and_dates(train,test)
        if not test['trade_date'].between(split['test_start'],split['test_end']).all():
            raise ValueError('测试集含协议之外的日期')
        unused,test_features=base.make_features(train,test,config['feature_mode'])
        del unused; gc.collect()
        if base.select_training_features(test_features,config) != features:
            raise ValueError('Prediction feature order differs from confirmed training inputs')
        test_features=test_features[['ts_code','trade_date']+features]
        output=test[['ts_code','trade_date']].copy()
        output['pred']=final_model.predict(test_features[features],num_iteration=final_rounds)
        if not np.isfinite(output['pred'].to_numpy()).all(): raise ValueError('预测非有限，禁止用零替代')
        if not output[['ts_code','trade_date']].equals(test[['ts_code','trade_date']]) or output[['ts_code','trade_date']].duplicated().any():
            raise ValueError('提交键不一致或重复')
        output.to_csv(run_dir/'submission.csv',index=False)
        pd.DataFrame({'feature':features,'importance_gain':final_model.booster_.feature_importance(importance_type='gain'),
                      'importance_split':final_model.booster_.feature_importance(importance_type='split')}).sort_values('importance_gain',ascending=False).to_csv(run_dir/'feature_importance.csv',index=False)
        write_json(run_dir/'feature_columns.json',features)
        write_json(run_dir/'data_manifest.json',{'train_source':base.file_manifest(train_path),'test_source':base.file_manifest(test_path),
                   'label_horizon_trading_days':1,'test_read_after_cv_and_comparison':True,
                   'label_time_policy':'next provided market trading date; unknown or out-of-period labels excluded',
                   'package_versions':{'lightgbm':lgb.__version__,'pandas':pd.__version__,'numpy':np.__version__}})
        sample_rows=int(config.get('save',{}).get('sample_rows',1000))
        valid_out.head(sample_rows).to_csv(run_dir/'data_valid_sample.csv',index=False)
        test_features.head(sample_rows).to_csv(run_dir/'data_test_sample.csv',index=False)
        summary={'run_id':run_id,'model_name':config['model_name'],'created_at':datetime.now().isoformat(timespec='seconds'),
                 'training_seconds':round(time.perf_counter()-started,3),'feature_mode':config['feature_mode'],'feature_count':len(features),'feature_columns':features,
                 'optimization_plan':config.get('optimization_plan'),'research_references':config.get('research_references',[]),
                 'model_params':config['model_params'],'effective_final_model_params':final_params,'split':split,'evaluation':protocol,
                 'financial_logic':config['financial_logic'],
                 'rows':{'fit':int(fit_mask.sum()),'valid':len(valid_out),'test':len(test)},'missing_y_train':int(train['y_ret_1d'].isna().sum()),'audit':audit,
                 'best_iteration':final_rounds,'metrics':metrics,'period_metrics':periods,'cross_validation':cross_validation,'run_dir':str(run_dir),
                 'constraints':{'source_paths_whitelisted':True,'source_schema_exact':True,'only_provided_columns':True,
                 'future_data_policy':'current_and_past_features_with_label_boundary_purge','time_split_protocol':protocol['protocol_id'],
                 'test_labels_used':False,'old_model_loaded':False,'init_model':None,'submission_alignment_checked':True,'submission_finite_checked':True,'outer_assessment_used_for_early_stopping':False},
                 'artifacts':{'model':str(run_dir/'model.txt'),'validation_predictions':str(run_dir/'validation_predictions.csv'),
                 'submission':str(run_dir/'submission.csv'),'feature_importance':str(run_dir/'feature_importance.csv'),
                 'data_manifest':str(run_dir/'data_manifest.json'),'cross_validation':str(run_dir/'cv_summary.json'),
                 'financial_logic':str(run_dir/'financial_logic_snapshot.json')}}
        write_json(run_dir/'summary.json',summary)
        write_json(run_dir/'run_status.json',{'status':'complete','actual_fits':fit_count})
        if not str(config.get('model_name', '')).startswith('research_1_'):
            rebuild_report(workspace)
        print(f"完成三折历史验证、最终比较和测试预测，共{fit_count}次独立拟合。运行目录：{run_dir}",flush=True)
        return run_dir
    except BaseException as exc:
        write_json(run_dir/'run_status.json',{'status':'failed','error_type':type(exc).__name__,'error':str(exc)})
        raise
