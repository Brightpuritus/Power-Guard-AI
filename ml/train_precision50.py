"""XGBoost recall search at predeclared validation precision floors 50% and 55%.

Test data never select parameters or thresholds. The 55% policy is a predeclared
buffer, not a guarantee on future data. Original 74 features and split are reused.
"""
import argparse
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from improve_recall import metrics, recall_threshold


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--out',default='models/experiments/results_precision50')
    args=ap.parse_args()
    out=Path(args.out)
    if out.exists():ap.error('Choose a new output directory')
    (out/'models').mkdir(parents=True)
    src=Path('models/experiments/results_all_updated')
    summary=json.loads((src/'run_summary.json').read_text())
    data=pd.read_csv(src/'features.csv',dtype={'CONS_NO':str})
    X=data[summary['features']];y=data.FLAG.to_numpy();ids=data.CONS_NO.to_numpy()
    tv,te=train_test_split(np.arange(len(y)),test_size=.15,stratify=y,random_state=42)
    tr,va=train_test_split(tv,test_size=.15/.85,stratify=y[tv],random_state=42)
    saved=pd.read_csv(src/'test_predictions.csv',dtype={'CONS_NO':str}).set_index('CONS_NO')
    assert set(ids[te])==set(saved.index)
    np.testing.assert_array_equal(y[te],saved.loc[ids[te],'y_true'])
    weight=(y[tr]==0).sum()/(y[tr]==1).sum()
    configs=[]
    for depth in [3,4,5,6]:
        for wname,w in [('one',1.),('sqrt',float(np.sqrt(weight))),('balanced',float(weight))]:
            configs.append((f'd{depth}_{wname}',dict(max_depth=depth,scale_pos_weight=w,min_child_weight=5,reg_lambda=5)))
    configs.extend([
        ('d5_regularized',dict(max_depth=5,scale_pos_weight=float(np.sqrt(weight)),min_child_weight=15,reg_lambda=15,reg_alpha=.5)),
        ('d6_regularized',dict(max_depth=6,scale_pos_weight=1.,min_child_weight=15,reg_lambda=15,reg_alpha=.5)),
        ('d4_gentle',dict(max_depth=4,scale_pos_weight=2.,min_child_weight=1,reg_lambda=1)),
        ('d7_regularized',dict(max_depth=7,scale_pos_weight=2.,min_child_weight=15,reg_lambda=15,reg_alpha=.5))])
    plan={'feature_count':74,'seed':42,'precision_floors':[.5,.55],
          'primary':'Max validation recall at precision >= .50',
          'secondary':'Predeclared buffer: max validation recall at precision >= .55',
          'tie_breaks':['Precision','PR-AUC'],'configs':configs,
          'test_policy':'Only frozen winners, baseline, and baseline threshold-only control evaluated on test.'}
    (out/'plan.json').write_text(json.dumps(plan,indent=2))
    baseline=joblib.load(src/'models/xgboost.joblib')
    old_threshold=summary['thresholds']['XGBoost']
    val_rows=[];thresholds={};score_frame=pd.DataFrame({'CONS_NO':ids[va],'y_true':y[va]})
    def register(name,model):
        scores=model.predict_proba(X.iloc[va])[:,1]
        score_frame[name]=scores
        joblib.dump(model,out/'models'/(name+'.joblib'))
        thresholds[name]={}
        for floor in [.5,.55]:
            t=recall_threshold(y[va],scores,floor)
            if t is None:continue
            thresholds[name][floor]=t
            row=metrics(name,y[va],scores,t);row['Precision_floor']=floor
            val_rows.append(row)
        line=[r for r in val_rows if r['Model']==name]
        print(name+': '+', '.join(f"floor {r['Precision_floor']:.0%}: P={r['Precision']:.3f}, R={r['Recall']:.3f}" for r in line),flush=True)
        pd.DataFrame(val_rows).to_csv(out/'validation_candidates.csv',index=False)
    register('current',baseline)
    for name,config in configs:
        params=dict(n_estimators=1800,learning_rate=.025,subsample=.85,colsample_bytree=.9,
                    eval_metric='aucpr',early_stopping_rounds=100,n_jobs=4,random_state=42,tree_method='hist')
        params.update(config)
        model=XGBClassifier(**params)
        model.fit(X.iloc[tr],y[tr],eval_set=[(X.iloc[va],y[va])],verbose=False)
        register(name,model)
    winners={}
    for floor in [.5,.55]:
        candidates=[r for r in val_rows if r['Precision_floor']==floor]
        winners[str(floor)]=max(candidates,key=lambda r:(r['Recall'],r['Precision'],r['PR-AUC']))
    (out/'selection.json').write_text(json.dumps(winners,indent=2))
    score_frame.to_csv(out/'validation_predictions.csv',index=False)
    # No test scoring until both policies have been frozen above.
    old_scores=baseline.predict_proba(X.iloc[te])[:,1]
    np.testing.assert_allclose(old_scores,saved.loc[ids[te],'score_XGBoost'],atol=1e-7)
    policies=[('Current', 'current',old_threshold),
              ('Current threshold-only P50','current',thresholds['current'][.5])]
    for floor,label in [(.5,'Retrained P50'),(.55,'Retrained P55 buffer')]:
        chosen=winners[str(floor)]
        policies.append((label,chosen['Model'],chosen['Threshold']))
    results=[]
    pred=pd.DataFrame({'CONS_NO':ids[te],'y_true':y[te]})
    from experiment_missing import score_raw
    import train_compare as tc
    raw_sample=pd.read_csv('data/raw/data.csv',nrows=40,dtype={'CONS_NO':str})
    datecols=[c for c in raw_sample if c not in ['CONS_NO','FLAG']]
    dates=pd.to_datetime(pd.Series(datecols));order=np.argsort(dates.values)
    raw=raw_sample[[datecols[i] for i in order]].to_numpy(dtype=np.float32)
    dow=dates.iloc[order].dt.dayofweek.to_numpy()
    for label,name,t in policies:
        model=joblib.load(out/'models'/(name+'.joblib'))
        scores=model.predict_proba(X.iloc[te])[:,1]
        row=metrics(label,y[te],scores,t);row['Candidate']=name
        row['Meets_test_precision50']=bool(row['Precision']>=.5)
        row['Recall_gain_pp']=(row['Recall']-results[0]['Recall'])*100 if results else 0.
        results.append(row);pred[label]=scores
        bundle={'model':model,'variant':'combined','threshold':t,'features':list(X.columns)}
        if label.startswith('Retrained'):
            joblib.dump(bundle,out/('selected_p50.joblib' if label=='Retrained P50' else 'selected_p55.joblib'))
        sample_scores,_=score_raw(bundle,raw,dow)
        expected=model.predict_proba(X.iloc[:40])[:,1]
        np.testing.assert_allclose(sample_scores,expected,atol=1e-7)
        # SHAP additivity with the same early stopping range as prediction.
        import xgboost as xgb
        f=X.iloc[te[:30]]
        contrib=model.get_booster().predict(xgb.DMatrix(f),pred_contribs=True,
                                           iteration_range=(0,model.best_iteration+1))
        np.testing.assert_allclose(contrib.sum(1),model.predict(f,output_margin=True),atol=1e-5)
    table=pd.DataFrame(results);table.to_csv(out/'test_comparison.csv',index=False)
    pred.to_csv(out/'test_predictions.csv',index=False)
    (out/'verification.json').write_text(json.dumps({'original_test_scores_match':True,'raw_inference_parity_rows':40,
            'SHAP_additivity_rows':30,'models_verified':len(policies)},indent=2))
    plot(out,table,y[te],pred)
    print(table.to_string(index=False),flush=True)


def plot(out,table,y,pred):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sklearn.metrics import precision_recall_curve
    colors=['#7C8794','#3487B8','#128048','#D4842A']
    labels=['Current','Threshold only','Retrained P50','Retrained P55']
    fig,axes=plt.subplots(1,2,figsize=(14,5.5))
    xx=np.arange(3)
    for i,row in table.iterrows():
        bars=axes[0].bar(xx+(i-1.5)*.2,[row.Precision,row.Recall,row.F1],.2,color=colors[i],label=labels[i])
        axes[0].bar_label(bars,fmt='%.3f',padding=3,fontsize=8)
        scores=pred[row.Model].to_numpy();p,r,_=precision_recall_curve(y,scores)
        if i!=1:axes[1].plot(r,p,color=colors[i],label=labels[i],alpha=.85)
        axes[1].scatter(row.Recall,row.Precision,color=colors[i],s=65,zorder=3)
    axes[0].set(xticks=xx,xticklabels=['Precision','Recall','F1'],ylim=(0,.8),title='Test metrics at frozen validation thresholds')
    axes[0].legend(fontsize=9)
    axes[1].axhline(.5,ls='--',color='black',label='Precision target 50%')
    axes[1].set(xlabel='Recall',ylabel='Precision',xlim=(0,.8),ylim=(0,1),title='Test PR curves and selected operating points')
    axes[1].legend(fontsize=9);axes[1].grid(alpha=.2)
    fig.tight_layout()
    for ext in ['png','svg']:fig.savefig(out/('comparison.'+ext),dpi=200,bbox_inches='tight')
    plt.close(fig)


if __name__=='__main__':main()
