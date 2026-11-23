"""Retrain original three model families on the updated 74-feature pipeline.

python train_all_updated.py --out results_all_updated
All model/threshold selection uses validation. Test is reported after freezing.
"""
import argparse
import copy
import json
from pathlib import Path
import warnings

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, confusion_matrix,
                             precision_recall_curve, roc_curve)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import QuantileTransformer, StandardScaler
from xgboost import XGBClassifier

import train_compare as tc
from experiment_missing import make_features
from improve_recall import metrics, recall_threshold

NAMES = ['Logistic Regression', 'XGBoost', 'Autoencoder']
COLORS = dict(zip(NAMES, ['#3476B8', '#128048', '#D5812B']))


def load_ae(directory):
    checkpoint = torch.load(Path(directory)/'autoencoder.pt', map_location='cpu', weights_only=True)
    d = checkpoint['input_dim']
    nn = torch.nn
    net = nn.Sequential(nn.Linear(d,256),nn.ReLU(),nn.Linear(256,64),nn.ReLU(),
                        nn.Linear(64,16),nn.ReLU(),nn.Linear(16,64),nn.ReLU(),
                        nn.Linear(64,256),nn.ReLU(),nn.Linear(256,d),nn.Sigmoid())
    net.load_state_dict(checkpoint['state_dict'])
    net.eval()
    return net


def ae_predict(net, values):
    values = np.ascontiguousarray(values, dtype=np.float32)
    result=[]
    with torch.no_grad():
        for start in range(0,len(values),2048):
            x=torch.from_numpy(values[start:start+2048])
            result.append(((net(x)-x)**2).mean(1).numpy())
    return np.concatenate(result)


def score_saved(directory, raw, dow):
    """Score sorted customers x days using the same feature/preprocessing pipeline."""
    directory=Path(directory)
    summary=json.loads((directory/'run_summary.json').read_text(encoding='utf-8'))
    raw=np.where(np.isfinite(raw)&(raw>=0),raw,np.nan)
    X=make_features(raw,dow,'combined')[summary['features']]
    models=directory/'models'
    lr=joblib.load(models/'logreg.joblib')
    xgb=joblib.load(models/'xgboost.joblib')
    ae_pre=joblib.load(models/'autoencoder_preprocessor.joblib')
    return {NAMES[0]:lr.predict_proba(X)[:,1],NAMES[1]:xgb.predict_proba(X)[:,1],
            NAMES[2]:ae_predict(load_ae(models),ae_pre.transform(X))}


def plot_all(out, y, scores, thresholds, table, original, history):
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
    charts=out/'charts'
    charts.mkdir(exist_ok=True)
    def save(fig,name):
        fig.tight_layout()
        fig.savefig(charts/(name+'.png'),dpi=200,bbox_inches='tight')
        fig.savefig(charts/(name+'.svg'),bbox_inches='tight')
        plt.close(fig)

    fig,axes=plt.subplots(1,2,figsize=(13,5.5))
    cols=['Precision','Recall','F1']
    xx=np.arange(len(cols))
    for i,row in table.iterrows():
        bars=axes[0].bar(xx+(i-1)*.24,[row[c] for c in cols],.24,color=COLORS[row.Model],label=row.Model)
        axes[0].bar_label(bars,fmt='%.3f',padding=3,fontsize=9)
    axes[0].set(xticks=xx,xticklabels=cols,ylim=(0,1.08),title='At validation-selected best-F1 thresholds')
    xx=np.arange(2)
    for i,row in table.iterrows():
        bars=axes[1].bar(xx+(i-1)*.24,[row['PR-AUC'],row['ROC-AUC']],.24,color=COLORS[row.Model],label=row.Model)
        axes[1].bar_label(bars,fmt='%.3f',padding=3,fontsize=9)
    axes[1].set(xticks=xx,xticklabels=['PR-AUC (AP)','ROC-AUC'],ylim=(0,1.08),title='Ranking quality on test')
    axes[1].legend(loc='upper left',fontsize=9)
    fig.suptitle('Updated models: same 74 input features and customer split',fontsize=15)
    save(fig,'01_model_comparison')

    fig,axes=plt.subplots(1,2,figsize=(13,5.5))
    for name,s in scores.items():
        p,r,_=precision_recall_curve(y,s)
        axes[0].plot(r,p,label=f'{name} (AP={average_precision_score(y,s):.3f})',color=COLORS[name],lw=2)
        fpr,tpr,_=roc_curve(y,s)
        axes[1].plot(fpr,tpr,label=name,color=COLORS[name],lw=2)
    axes[0].axhline(y.mean(),color='gray',ls='--',label=f'Prevalence {y.mean():.1%}')
    axes[1].plot([0,1],[0,1],color='gray',ls='--')
    axes[0].set(xlabel='Recall',ylabel='Precision',title='Precision-recall curves',xlim=(0,1),ylim=(0,1.02))
    axes[1].set(xlabel='False positive rate',ylabel='True positive rate',title='ROC curves',xlim=(0,1),ylim=(0,1.02))
    for ax in axes: ax.legend(fontsize=9);ax.grid(alpha=.2)
    save(fig,'02_pr_roc_curves')

    fig,axes=plt.subplots(1,3,figsize=(14,4.7))
    for ax,(name,s) in zip(axes,scores.items()):
        cm=confusion_matrix(y,s>=thresholds[name],labels=[0,1])
        ax.imshow(cm,cmap='Greens')
        for i in range(2):
            for j in range(2):
                ax.text(j,i,f'{cm[i,j]:,}',ha='center',va='center',fontsize=18,
                        color='white' if cm[i,j]>cm.max()/2 else '#17213D')
        ax.set(xticks=[0,1],yticks=[0,1],xticklabels=['Normal','Theft'],yticklabels=['Normal','Theft'],
               xlabel='Predicted',ylabel='Actual',title=name)
    fig.suptitle('Test confusion matrices: best-F1 threshold chosen on validation',fontsize=14)
    save(fig,'03_confusion_matrices')

    fig,axes=plt.subplots(1,2,figsize=(13,5.5))
    ks=np.arange(20,1001,10)
    for name,s in scores.items():
        cumulative=np.cumsum(y[np.argsort(-s,kind='stable')])
        axes[0].plot(ks,cumulative[ks-1]/ks,label=name,color=COLORS[name],lw=2)
        axes[1].plot(ks,cumulative[ks-1]/y.sum(),label=name,color=COLORS[name],lw=2)
    axes[0].axhline(y.mean(),ls='--',color='gray',label='Random inspection')
    axes[0].set(ylabel='Precision@K',title='Yield at the same inspection budget')
    axes[1].set(ylabel='Recall@K',title='Coverage at the same inspection budget')
    for ax in axes:
        ax.set(xlabel='K: number of customers inspected',ylim=(0,1));ax.legend(fontsize=9);ax.grid(alpha=.2)
    save(fig,'04_precision_recall_at_k')

    fig,axes=plt.subplots(1,3,figsize=(15,5))
    cols=['Precision','Recall','F1','PR-AUC']
    for ax,name in zip(axes,NAMES):
        old=original.set_index('Model').loc[name]
        new=table.set_index('Model').loc[name]
        xx=np.arange(4)
        for offset,row,label,color in [(-.18,old,'Original','#A9B1BA'),(.18,new,'Updated',COLORS[name])]:
            bars=ax.bar(xx+offset,[row[c] for c in cols],.36,label=label,color=color)
            ax.bar_label(bars,fmt='%.2f',fontsize=8,padding=3)
        ax.set(xticks=xx,xticklabels=cols,ylim=(0,1.08),title=name)
        ax.tick_params(axis='x',labelsize=9);ax.legend(fontsize=9)
    fig.suptitle('Original vs updated (both use validation best-F1 thresholds)\nAutoencoder input changed: daily series to 74 engineered features',fontsize=13)
    save(fig,'05_original_vs_updated')

    fig,ax=plt.subplots(figsize=(8,4))
    ax.plot([v[0] for v in history],label='Train: normal only')
    ax.plot([v[1] for v in history],label='Validation: normal only')
    ax.set(xlabel='Epoch',ylabel='Reconstruction MSE',title='Feature autoencoder training loss')
    ax.legend();save(fig,'06_autoencoder_loss')


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--out',default='results_all_updated')
    ap.add_argument('--epochs',type=int,default=60)
    args=ap.parse_args()
    out=Path(args.out)
    if out.exists():ap.error('Choose a new output directory')
    out.mkdir(parents=True);(out/'models').mkdir()
    warnings.filterwarnings('ignore',category=RuntimeWarning)
    torch.set_num_threads(4)
    np.random.seed(42)
    print('Building updated 74 features...',flush=True)
    ids,y,raw,dow=tc.load_sgcc('data.csv')
    X=make_features(raw,dow,'combined')
    assert X.shape[1]==74 and not np.isinf(X.to_numpy()).any()
    tv,te=train_test_split(np.arange(len(y)),test_size=.15,stratify=y,random_state=42)
    tr,va=train_test_split(tv,test_size=.15/.85,stratify=y[tv],random_state=42)
    saved=pd.read_csv('results/test_predictions.csv',dtype={'CONS_NO':str}).set_index('CONS_NO')
    assert set(ids[te])==set(saved.index)
    np.testing.assert_array_equal(y[te],saved.loc[ids[te],'y_true'])
    pd.DataFrame({'CONS_NO':ids,'split':np.select([np.isin(np.arange(len(y)),tr),np.isin(np.arange(len(y)),va)],['train','validation'],default='test')}).to_csv(out/'split.csv',index=False)
    features=X.copy();features.insert(0,'CONS_NO',ids);features['FLAG']=y
    features.to_csv(out/'features.csv',index=False)
    models,vs={},{}
    # Tune only C for the baseline; fit imputation/scaling on train only.
    lr_candidates=[]
    for c in [.1,1.,10.]:
        lr=make_pipeline(SimpleImputer(strategy='median',keep_empty_features=True),StandardScaler(),
                         LogisticRegression(C=c,class_weight='balanced',max_iter=5000,random_state=42))
        lr.fit(X.iloc[tr],y[tr])
        score=lr.predict_proba(X.iloc[va])[:,1]
        apval=average_precision_score(y[va],score)
        lr_candidates.append((apval,c,lr,score))
        print(f'Logistic Regression C={c}: validation AP={apval:.4f}',flush=True)
    _,best_c,models[NAMES[0]],vs[NAMES[0]]=max(lr_candidates,key=lambda z:z[0])
    joblib.dump(models[NAMES[0]],out/'models/logreg.joblib')
    # Refit XGBoost with the already selected configuration on exactly the same train rows.
    prior=joblib.load('results_recall/models/xgboost.joblib')
    xgb=XGBClassifier(**prior.get_params())
    xgb.fit(X.iloc[tr],y[tr],eval_set=[(X.iloc[va],y[va])],verbose=False)
    models[NAMES[1]]=xgb;vs[NAMES[1]]=xgb.predict_proba(X.iloc[va])[:,1]
    joblib.dump(xgb,out/'models/xgboost.joblib')
    print(f'XGBoost validation AP={average_precision_score(y[va],vs[NAMES[1]]):.4f}',flush=True)
    # Bounded rank scaling avoids a single feature dominating reconstruction MSE.
    # Both preprocessing and AE weights learn from NORMAL TRAIN customers only.
    tr_n=tr[y[tr]==0];va_n=va[y[va]==0]
    pre=make_pipeline(SimpleImputer(strategy='median',keep_empty_features=True),
                      QuantileTransformer(n_quantiles=500,output_distribution='uniform',random_state=42))
    pre.fit(X.iloc[tr_n])
    joblib.dump(pre,out/'models/autoencoder_preprocessor.joblib')
    ae_values=np.ascontiguousarray(pre.transform(X),dtype=np.float32)
    net,ae_score,history=tc.train_autoencoder(ae_values[tr_n],ae_values[va_n],epochs=args.epochs)
    vs[NAMES[2]]=ae_score(ae_values[va])
    torch.save({'input_dim':74,'state_dict':{k:v.detach().cpu() for k,v in net.state_dict().items()}},out/'models/autoencoder.pt')
    thresholds={n:tc.best_f1_threshold(y[va],s) for n,s in vs.items()}
    floor=json.loads(Path('results_missing_v2/selection.json').read_text())['precision_floor']
    constrained={n:recall_threshold(y[va],s,floor) for n,s in vs.items()}
    val_rows=[metrics(n,y[va],s,thresholds[n]) for n,s in vs.items()]
    best=max(vs,key=lambda n:average_precision_score(y[va],vs[n]))
    summary={'features':list(X.columns),'feature_pipeline':'combined_v2','n_train':len(tr),'n_val':len(va),'n_test':len(te),
             'thresholds':thresholds,'constrained_thresholds':constrained,'precision_floor':floor,
             'best_model_by_validation_AP':best,'logreg_selected_C':best_c,
             'autoencoder_input':'74 features; median + quantile uniform fit on normal train only',
             'autoencoder_epochs':len(history),'seed':42,
             'primary_threshold_rule':'best F1 on validation for each model, matching original report',
             'selection_note':'All settings frozen before test evaluation; test set previously inspected during development.'}
    (out/'run_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    pd.DataFrame(val_rows).to_csv(out/'validation_metrics.csv',index=False)
    pd.DataFrame([{'C':c,'validation_AP':apv} for apv,c,_,_ in lr_candidates]).to_csv(out/'logreg_candidates.csv',index=False)
    # Test evaluation starts only after writing selection metadata.
    scores={NAMES[0]:models[NAMES[0]].predict_proba(X.iloc[te])[:,1],
            NAMES[1]:xgb.predict_proba(X.iloc[te])[:,1],NAMES[2]:ae_score(ae_values[te])}
    table=pd.DataFrame([metrics(n,y[te],s,thresholds[n]) for n,s in scores.items()])
    table.to_csv(out/'metrics_table.csv',index=False)
    cr=[]
    for n,s in scores.items():
        if constrained[n] is not None:
            row=metrics(n,y[te],s,constrained[n]);row['Feasible_on_validation']=True
        else:row={'Model':n,'Feasible_on_validation':False}
        cr.append(row)
    pd.DataFrame(cr).to_csv(out/'constrained_metrics.csv',index=False)
    pred=pd.DataFrame({'CONS_NO':ids[te],'y_true':y[te]})
    for n,s in scores.items():pred['score_'+n]=s
    pred.to_csv(out/'test_predictions.csv',index=False)
    original=pd.read_csv('results/metrics_table.csv')
    pd.concat([original.assign(Run='Original'),table.assign(Run='Updated')],ignore_index=True).to_csv(out/'original_vs_updated.csv',index=False)
    pd.DataFrame(history,columns=['train_loss','validation_normal_loss']).to_csv(out/'autoencoder_history.csv',index=False)
    # Verify serialized models and independently rebuilt features on 30 held-out rows.
    reloaded=score_saved(out,raw[te[:30]],dow)
    for n in NAMES:
        np.testing.assert_allclose(reloaded[n],scores[n][:30],atol=1e-6,rtol=1e-5)
    plot_all(out,y[te],scores,thresholds,table,original,history)
    print('\nTEST: BEST-F1 THRESHOLDS\n'+table.to_string(index=False),flush=True)
    print('\nTEST: VALIDATION PRECISION CONSTRAINT\n'+pd.DataFrame(cr).to_string(index=False),flush=True)
    print('Selected on validation AP: '+best,flush=True)
    (out/'verification.json').write_text(json.dumps({'serialization_and_raw_inference_parity':True,'heldout_rows_checked':30,
                                                   'test_split_matches_original':True},indent=2))


if __name__=='__main__':
    main()
