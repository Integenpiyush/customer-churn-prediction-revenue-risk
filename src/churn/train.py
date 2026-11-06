from __future__ import annotations
import json, logging, os
from pathlib import Path
import joblib, numpy as np, pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, confusion_matrix, classification_report, precision_recall_curve, RocCurveDisplay, PrecisionRecallDisplay)
from .features import prepare_frame, clean_target
from .risk import risk_report
from .data import load_dataset

ROOT=Path(__file__).resolve().parents[2]
SEED=42
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

def model_candidates():
    models={
      "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced", C=.5),
      "Random Forest": RandomForestClassifier(n_estimators=350, min_samples_leaf=3, class_weight="balanced_subsample", max_features="sqrt", random_state=SEED, n_jobs=-1),
      "Extra Trees": ExtraTreesClassifier(n_estimators=350, min_samples_leaf=3, class_weight="balanced", max_features="sqrt", random_state=SEED, n_jobs=-1),
    }
    try:
      from xgboost import XGBClassifier
      models["XGBoost"] = XGBClassifier(n_estimators=300, max_depth=3, learning_rate=.035, subsample=.85, colsample_bytree=.85, reg_lambda=2, eval_metric="logloss", random_state=SEED, n_jobs=-1)
    except ImportError: logging.warning("XGBoost unavailable; skipping")
    try:
      from catboost import CatBoostClassifier
      models["CatBoost"] = CatBoostClassifier(iterations=350, depth=5, learning_rate=.04, verbose=False, random_seed=SEED, auto_class_weights="Balanced", allow_writing_files=False)
    except ImportError: logging.warning("CatBoost unavailable; skipping; install requirements to compare")
    return models

def main():
    data_path=ROOT/'data/raw/Telco-Customer-Churn.csv'
    df=load_dataset(data_path)
    if df.shape != (7043,21): logging.warning("Unexpected dataset shape: %s", df.shape)
    y=clean_target(df.Churn)
    X=prepare_frame(df)
    Xtr,Xhold,ytr,yhold=train_test_split(X,y,test_size=.30,random_state=SEED,stratify=y)
    Xv,Xte,yv,yte=train_test_split(Xhold,yhold,test_size=2/3,random_state=SEED,stratify=yhold)
    numeric=X.select_dtypes(include=np.number).columns.tolist()
    categorical=[c for c in X.columns if c not in numeric]
    pre=ColumnTransformer([("num",Pipeline([("impute",SimpleImputer(strategy="median", add_indicator=True)),("scale",StandardScaler())]),numeric),
                           ("cat",Pipeline([("impute",SimpleImputer(strategy="most_frequent")),("onehot",OneHotEncoder(handle_unknown="ignore", min_frequency=2))]),categorical)])
    results=[]; fitted={}
    for name,est in model_candidates().items():
      pipe=Pipeline([("prep",pre),("model",est)])
      if name=="XGBoost":
        search=RandomizedSearchCV(pipe,{"model__n_estimators":[200,300],"model__max_depth":[2,3,4],"model__learning_rate":[.03,.06]},n_iter=4,scoring="f1",cv=3,random_state=SEED,n_jobs=1)
        search.fit(Xtr,ytr); pipe=search.best_estimator_
      else:
        pipe.fit(Xtr,ytr)
      pv=pipe.predict_proba(Xv)[:,1]
      prec,rec,thresholds=precision_recall_curve(yv,pv)
      f=2*prec[:-1]*rec[:-1]/np.maximum(prec[:-1]+rec[:-1],1e-12)
      threshold=float(thresholds[int(np.nanargmax(f))]) if len(thresholds) else .5
      # Keep test set untouched for final comparison; threshold chosen on validation only.
      fitted[name]=(pipe,threshold)
      results.append({"model":name,"validation_f1":float(np.nanmax(f)),"threshold":threshold,"validation_roc_auc":float(roc_auc_score(yv,pv)),"validation_pr_auc":float(average_precision_score(yv,pv)),"best_params":str(search.best_params_) if name=="XGBoost" else "default configuration"})
      logging.info("Fit %s validation F1 %.3f",name,np.nanmax(f))
    best=max(results,key=lambda r:r["validation_f1"])
    for row in results:
      fitted_pipe,cutoff=fitted[row['model']]
      test_probs=fitted_pipe.predict_proba(Xte)[:,1]; test_labels=(test_probs>=cutoff).astype(int)
      row.update({"test_f1":float(f1_score(yte,test_labels)),"test_precision":float(precision_score(yte,test_labels)),"test_recall":float(recall_score(yte,test_labels)),"test_roc_auc":float(roc_auc_score(yte,test_probs)),"test_pr_auc":float(average_precision_score(yte,test_probs))})
    model,threshold=fitted[best['model']]
    p=model.predict_proba(Xte)[:,1]; pred=(p>=threshold).astype(int)
    baseline_p=fitted['Logistic Regression'][0].predict_proba(Xte)[:,1]
    baseline_f1=float(f1_score(yte,(baseline_p>=.5).astype(int)))
    metrics={"selected_model":best['model'],"threshold":threshold,"test_size":len(yte),"accuracy":accuracy_score(yte,pred),"precision":precision_score(yte,pred),"recall":recall_score(yte,pred),"f1":f1_score(yte,pred),"roc_auc":roc_auc_score(yte,p),"pr_auc":average_precision_score(yte,p),"confusion_matrix":confusion_matrix(yte,pred).tolist(),"classification_report":classification_report(yte,pred,output_dict=True),"baseline_logistic_test_f1_at_0_5":baseline_f1,"selected_vs_baseline_f1_relative_change":float(f1_score(yte,pred)/baseline_f1-1)}
    # CV is measured on training partition only, with shared preprocessing in each fold.
    cv=StratifiedKFold(n_splits=5,shuffle=True,random_state=SEED)
    metrics['selected_model_cv_f1_mean']=float(cross_val_score(fitted[best['model']][0],Xtr,ytr,cv=cv,scoring='f1',n_jobs=1).mean())
    # Persist deployed pipeline and split test predictions for reproducible app/report.
    joblib.dump(model,ROOT/'artifacts/churn_model.joblib')
    with open(ROOT/'artifacts/metrics.json','w') as f: json.dump(metrics,f,indent=2)
    pd.DataFrame(results).sort_values('validation_f1',ascending=False).to_csv(ROOT/'reports/model_comparison.csv',index=False)
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(7,5)); RocCurveDisplay.from_predictions(yte,p,ax=ax); fig.tight_layout(); fig.savefig(ROOT/'reports/figures/roc_curve.png',dpi=160); plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,5)); PrecisionRecallDisplay.from_predictions(yte,p,ax=ax); fig.tight_layout(); fig.savefig(ROOT/'reports/figures/precision_recall_curve.png',dpi=160); plt.close(fig)
    testraw=df.loc[Xte.index].copy()
    ranked,top=risk_report(testraw,p)
    if os.getenv('EXPORT_CUSTOMER_LEVEL_REPORTS') == '1':
      private=ROOT/'data/private'; private.mkdir(parents=True,exist_ok=True)
      ranked.to_csv(private/'test_customer_risk.csv',index=False)
    risk={"test_customer_count":len(testraw),"annualized_expected_revenue_risk":float(ranked.expected_revenue_at_risk.sum()),"top_decile_count":len(top),"top_decile_risk":float(top.expected_revenue_at_risk.sum()),"top_decile_share":float(top.expected_revenue_at_risk.sum()/ranked.expected_revenue_at_risk.sum()),"actual_churners":int(yte.sum())}
    with open(ROOT/'artifacts/revenue_risk.json','w') as f: json.dump(risk,f,indent=2)
    try:
      from .explain import explain
      import matplotlib.pyplot as plt, shap
      sample=Xte.sample(min(250,len(Xte)),random_state=SEED)
      names,values=explain(model,sample,background=Xtr.sample(min(100,len(Xtr)),random_state=SEED))
      vals=values.values
      if vals.ndim==3: vals=vals[:,:,1]
      importance=np.abs(vals).mean(axis=0)
      pd.DataFrame({'feature':names,'mean_abs_shap':importance}).sort_values('mean_abs_shap',ascending=False).to_csv(ROOT/'reports/shap_global_importance.csv',index=False)
      shap.plots.beeswarm(values,max_display=20,show=False); plt.gcf().set_size_inches(10,7); plt.tight_layout(); plt.savefig(ROOT/'reports/figures/shap_summary.png',dpi=160); plt.close()
      shap.plots.bar(values,max_display=20,show=False); plt.gcf().set_size_inches(10,7); plt.tight_layout(); plt.savefig(ROOT/'reports/figures/shap_global_importance.png',dpi=160); plt.close()
    except Exception:
      logging.exception('SHAP artifact generation failed; interactive SHAP remains available for compatible runtimes')
    logging.info("Test metrics: %s",metrics)
    logging.info("Risk metrics: %s",risk)

if __name__=='__main__': main()
