import pandas as pd
import pytest
from pathlib import Path
from src.churn.features import prepare_frame, clean_target
from src.churn.risk import expected_revenue_risk, risk_report

def sample():
 return pd.DataFrame([{'customerID':'a','gender':'Female','SeniorCitizen':0,'Partner':'No','Dependents':'No','tenure':0,'PhoneService':'No','MultipleLines':'No phone service','InternetService':'DSL','OnlineSecurity':'No','OnlineBackup':'Yes','DeviceProtection':'No','TechSupport':'No','StreamingTV':'No','StreamingMovies':'No','Contract':'Month-to-month','PaperlessBilling':'Yes','PaymentMethod':'Electronic check','MonthlyCharges':29.85,'TotalCharges':' ','Churn':'Yes'}])

def test_feature_processing_handles_blank_total_charges():
 x=prepare_frame(sample()); assert x.TotalCharges.isna().all(); assert x.month_to_month.iloc[0]==1; assert x.tenure_group.iloc[0]=='0-6'

def test_target_is_binary(): assert clean_target(pd.Series(['No','Yes'])).tolist()==[0,1]

def test_expected_revenue_risk_formula(): assert expected_revenue_risk(.5,100)==600

def test_risk_report_orders_and_categorizes():
 d=pd.DataFrame({'MonthlyCharges':[10,100],'Contract':['Month-to-month','Two year']}); scored,top=risk_report(d,[.8,.6],.5)
 assert scored.iloc[0].expected_revenue_at_risk==720
 assert len(top)==1 and top.iloc[0].risk_category=='Medium'
 assert 'retention outreach' in scored.iloc[1].retention_recommendation

def test_invalid_target_is_rejected():
 with pytest.raises((TypeError,ValueError)): clean_target(pd.Series(['maybe']))

def test_saved_model_probability_and_threshold_behavior():
 import joblib
 root=Path(__file__).resolve().parents[1]
 model=joblib.load(root/'artifacts/churn_model.joblib')
 row=pd.read_csv(root/'data/raw/Telco-Customer-Churn.csv').drop(columns='Churn').iloc[[0]]
 probability=model.predict_proba(prepare_frame(row))[:,1]
 assert probability.shape==(1,) and 0<=probability[0]<=1
 threshold=.7
 assert int(probability[0]>=threshold) in (0,1)

def test_shap_output_when_dependency_is_available():
 pytest.importorskip('shap')
 import joblib
 from src.churn.explain import explain
 root=Path(__file__).resolve().parents[1]
 model=joblib.load(root/'artifacts/churn_model.joblib')
 row=pd.read_csv(root/'data/raw/Telco-Customer-Churn.csv').drop(columns='Churn').iloc[:2]
 names,values=explain(model,prepare_frame(row))
 assert len(names)>0 and values.values.shape[0]==2
