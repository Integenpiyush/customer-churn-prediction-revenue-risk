from pathlib import Path
import json, joblib, pandas as pd, streamlit as st
from src.churn.features import prepare_frame
ROOT=Path(__file__).resolve().parent
st.set_page_config(page_title='Churn & Revenue Risk',page_icon='📉',layout='wide')
st.markdown('<style>.block-container{padding-top:1.5rem}</style>',unsafe_allow_html=True)
@st.cache_resource
def load(): return joblib.load(ROOT/'artifacts/churn_model.joblib')
@st.cache_data
def metrics(): return json.loads((ROOT/'artifacts/metrics.json').read_text())
if not (ROOT/'artifacts/churn_model.joblib').exists(): st.error('Model artifact missing. Run `python -m src.churn.train` first.'); st.stop()
model=load(); m=metrics(); page=st.sidebar.radio('Navigate',['Overview','Customer Analytics','Model Performance','Customer Prediction','Revenue Risk','Explainability'])
st.title('Customer Churn Prediction & Revenue Risk')
if page=='Overview':
 risk=json.loads((ROOT/'artifacts/revenue_risk.json').read_text()); eda=json.loads((ROOT/'reports/eda_summary.json').read_text()); c=st.columns(4)
 c[0].metric('Customers',f'{eda["rows"]:,}'); c[1].metric('Observed churn rate',f'{eda["churn_rate"]:.1%}'); c[2].metric('Test F1',f'{m["f1"]:.3f}'); c[3].metric('Test ROC-AUC',f'{m["roc_auc"]:.3f}')
 st.subheader('Model performance'); st.write(f"Selected **{m['selected_model']}** at validation-selected threshold **{m['threshold']:.3f}**. Test-set revenue-risk proxy: **${risk['annualized_expected_revenue_risk']:,.0f}**.")
 st.caption('Annualized proxy = churn probability × monthly charges × 12; it is not guaranteed lost revenue or customer lifetime value.')
 st.image(str(ROOT/'reports/figures/churn_distribution.png'),width=500)
elif page=='Customer Analytics':
 st.subheader('Observed customer patterns'); tabs=st.tabs(['Contract','Tenure','Monthly charges','Payment','Correlation'])
 for tab,name in zip(tabs,['contract_churn','tenure_churn','monthly_charges_churn','payment_churn','numeric_correlation']):
  with tab: st.image(str(ROOT/f'reports/figures/{name}.png'),use_container_width=True)
elif page=='Model Performance':
 st.json(m); st.dataframe(pd.read_csv(ROOT/'reports/model_comparison.csv'),use_container_width=True)
 st.metric('Validation-selected threshold',f"{m['threshold']:.3f}")
 st.subheader('Confusion matrix at selected threshold'); st.dataframe(pd.DataFrame(m['confusion_matrix'],index=['Actual no churn','Actual churn'],columns=['Predicted no churn','Predicted churn']))
 c=st.columns(2); c[0].image(str(ROOT/'reports/figures/roc_curve.png'),use_container_width=True); c[1].image(str(ROOT/'reports/figures/precision_recall_curve.png'),use_container_width=True)
 st.caption('Threshold was selected on validation to maximize F1; no customer-level test predictions are published in this dashboard.')
elif page=='Customer Prediction':
 with st.form('prediction'):
  cols=st.columns(3); gender=cols[0].selectbox('Gender',['Female','Male']); senior=cols[1].selectbox('Senior citizen',[0,1]); partner=cols[2].selectbox('Partner',['Yes','No'])
  depend=cols[0].selectbox('Dependents',['Yes','No']); tenure=cols[1].number_input('Tenure (months)',0,72,12); phone=cols[2].selectbox('Phone service',['Yes','No'])
  multiple=cols[0].selectbox('Multiple lines',['No phone service','No','Yes']); internet=cols[1].selectbox('Internet service',['DSL','Fiber optic','No']); contract=cols[2].selectbox('Contract',['Month-to-month','One year','Two year'])
  service={}
  for i,label in enumerate(['OnlineSecurity','OnlineBackup','DeviceProtection','TechSupport','StreamingTV','StreamingMovies']):
   service[label]=cols[i%3].selectbox(label,['No internet service','No','Yes'])
  paperless=cols[0].selectbox('Paperless billing',['Yes','No']); payment=cols[1].selectbox('Payment method',['Electronic check','Mailed check','Bank transfer (automatic)','Credit card (automatic)']); monthly=cols[2].number_input('Monthly charges',0.0,200.0,70.0)
  total=st.number_input('Total charges',0.0,10000.0,float(monthly*tenure)); submitted=st.form_submit_button('Predict churn')
 if submitted:
  row={'customerID':'CUSTOMER_INPUT','gender':gender,'SeniorCitizen':senior,'Partner':partner,'Dependents':depend,'tenure':tenure,'PhoneService':phone,'MultipleLines':multiple,'InternetService':internet,'Contract':contract,'PaperlessBilling':paperless,'PaymentMethod':payment,'MonthlyCharges':monthly,'TotalCharges':total,'Churn':'No',**service}
  prob=float(model.predict_proba(prepare_frame(pd.DataFrame([row])))[:,1][0]); st.metric('Churn probability',f'{prob:.1%}'); st.write('Risk category:', 'High' if prob>=.6 else 'Medium' if prob>=.3 else 'Low'); st.caption('Associations support prioritization; they do not establish causes.')
  try:
   from src.churn.explain import explain
   import shap, matplotlib.pyplot as plt
   names,values=explain(model,prepare_frame(pd.DataFrame([row])))
   shap.plots.waterfall(values[0],max_display=12,show=False); st.pyplot(plt.gcf()); plt.close()
   local=pd.DataFrame({'feature':names,'contribution':values.values[0]}).assign(magnitude=lambda d:d.contribution.abs()).sort_values('magnitude',ascending=False).head(12)
   st.dataframe(local.drop(columns='magnitude'),use_container_width=True)
  except Exception as e: st.info(f'Local SHAP explanation unavailable: {e}')
elif page=='Revenue Risk':
 risk=json.loads((ROOT/'artifacts/revenue_risk.json').read_text()); c=st.columns(3)
 c[0].metric('Test-set annualized revenue risk',f"${risk['annualized_expected_revenue_risk']:,.0f}"); c[1].metric('Top 10% risk captured',f"{risk['top_decile_share']:.1%}"); c[2].metric('Customers in priority decile',f"{risk['top_decile_count']:,}")
 segments=pd.read_csv(ROOT/'reports/risk_segment_summary.csv')
 for segment in ['Contract','tenure_group','PaymentMethod','InternetService','customer_segment']:
  st.subheader(f'Risk by {segment}'); st.dataframe(segments[segments.segment_type==segment].drop(columns='segment_type').sort_values('annualized_expected_revenue_risk',ascending=False),use_container_width=True)
 st.caption('Customer-level risk rankings and downloadable reports stay local; this public dashboard shows segment totals only.')
elif page=='Explainability':
 st.write('SHAP values measure each feature contribution relative to background data. They explain model output, not causation.')
 try:
  from src.churn.explain import explain
  import shap, matplotlib.pyplot as plt
  st.image(str(ROOT/'reports/figures/shap_global_importance.png'),use_container_width=True)
  st.image(str(ROOT/'reports/figures/shap_summary.png'),use_container_width=True)
  st.dataframe(pd.read_csv(ROOT/'reports/shap_global_importance.csv').head(20),use_container_width=True)
  st.info('Use the Customer Prediction page to generate a local SHAP explanation for an entered profile.')
 except Exception as e: st.info(f'SHAP is unavailable or incompatible in this runtime: {e}')
