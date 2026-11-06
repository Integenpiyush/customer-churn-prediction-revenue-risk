from pathlib import Path
import json, joblib, pandas as pd, streamlit as st
from src.churn.features import prepare_frame
from src.churn.risk import risk_report
from src.churn.data import load_dataset
ROOT=Path(__file__).resolve().parent
st.set_page_config(page_title='Churn & Revenue Risk',page_icon='📉',layout='wide')
st.markdown('<style>.block-container{padding-top:1.5rem}</style>',unsafe_allow_html=True)
@st.cache_resource
def load(): return joblib.load(ROOT/'artifacts/churn_model.joblib')
@st.cache_data
def data(): return load_dataset(ROOT/'data/raw/Telco-Customer-Churn.csv')
@st.cache_data
def metrics(): return json.loads((ROOT/'artifacts/metrics.json').read_text())
if not (ROOT/'artifacts/churn_model.joblib').exists(): st.error('Model artifact missing. Run `python -m src.churn.train` first.'); st.stop()
model=load(); df=data(); m=metrics(); page=st.sidebar.radio('Navigate',['Overview','Customer Analytics','Model Performance','Customer Prediction','Revenue Risk','Explainability'])
st.title('Customer Churn Prediction & Revenue Risk')
if page=='Overview':
 risk=json.loads((ROOT/'artifacts/revenue_risk.json').read_text()); c=st.columns(4)
 c[0].metric('Customers',f'{len(df):,}'); c[1].metric('Observed churn rate',f'{(df.Churn=="Yes").mean():.1%}'); c[2].metric('Test F1',f'{m["f1"]:.3f}'); c[3].metric('Test ROC-AUC',f'{m["roc_auc"]:.3f}')
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
elif page=='Revenue Risk':
 scored=model.predict_proba(prepare_frame(df))[:,1]; ranked,top=risk_report(df,scored)
 c=st.columns(3); c[0].metric('Portfolio expected annualized revenue risk',f"${ranked.expected_revenue_at_risk.sum():,.0f}"); c[1].metric('Top 10% risk captured',f"{top.expected_revenue_at_risk.sum()/ranked.expected_revenue_at_risk.sum():.1%}"); c[2].metric('Customers in priority decile',f'{len(top):,}')
 ranked['tenure_group']=pd.cut(ranked.tenure,[-1,6,12,24,48,72],labels=['0-6','7-12','13-24','25-48','49-72'])
 ranked['monthly_charge_segment']=pd.cut(ranked.MonthlyCharges,[-1,35,65,90,200],labels=['low','mid','high','very high'])
 ranked['customer_segment']=ranked.tenure_group.astype(str)+' tenure / '+ranked.monthly_charge_segment.astype(str)+' monthly charge'
 for segment in ['Contract','tenure_group','PaymentMethod','InternetService','customer_segment']:
  st.subheader(f'Risk by {segment}'); st.dataframe(ranked.groupby(segment,observed=True).expected_revenue_at_risk.sum().sort_values(ascending=False).rename('annualized risk').reset_index(),use_container_width=True)
 st.caption('Customer-level risk rankings and downloadable reports stay local; this public dashboard shows segment totals only.')
elif page=='Explainability':
 st.write('SHAP values measure each feature contribution relative to background data. They explain model output, not causation.')
 try:
  from src.churn.explain import explain
  import shap, matplotlib.pyplot as plt
  sample=df.sample(min(150,len(df)),random_state=42); names,values=explain(model,prepare_frame(sample))
  shap.plots.bar(values,max_display=15,show=False); st.pyplot(plt.gcf()); plt.close()
  shap.plots.beeswarm(values,max_display=15,show=False); st.pyplot(plt.gcf()); plt.close()
  i=st.number_input('Customer row in sample',0,len(sample)-1,0); shap.plots.waterfall(values[int(i)],max_display=12,show=False); st.pyplot(plt.gcf()); plt.close()
  top_feature=str(values.feature_names[int(abs(values.values).mean(axis=0).argmax())]); st.caption(f'Top global encoded feature: {top_feature}. SHAP attribution is not causal.')
  shap.plots.scatter(values[:,top_feature],show=False); st.pyplot(plt.gcf()); plt.close()
  local=pd.DataFrame({'feature':names,'contribution':values.values[int(i)]}).assign(magnitude=lambda d:d.contribution.abs()).sort_values('magnitude',ascending=False).head(12)
  st.subheader('Largest local contributions'); st.dataframe(local.drop(columns='magnitude'),use_container_width=True)
 except Exception as e: st.info(f'SHAP is unavailable or incompatible in this runtime: {e}')
