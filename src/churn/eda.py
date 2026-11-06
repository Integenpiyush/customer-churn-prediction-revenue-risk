from pathlib import Path
import pandas as pd, matplotlib.pyplot as plt, seaborn as sns
from scipy.stats import chi2_contingency
from .data import load_dataset
ROOT=Path(__file__).resolve().parents[2]
def run():
 d=load_dataset(ROOT/'data/raw/Telco-Customer-Churn.csv'); d['TotalCharges']=pd.to_numeric(d.TotalCharges.replace(r'^\s*$',pd.NA,regex=True),errors='coerce')
 out=ROOT/'reports/figures'; out.mkdir(parents=True,exist_ok=True); sns.set_theme(style='whitegrid',palette='Set2')
 plots=[('churn_distribution',lambda ax:sns.countplot(data=d,x='Churn',ax=ax)),('contract_churn',lambda ax:sns.barplot(data=d,x='Contract',y=d.Churn.map({'Yes':1,'No':0}),ax=ax)),('tenure_churn',lambda ax:sns.boxplot(data=d,x='Churn',y='tenure',ax=ax)),('monthly_charges_churn',lambda ax:sns.boxplot(data=d,x='Churn',y='MonthlyCharges',ax=ax)),('payment_churn',lambda ax:sns.barplot(data=d,x='PaymentMethod',y=d.Churn.map({'Yes':1,'No':0}),ax=ax))]
 for name,draw in plots:
  fig,ax=plt.subplots(figsize=(9,5)); draw(ax); ax.set_title(name.replace('_',' ').title()); ax.tick_params(axis='x',rotation=25); fig.tight_layout(); fig.savefig(out/f'{name}.png',dpi=160); plt.close(fig)
 numeric=d.select_dtypes('number'); fig,ax=plt.subplots(figsize=(8,6)); sns.heatmap(numeric.corr(),annot=True,cmap='vlag',center=0,ax=ax); fig.tight_layout(); fig.savefig(out/'numeric_correlation.png',dpi=160); plt.close(fig)
 # Cramer's V summarizes pairwise association between categorical variables.
 categorical=d.select_dtypes(exclude='number').drop(columns=['customerID']).columns.tolist()
 associations=[]
 for i,a in enumerate(categorical):
  for b in categorical[i+1:]:
   tab=pd.crosstab(d[a].fillna('Missing'),d[b].fillna('Missing'))
   if min(tab.shape)>1:
    chi2=chi2_contingency(tab,correction=False)[0]; n=tab.to_numpy().sum(); k=min(tab.shape)-1
    associations.append({'feature_a':a,'feature_b':b,'cramers_v':float((chi2/(n*k))**.5)})
 pd.DataFrame(associations).sort_values('cramers_v',ascending=False).to_csv(ROOT/'reports/categorical_association.csv',index=False)
 d['churn_flag']=(d.Churn=='Yes').astype(int)
 segment_cols=['Contract','PaymentMethod','InternetService','tenure','MonthlyCharges']
 segments=[]
 for col in segment_cols:
  if col=='tenure': groups=pd.cut(d.tenure,[-1,6,12,24,48,72],labels=['0-6','7-12','13-24','25-48','49-72'])
  elif col=='MonthlyCharges': groups=pd.cut(d.MonthlyCharges,[-1,35,65,90,200],labels=['low','mid','high','very high'])
  else: groups=d[col]
  for key,part in d.groupby(groups,observed=True): segments.append({'segment_feature':col,'segment':str(key),'customers':len(part),'churn_rate':part.churn_flag.mean()})
 pd.DataFrame(segments).to_csv(ROOT/'reports/churn_segment_rates.csv',index=False)
 outliers=[]
 for col in ['tenure','MonthlyCharges','TotalCharges']:
  q1,q3=d[col].quantile([.25,.75]); iqr=q3-q1; low=q1-1.5*iqr; high=q3+1.5*iqr
  outliers.append({'feature':col,'iqr_lower_bound':low,'iqr_upper_bound':high,'outlier_count':int(((d[col]<low)|(d[col]>high)).sum()),'interpretation':'flagged for review; retained in analysis'})
 pd.DataFrame(outliers).to_csv(ROOT/'reports/outlier_review.csv',index=False)
 summary={'rows':len(d),'columns':len(d.columns),'duplicates':int(d.duplicated().sum()),'missing':d.isna().sum().to_dict(),'totalcharges_blank':int(d.TotalCharges.isna().sum()),'churn_rate':float((d.Churn=='Yes').mean())}
 pd.Series(summary).to_json(ROOT/'reports/eda_summary.json',indent=2)
if __name__=='__main__': run()
