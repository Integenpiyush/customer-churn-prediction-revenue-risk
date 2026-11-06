# Dataset and data handling

The application uses IBM's Telco Customer Churn sample CSV from the [IBM telco-customer-churn-on-icp4d repository](https://github.com/IBM/telco-customer-churn-on-icp4d) (direct file: `data/Telco-Customer-Churn.csv`). The repository describes the file as a sample dataset for its churn prediction code pattern. IBM's repository does not publish a standalone license for the CSV; this project therefore records the upstream source and makes no independent license grant. The CSV is downloaded locally with `python -m src.churn.data` or fetched from IBM when an app starts without a local copy. Confirm reuse terms with IBM before commercial redistribution.

The upstream source file has 7,043 records and 21 columns including customer ID and target. `TotalCharges` has 11 blank strings, which represent customers at zero tenure; they are parsed to missing values and median-imputed inside the training pipeline. The target is `Churn` (`Yes`/`No`). No customer rows are fabricated, and the CSV is not redistributed in this repository.

## Data quality and splits

`customerID` is excluded from predictors. The split is stratified and deterministic (seed 42): 70% training, 10% validation, and 20% final test. Threshold/model selection uses validation only; test metrics are calculated after selection. Preprocessing is fit inside sklearn pipelines on training folds, avoiding imputation/encoding leakage. Numeric values are median-imputed and standardized; categoricals are mode-imputed and one-hot encoded with unknown-category tolerance. Derived features are generated identically for training and inference.
