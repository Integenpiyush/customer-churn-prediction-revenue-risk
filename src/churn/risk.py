import pandas as pd

def expected_revenue_risk(probability, monthly_charges):
    return probability * monthly_charges * 12

def risk_report(frame: pd.DataFrame, probabilities, top_fraction=.10):
    out = frame.copy().reset_index(drop=True)
    out["churn_probability"] = probabilities
    out["expected_revenue_at_risk"] = expected_revenue_risk(out.churn_probability, out.MonthlyCharges)
    out["risk_category"] = pd.cut(out.churn_probability, [-.001, .3, .6, 1.001], labels=["Low", "Medium", "High"])
    def recommendation(row):
        if row.risk_category == "High" and row.get("Contract") == "Month-to-month":
            return "Priority retention outreach; review service fit and renewal options"
        if row.risk_category == "High" and row.get("PaymentMethod") == "Electronic check":
            return "Priority retention outreach; offer billing and payment support"
        if row.get("tenure", 99) <= 6:
            return "Early-tenure check-in; offer onboarding and setup support"
        if row.risk_category == "High":
            return "Priority personalized retention outreach"
        if row.risk_category == "Medium":
            return "Targeted engagement; monitor account signals"
        return "Routine engagement and service updates"
    out["retention_recommendation"] = out.apply(recommendation, axis=1)
    n = max(1, int(len(out) * top_fraction))
    top = out.nlargest(n, "expected_revenue_at_risk")
    return out.sort_values("expected_revenue_at_risk", ascending=False), top
