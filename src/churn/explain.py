"""SHAP utilities; explanations describe model behavior, not causal effects."""
def explain(model, X, background=None):
    import shap
    prep=model.named_steps['prep']; estimator=model.named_steps['model']
    transformed=prep.transform(X); names=prep.get_feature_names_out()
    background_t=prep.transform(background) if background is not None else None
    if estimator.__class__.__module__.startswith('catboost'):
        transformed=transformed.toarray() if hasattr(transformed,'toarray') else transformed
        if background_t is not None: background_t=background_t.toarray() if hasattr(background_t,'toarray') else background_t
    try:
        explainer=shap.TreeExplainer(estimator) if background_t is None else shap.TreeExplainer(estimator,data=background_t,feature_perturbation='interventional')
    except Exception: explainer=shap.Explainer(estimator,background_t,feature_names=names)
    return names,explainer(transformed)
