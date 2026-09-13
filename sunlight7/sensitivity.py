from __future__ import annotations

import pandas as pd
from scipy.stats import spearmanr
from .core import Parameters, compute_spectra
from .cluster import ClusterResult, classify_with_baseline

SCENARIOS = [
    ("Scenario 1","S1","Lower λ_CO2",{"lambda_co2":0.00},"Tests weaker climate weighting in φ1."),
    ("Scenario 2","S2","Higher λ_CO2",{"lambda_co2":0.20},"Tests stronger climate weighting in φ1."),
    ("Scenario 3","S3","Lower λ_F",{"lambda_f":0.10},"Tests weaker fairness penalty in φ3."),
    ("Scenario 4","S4","Higher λ_F",{"lambda_f":0.30},"Tests stronger fairness penalty in φ3."),
    ("Scenario 5","S5","Lower θ4, θ7",{"theta4":0.55,"theta7":0.55},"Tests lower activation / adequacy thresholds."),
    ("Scenario 6","S6","Higher θ4, θ7",{"theta4":0.75,"theta7":0.75},"Tests higher activation / adequacy thresholds."),
    ("Scenario 7","S7","Lower γ4",{"gamma4":5.0},"Tests a flatter institutional sigmoid."),
    ("Scenario 8","S8","Higher γ4",{"gamma4":15.0},"Tests a steeper institutional sigmoid."),
    ("Scenario 9","S9","Lower τ",{"tau":2.6},"Tests a weaker variance effect in φ5."),
    ("Scenario 10","S10","Higher τ",{"tau":5.6},"Tests a stronger variance effect in φ5."),
    ("Scenario 11","S11","Lower α",{"alpha":0.30},"Tests lower weight on the bottleneck term in φ7."),
    ("Scenario 12","S12","Higher α",{"alpha":0.70},"Tests higher weight on the bottleneck term in φ7."),
    ("Scenario 13","S13","Lower θ_Ω",{"theta_omega":0.55},"Tests a more permissive activation threshold for the final integrated score."),
    ("Scenario 14","S14","Higher θ_Ω",{"theta_omega":0.65},"Tests a more stringent activation threshold for the final integrated score."),
]


def run_sensitivity(source: pd.DataFrame, baseline_scores: pd.DataFrame, p: Parameters, baseline_gmm: ClusterResult):
    base_class = classify_with_baseline(baseline_scores, baseline_gmm)
    base_name = base_class.set_index('ISO3')['Assigned archetype']
    country_rows=[]
    base_tmp=base_class.copy(); base_tmp.insert(0,'Scenario','Baseline'); country_rows.append(base_tmp)
    summary=[]
    for scenario, short, modified, changes, note in SCENARIOS:
        score=compute_spectra(source,p.updated(**changes))
        cls=classify_with_baseline(score,baseline_gmm)
        cls.insert(0,'Scenario',scenario)
        country_rows.append(cls)
        names=cls.set_index('ISO3')['Assigned archetype']
        unchanged=float((names.loc[base_name.index]==base_name).mean()*100)
        rho=float(spearmanr(score['Ω₆'],baseline_scores['Ω₆']).statistic)
        dmean=float(score['Ω₆'].mean()-baseline_scores['Ω₆'].mean())
        summary.append([scenario,short,modified,rho,unchanged,dmean,note])
    summary_df=pd.DataFrame(summary,columns=['Scenario','Scenario short','Modified parameter(s)','Rank correlation with baseline','Share of unchanged archetype assignments (%)','Change in mean Ω₆','Notes'])
    country_df=pd.concat(country_rows,ignore_index=True)
    table=pd.concat([pd.DataFrame([["Baseline","—",1.0,100.0,0.0,"Reference configuration."]],columns=['Scenario','Modified parameter(s)','Rank correlation with baseline','Share of unchanged archetype assignments (%)','Change in mean Ω₆','Notes']), summary_df.drop(columns='Scenario short')],ignore_index=True)
    return table, summary_df, country_df
