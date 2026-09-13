from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable
import json
import numpy as np
import pandas as pd
from scipy.special import expit

PHI_COLUMNS = [
    "φ1 Planetary Health",
    "φ2 Sustainable Prosperity",
    "φ3 Inclusive Human Well-being",
    "φ4 Institutional Effectiveness",
    "φ5 Transformative Capacity",
    # "φ6 Cultural-Ethical Alignment",
    "φ7 Adaptive Capacity",
]

@dataclass(frozen=True)
class Parameters:
    beta_weights: tuple[float, ...] = (1/6,)*6
    theta_omega: float = 0.60 #0.70
    gamma_omega: float = 8.0
    lambda_co2: float = 0.10
    epsilon: float = 0.01
    lambda_f: float = 0.20
    theta4: float = 0.65
    gamma4: float = 10.0
    tau: float = 4.0
    alpha: float = 0.40
    theta7: float = 0.65
    foundational_goals: tuple[int, ...] = (2,3,6,9,11)

    @classmethod
    def from_json(cls, path: str | Path) -> "Parameters":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        d.pop("gmm", None)
        d["beta_weights"] = tuple(d["beta_weights"])
        d["foundational_goals"] = tuple(d["foundational_goals"])
        return cls(**d)

    def updated(self, **kwargs) -> "Parameters":
        return replace(self, **kwargs)


def _available(row: np.ndarray, goals: Iterable[int]) -> np.ndarray:
    idx = np.asarray(list(goals), dtype=int) - 1
    x = row[idx]
    return x[~np.isnan(x)]


def _safe_mean(x: np.ndarray) -> float:
    return float(np.mean(x)) if len(x) else np.nan


def compute_country_spectrum(row: np.ndarray, p: Parameters) -> dict[str, float]:
    x = _available(row, (6,13,14,15))
    phi1 = _safe_mean(np.log1p(x))
    if not np.isnan(row[12]):
        phi1 += p.lambda_co2 * row[12]

    # φ2: geometric mean plus small prosperity-alignment correction.
    x = _available(row, (7,8,9,12))
    phi2 = float(np.exp(np.mean(np.log(x)))) if len(x) else np.nan
    g89 = _available(row, (8,9))
    if len(g89):
        denom = 1.0
        if not np.isnan(row[6]) and not np.isnan(row[11]):
            denom += abs(row[6] - row[11])
        phi2 += p.epsilon * float(np.max(g89)) / denom
    phi2 = float(np.clip(phi2, 0.0, 1.0))

    # φ3: social mean minus fairness imbalance penalty.
    phi3 = _safe_mean(_available(row, (1,2,3,4,5,10)))
    foundational = _available(row, (1,2,5))
    higher_order = _available(row, (3,4,10))
    if len(foundational) and len(higher_order):
        phi3 -= p.lambda_f * abs(float(np.mean(foundational)) - float(np.mean(higher_order)))
    phi3 = float(np.clip(phi3, 0.0, 1.0))

    # φ4: institutional sigmoid.
    x4 = _safe_mean(_available(row, (16,17)))
    phi4 = float(expit(p.gamma4 * (x4 - p.theta4)))

    # φ5: innovation signal penalized by system-wide SDG variance.
    variance = float(np.nanvar(row, ddof=0))
    phi5 = float(row[8] * np.exp(-p.tau * variance)) if not np.isnan(row[8]) else np.nan

    # φ6: coordination proxy based on population standard deviation.
    phi6 = float(1.0 / (1.0 + np.std(_available(row, (4,11,16)), ddof=0)))

    # φ7: weakest link plus proportion of foundational goals above threshold.
    bottleneck = float(np.min(_available(row, (11,13))))
    c = _available(row, p.foundational_goals)
    robustness = float(np.mean(c >= p.theta7)) if len(c) else np.nan
    phi7 = p.alpha * bottleneck + (1.0 - p.alpha) * robustness

    phi = np.array([phi1,phi2,phi3,phi4,phi5,phi7], dtype=float)
    weights = np.asarray(p.beta_weights, dtype=float)
    valid = ~np.isnan(phi)
    weights = weights[valid] / weights[valid].sum()
    omega_input = float(np.sum(phi[valid] * weights))
    omega = float(expit(p.gamma_omega * (omega_input - p.theta_omega)))

    return {
        "Ω₆": omega,
        "Ω input mean(φ)": omega_input,
        PHI_COLUMNS[0]: phi1,
        PHI_COLUMNS[1]: phi2,
        PHI_COLUMNS[2]: phi3,
        PHI_COLUMNS[3]: phi4,
        PHI_COLUMNS[4]: phi5,
        #PHI_COLUMNS[5]: phi6,
        PHI_COLUMNS[5]: phi7,
        "Var(G1:G17)": variance,
        "Institutional input x4": x4,
    }


def load_sdr2025(path: str | Path) -> pd.DataFrame:
    df = pd.read_excel(path)
    required = ["Country Code ISO3", "Country", "2026 SDG Index Score", "2026 SDG Index Rank", "Regions used for the SDR"] + [f"Goal {i} Score" for i in range(1,18)]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    return df


def compute_spectra(source: pd.DataFrame, p: Parameters) -> pd.DataFrame:
    goal_cols = [f"Goal {i} Score" for i in range(1,18)]
    G = source[goal_cols].to_numpy(dtype=float) / 100.0
    rows = [compute_country_spectrum(row, p) for row in G]
    score = pd.DataFrame(rows)
    out = pd.DataFrame({
        "ISO3": source["Country Code ISO3"].astype(str),
        "Country": source["Country"],
        "SDR Region": source["Regions used for the SDR"],
        "SDSN Index Score": source["2026 SDG Index Score"],
        "SDSN Index Rank": source["2026 SDG Index Rank"].astype(int),
    })
    out = pd.concat([out, score], axis=1)
    out["% Missing Values"] = source.get("Percentage missing values", np.nan)
    out["Ω Rank"] = out["Ω₆"].rank(method="min", ascending=False).astype(int)
    out["ΔRank (Ω - SDSN)"] = out["Ω Rank"] - out["SDSN Index Rank"]
    return out


def summary_statistics(scores: pd.DataFrame) -> pd.DataFrame:
    cols = ["Ω₆", "Ω input mean(φ)"] + PHI_COLUMNS
    labels = ["Ω₆", "Mean φ input to Ω"] + PHI_COLUMNS
    rows=[]
    for label, col in zip(labels, cols):
        s=scores[col]
        rows.append([label,int(s.notna().sum()),float(s.mean()),float(s.std(ddof=1)),float(s.min()),float(s.max())])
    return pd.DataFrame(rows, columns=["Variable","N","Mean","Std. Dev.","Minimum","Maximum"])


def parameter_table(p: Parameters) -> pd.DataFrame:
    return pd.DataFrame([
        ["β_k","Final integration Ω₆","1/6","0.10–0.20 or ±30% and renormalized","Equal baseline weighting."],
        ["θ_Ω","Final integration Ω₆",p.theta_omega,"0.60–0.75","Near empirical median of mean national SDG performance."],
        ["γ_Ω","Final integration Ω₆",p.gamma_omega,"5–12","Moderate sigmoid steepness."],
        ["λ_CO2","φ1 Planetary Health",p.lambda_co2,"0–0.20","Modest additional weighting for climate action."],
        ["ε","φ2 Sustainable Prosperity",p.epsilon,"0.001–0.05","Numerical stability and small alignment correction."],
        ["λ_F","φ3 Inclusive Human Well-being",p.lambda_f,"0.10–0.30","Moderate fairness penalty."],
        ["θ4","φ4 Institutional Effectiveness",p.theta4,"0.55–0.75","Near empirical median of (G16+G17)/2."],
        ["γ4","φ4 Institutional Effectiveness",p.gamma4,"5–15","Threshold-like but smooth activation."],
        ["τ","φ5 Transformative Capacity",p.tau,"2.6–5.6","Q75 system variance gives about 15% reduction."],
        ["α","φ7 Adaptive Capacity",p.alpha,"0.30–0.70","Balances bottleneck and broader robustness."],
        ["θ7","φ7 Adaptive Capacity",p.theta7,"0.55–0.75","Adequacy threshold for foundational goals."],
        ["C","φ7 Adaptive Capacity",str(set(p.foundational_goals)),"Alternative sets","Foundational goal set."],
    ], columns=["Parameter","Formula / dimension","Baseline value","Sensitivity range","Rationale / note"])


def formula_notes() -> pd.DataFrame:
    return pd.DataFrame([
        ["Input scaling","Goal-level scores divided by 100 to obtain G_i in [0,1]."],
        ["Missing scores","No extra imputation; formulas use available constituent goals only."],
        ["Variance / SD","Population variance and population standard deviation."],
        ["φ1","mean[ln(1+G6), ln(1+G13), ln(1+G14), ln(1+G15)] + λ_CO2×G13."],
        ["φ2","geometric_mean(G7,G8,G9,G12) + ε×max(G8,G9)/(1+|G7−G12|), capped to [0,1]."],
        ["φ3","mean(G1,G2,G3,G4,G5,G10) − λ_F×|mean(G1,G2,G5)−mean(G3,G4,G10)|."],
        ["φ4","sigmoid(mean(G16,G17); θ4, γ4)."],
        ["φ5","G9×exp(−τ×Var(G1,…,G17))."],
        ["φ6","1/(1+SD(G4,G11,G16))."],
        ["φ7","α×min(G11,G13)+(1−α)×proportion of C above θ7."],
        ["Ω₆","sigmoid(weighted mean of φ1–φ5 and φ7; θΩ, γΩ)."],
    ], columns=["Item","Implementation"])
