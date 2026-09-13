from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

from .core import PHI_COLUMNS

@dataclass
class ClusterResult:
    scaler: StandardScaler
    model: GaussianMixture
    raw_to_name: dict[int,str]
    assignments: pd.DataFrame


def semantic_label_mapping(scores: pd.DataFrame, raw_labels: np.ndarray) -> dict[int,str]:
    means = scores.assign(_raw=raw_labels).groupby('_raw')[PHI_COLUMNS].mean()
    overall = means.mean(axis=1)
    balanced = int(overall.idxmax())
    laggards = int(overall.idxmin())
    remaining = [int(x) for x in means.index if int(x) not in (balanced,laggards)]
    economy = min(remaining, key=lambda k: means.loc[k, PHI_COLUMNS[0]])
    environment = [k for k in remaining if k != economy][0]
    return {
        balanced: "Balanced Achievers",
        economy: "Economy–Innovation Driven",
        environment: "Planetary–Social Intermediate",
        laggards: "Multidimensional Laggards",
    }


def fit_baseline_gmm(scores_source_order: pd.DataFrame, gmm_cfg: dict) -> ClusterResult:
    X = scores_source_order[PHI_COLUMNS].to_numpy(dtype=float)
    scaler = StandardScaler()
    Z = scaler.fit_transform(X)
    model = GaussianMixture(
        n_components=int(gmm_cfg.get('n_components',4)),
        covariance_type=gmm_cfg.get('covariance_type','spherical'),
        random_state=int(gmm_cfg.get('random_state',42)),
        n_init=int(gmm_cfg.get('n_init',50)),
        max_iter=int(gmm_cfg.get('max_iter',1000)),
        tol=float(gmm_cfg.get('tol',0.001)),
    )
    raw = model.fit_predict(Z)
    posterior = model.predict_proba(Z)
    pmax = posterior.max(axis=1)
    mapping = semantic_label_mapping(scores_source_order, raw)
    out = scores_source_order.copy()
    out['Archetype'] = pd.Series(raw).map(mapping).to_numpy()
    id_map = {"Balanced Achievers":1,"Economy–Innovation Driven":2,"Planetary–Social Intermediate":3,"Multidimensional Laggards":4}
    out['Archetype ID'] = out['Archetype'].map(id_map)
    out['Posterior probability'] = pmax
    out['Classification uncertainty'] = 1.0-pmax
    return ClusterResult(scaler,model,mapping,out)


def gmm_diagnostics(scores_source_order: pd.DataFrame, gmm_cfg: dict, quick: bool=False) -> pd.DataFrame:
    X=scores_source_order[PHI_COLUMNS].to_numpy(dtype=float)
    Z=StandardScaler().fit_transform(X)
    rows=[]
    n_init = 5 if quick else int(gmm_cfg.get('n_init',50))
    for k in range(int(gmm_cfg.get('k_min',2)), int(gmm_cfg.get('k_max',10))+1):
        model=GaussianMixture(
            n_components=k,
            covariance_type=gmm_cfg.get('covariance_type','spherical'),
            random_state=int(gmm_cfg.get('random_state',42)),
            n_init=n_init,
            max_iter=int(gmm_cfg.get('max_iter',1000)),
            tol=float(gmm_cfg.get('tol',0.001)),
        )
        labels=model.fit_predict(Z)
        pmax=model.predict_proba(Z).max(axis=1)
        sil=silhouette_score(Z,labels) if len(set(labels))>1 else np.nan
        rows.append([k,model.bic(Z),sil,pmax.mean(),(1-pmax).mean(),int(((1-pmax)>0.20).sum())])
    return pd.DataFrame(rows,columns=['K','BIC','Silhouette score','Mean posterior probability','Mean classification uncertainty','N with uncertainty > 0.20'])


def archetype_summary(assignments: pd.DataFrame) -> pd.DataFrame:
    order=["Balanced Achievers","Economy–Innovation Driven","Planetary–Social Intermediate","Multidimensional Laggards"]
    rows=[]
    for name in order:
        s=assignments[assignments['Archetype']==name]
        row={
            'Archetype':name,
            'Number of countries':len(s),
            'Share of sample (%)':100*len(s)/len(assignments),
            'Mean Ω₆':s['Ω₆'].mean(),
            'Mean posterior probability':s['Posterior probability'].mean(),
            'Mean classification uncertainty':s['Classification uncertainty'].mean(),
        }
        for i,c in enumerate(PHI_COLUMNS,1): row[f'Mean φ{i}']=s[c].mean()
        rows.append(row)
    return pd.DataFrame(rows)


def classify_with_baseline(scores: pd.DataFrame, baseline: ClusterResult) -> pd.DataFrame:
    Z=baseline.scaler.transform(scores[PHI_COLUMNS].to_numpy(dtype=float))
    raw=baseline.model.predict(Z)
    pmax=baseline.model.predict_proba(Z).max(axis=1)
    out=scores[['ISO3','Country','Ω₆']].copy()
    out['Assigned archetype']=pd.Series(raw).map(baseline.raw_to_name).to_numpy()
    out['Posterior probability']=pmax
    out['Classification uncertainty']=1.0-pmax
    return out
