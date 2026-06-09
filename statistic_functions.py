import numpy as np
import pandas as pd
from scipy import stats
from sklearn.pipeline import make_pipeline
import statsmodels.api as sm
from statsmodels.formula.api import ols
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp, mannwhitneyu, levene
import numpy as np
from sklearn.linear_model import RANSACRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import HuberRegressor
from sklearn.preprocessing import StandardScaler


# --- Bayesian factor approximation (Jeffreys-Zellner-Siow approx) ---
def bayes_factor_t(t, n1, n2):
    # fallback if t is nan or string
    if not isinstance(t, (int, float)) or np.isnan(t):
        return np.nan
    # Rouder et al. (2009) JZS BF10 approximation
    r = 0.707
    dfree = n1 + n2 - 2
    return np.sqrt(n1*n2/(n1+n2)) * (1 + (t**2)/(dfree*r**2))**(-(dfree+1)/2)


 # --- Effect size (Cohen's d) ---
def cohens_d(x, y):
    nx, ny = len(x), len(y)
    pooled = np.sqrt(((nx - 1)*np.var(x, ddof=1) + (ny - 1)*np.var(y, ddof=1)) / (nx + ny - 2))
    return (np.mean(x) - np.mean(y)) / pooled


def analyze_column(df, value_col):
    # --- clean & prepare ---
    df = df.copy()
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
    df = df.dropna(subset=[value_col, "task", "band"])

    # Split by factors
    t0 = df[df["task"] == 0][value_col]
    t1 = df[df["task"] == 1][value_col]

    b0 = df[df["band"] == 0][value_col]
    b1 = df[df["band"] == 1][value_col]

    model = ols(f"{value_col} ~ C(task) * C(band)", data=df).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)

    # --- T-tests ---
    t_task, p_task = stats.ttest_ind(t0, t1, equal_var=False)
    t_band, p_band = stats.ttest_ind(b0, b1, equal_var=False)

    d_task = cohens_d(t0, t1)
    d_band = cohens_d(b0, b1)

    BF10_task = bayes_factor_t(t_task, len(t0), len(t1))
    BF10_band = bayes_factor_t(t_band, len(b0), len(b1))

    results = {
        "ANOVA": anova_table,
        "T-test Task (0 vs 1)": {"t": t_task, "p": p_task, "d": d_task, "BF10": BF10_task},
        "T-test Band (0 vs 1)": {"t": t_band, "p": p_band, "d": d_band, "BF10": BF10_band},
    }

    return results



def plot_runs_grid(
    f1_distance1, f1_distance2, g6_distance1, g6_distance2,
    xlim=None, ylim=None, scatter=False, fitLine=False, fitSegments=None,
    mergeTask=False, averageOnTask=False, useRANSAC=False):

    
    """
    Plots evolution of radius per run for all datasets and optionally fits lines.

    Returns
    -------
    slopes : dict
        Dictionary with structure slopes[dataset][task][band] = list of slopes for each segment.
    """

    datasets = {
        'F1 Session 1': f1_distance1,
        'F1 Session 2': f1_distance2,
        'G6 Session 1': g6_distance1,
        'G6 Session 2': g6_distance2
    }

    if fitLine:
        scatter = True

    if fitSegments is None:
        fitSegments = {}

    if xlim is None:
        xlim = [0, max(data.shape[0] for data in datasets.values())]

    # --- Determine layout
    use_single_row = mergeTask or averageOnTask
    n_rows = 1 if use_single_row else 2
    n_cols = 2
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 6 * n_rows), sharex=True, sharey=True)

    axes = np.atleast_2d(axes)
    fig.suptitle(
        'Radius vs Run (Task x Band)' if not use_single_row
        else 'Radius vs Run (Merged/Averaged Tasks)', fontsize=14
    )

    task_range = [0] if use_single_row else range(2)

    # --- Dictionary to store slopes
    slopes = {label: {t: {b: [] for b in range(2)} for t in task_range} for label in datasets}

    for t_idx, task in enumerate(task_range):
        for band in range(2):
            ax = axes[t_idx, band]
            for label, data in datasets.items():
                n_runs = data.shape[0]
                x = np.arange(n_runs)

                # --- Handle merge/average logic
                if averageOnTask:
                    y_vals = np.mean(data[:, :, band], axis=1)
                    x_vals = x
                elif mergeTask:
                    y_vals = data[:, :, band].reshape(-1)
                    x_vals = np.repeat(x, 2)
                else:
                    y_vals = data[:, task, band]
                    x_vals = x

                # --- Plot points or lines
                if scatter:
                    scatter_plot = ax.scatter(x_vals, y_vals, label=label, alpha=0.6)
                    color = scatter_plot.get_facecolor()[0]
                else:
                    line_plot, = ax.plot(x_vals, y_vals, label=label)
                    color = line_plot.get_color()

                # --- Optional fit line(s)
                if fitLine:
                    cuts = sorted(set([0] + fitSegments.get(label, []) + [n_runs]))
                    for i in range(len(cuts) - 1):
                        start, end = cuts[i], cuts[i + 1]
                        seg_mask = (x_vals >= start) & (x_vals <= end)
                        seg_x = x_vals[seg_mask]
                        seg_y = y_vals[seg_mask]

                        if len(seg_x) > 1:

                            # ----------------------------
                            # RANSAC or normal polyfit
                            # ----------------------------
                            if useRANSAC:
                                # reshape for sklearn
                                # Xseg = seg_x.reshape(-1, 1)

                                # model = make_pipeline(
                                #     PolynomialFeatures(1),   # degree 1
                                #     RANSACRegressor()
                                # )
                                # model.fit(Xseg, seg_y)

                                # seg_pred = model.predict(Xseg)

                                # # slope = coeff for x^1
                                # lin = model.named_steps['ransacregressor'].estimator_
                                # slope = lin.coef_[1]      # coef_[0] = bias, coef_[1] = slope
                            
                                # Reshape properly
                                seg_x_reshaped = seg_x.reshape(-1, 1)

                                # Scale
                                x_scaler = StandardScaler()
                                y_scaler = StandardScaler()
                                x_train = x_scaler.fit_transform(seg_x_reshaped)
                                y_train = y_scaler.fit_transform(seg_y.reshape(-1, 1)).ravel()

                                # Fit model
                                model = HuberRegressor(epsilon=2)
                                model.fit(x_train, y_train)

                                # Predict
                                seg_pred = y_scaler.inverse_transform(
                                    model.predict(x_scaler.transform(seg_x_reshaped)).reshape(-1, 1)
                                ).ravel()

                                # FIX: compute slope in original units
                                slope = model.coef_[0] * (y_scaler.scale_[0] / x_scaler.scale_[0])



                            else:
                                coeffs = np.polyfit(seg_x, seg_y, deg=1)
                                slope = coeffs[0]
                                seg_pred = np.poly1d(coeffs)(seg_x)

                            # ----------------------------
                            # plot trendline
                            # ----------------------------
                            ax.plot(seg_x, seg_pred, color=color, linewidth=2)

                            # store slope
                            slopes[label][task][band].append(slope)


            ax.set_xlim(xlim)
            if ylim is not None:
                ax.set_ylim(ylim)

            if t_idx == n_rows - 1:
                ax.set_xlabel('Run index')
            if band == 0:
                ax.set_ylabel('Radius')

            title_task = (
                f'Task {task}, Band {band}' if not use_single_row
                else f'Band {band}'
            )
            ax.set_title(title_task)

    # --- Shared legend
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=4, frameon=False)

    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.show()

    return slopes


def plot_df_runs_grid(
    df, y_col,
    fitColumn=None,
    xlim=None, ylim=None,
    fitLine=False,
    scatter=True,
    mergeTask=False,
    averageOnTask=False,
    useHuber=False,
    title=None
):
    """
    Plot y_col vs run for each band & task in a grid.
    Supports:
        - segmentation by fitColumn
        - merging tasks
        - averaging tasks BEFORE segmentation (no duplicates)
        - optional linear or Huber regression
    """

    use_single_row = mergeTask or averageOnTask
    n_tasks = df["task"].nunique()
    n_bands = df["band"].nunique()

    n_rows = 1 if use_single_row else n_tasks
    n_cols = n_bands

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(16, 6 * n_rows),
        sharex=True, sharey=True
    )
    axes = np.atleast_2d(axes)

    if title is not None:
        fig.suptitle(
            title,
            fontsize=14
        )


    for idxTask in range(n_rows):
        for idxBand in range(n_cols):

            ax = axes[idxTask, idxBand]

       
            if not mergeTask and not averageOnTask:
                # Normal mode: separate per task
                data = df[(df["task"] == idxTask) & (df["band"] == idxBand)]
            else:
                # Merge or average: ignore task
                data = df[df["band"] == idxBand]

            if data.empty:
                ax.set_title(f"Task {idxTask}, Band {idxBand} (NO DATA)")
                continue

            if averageOnTask:
                group_cols = ["nRun"]
                if fitColumn is not None and fitColumn in data.columns:
                    group_cols.append(fitColumn)

                data = (
                    data.groupby(group_cols)[y_col]
                    .mean()
                    .reset_index()
                )

            runs = np.sort(data["nRun"].unique())

            x_vals = data["nRun"].values
            y_vals = data[y_col].values


            if fitColumn is not None and fitColumn in data.columns:
                segments = np.sort(data[fitColumn].unique())
                cmap = plt.get_cmap("tab10")

                for i, seg in enumerate(segments):
                    seg_data = data[data[fitColumn] == seg]
                    xs = seg_data["nRun"].values
                    ys = seg_data[y_col].values

                    color = cmap(i % cmap.N)

                    # scatter or line
                    if scatter:
                        ax.scatter(xs, ys, color=color, edgecolors='k',
                                   alpha=0.7, label=f"{fitColumn}={seg}")
                    else:
                        ax.plot(xs, ys, color=color, label=f"{fitColumn}={seg}")

                    # ------------------ FITTING ------------------
                    if fitLine and len(xs) > 1:
                        if useHuber:
                            X = xs.reshape(-1, 1)

                            x_scaler = StandardScaler()
                            y_scaler = StandardScaler()

                            Xs = x_scaler.fit_transform(X)
                            Ys = y_scaler.fit_transform(ys.reshape(-1, 1)).ravel()

                            model = HuberRegressor(epsilon=2).fit(Xs, Ys)
                            pred_scaled = model.predict(x_scaler.transform(X))
                            pred = y_scaler.inverse_transform(
                                pred_scaled.reshape(-1, 1)
                            ).ravel()
                        else:
                            coeffs = np.polyfit(xs, ys, 1)
                            pred = np.poly1d(coeffs)(xs)

                        ax.plot(xs, pred, color=color, linewidth=2)

            else:
              
                if scatter:
                    ax.scatter(x_vals, y_vals, edgecolors='k', alpha=0.7)
                else:
                    ax.plot(x_vals, y_vals)


            if xlim is None:
                ax.set_xlim([runs.min(), runs.max()])
            else:
                ax.set_xlim(xlim)

            if ylim is None:
                ax.set_ylim([0, data[y_col].max() * 1.05])
            else:
                ax.set_ylim(ylim)

            # Labels
            if idxTask == n_rows - 1:
                ax.set_xlabel("Run index")
            if idxBand == 0:
                ax.set_ylabel(y_col)

            ax.set_title(
                f"Band {idxBand}" if use_single_row
                else f"Task {idxTask}, Band {idxBand}"
            )


    handles, labels = axes[-1, -1].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc='lower center', ncol=4, frameon=False)

    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.show()
    return fig



def from_column_toNumpy_array(df, value_name, index_name='runId', column_names=['task', 'band']):
    missing_cols = [c for c in [index_name, value_name] + column_names if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns in DataFrame: {missing_cols}")
    
    runs = sorted(df[index_name].unique())
    uniques = [sorted(df[col].unique()) for col in column_names]
    
    shape = (len(runs),) + tuple(len(u) for u in uniques)
    
    pivot = df.pivot_table(index=index_name, columns=column_names, values=value_name)
    arr = pivot.to_numpy().reshape(shape)
    return arr


def compare_groups_all(df, split_col, metric_col, split_point):
    """
    Perform statistical comparison (KS, Mann-Whitney, Levene)
    for each combination of band (0/1) and task (0/1).

    Parameters
    ----------
    df : pd.DataFrame
        The full dataset.
    split_col : str
        Column separating the two conditions (e.g., 'runId').
    metric_col : str
        Column to compare statistically (e.g., 'radius', 'accuracy').
    split_point : int/float
        Threshold separating the two setups.

    Returns
    -------
    summary_df : pd.DataFrame
        One row per (band, task) combination with all test results.
    """

    results = []

    for band_val in [0, 1]:
        for task_val in [0, 1]:

            # Subset for this band/task combination
            subset = df[(df["band"] == band_val) & (df["task"] == task_val)]

            # Split into two conditions
            group1 = subset[subset[split_col] < split_point][metric_col].dropna()
            group2 = subset[subset[split_col] >= split_point][metric_col].dropna()

            if len(group1) == 0 or len(group2) == 0:
                results.append({
                    "band": band_val,
                    "task": task_val,
                    "metric": metric_col,
                    "n_group1": len(group1),
                    "n_group2": len(group2),
                    "ks_p": None,
                    "mw_p": None,
                    "levene_p": None
                })
                continue

            # Statistical tests
            ks_stat, ks_p = ks_2samp(group1, group2)
            mw_stat, mw_p = mannwhitneyu(group1, group2, alternative="two-sided")
            lev_stat, lev_p = levene(group1, group2)

            results.append({
                "band": band_val,
                "task": task_val,
                "metric": metric_col,
                "n_group1": len(group1),
                "n_group2": len(group2),
                "ks_p": ks_p,
                "mw_p": mw_p,
                "levene_p": lev_p
            })

    return pd.DataFrame(results)

def add_cos_sin_angles(df, column_name='angle'):
    df[f'cos_{column_name}'] =  np.cos(df[column_name])
    df[f'sin_{column_name}'] =  np.sin(df[column_name])
