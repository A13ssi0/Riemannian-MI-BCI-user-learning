import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import HuberRegressor
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import HuberRegressor
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.lines as mlines

def plot_array_runs_grid(
    data_array,
    lbl=None,                  # shape: (n_runs,)
    y_col_name="Y",
    xlim=None, ylim=None,
    fitLine=False,
    scatter=True,
    mergeTask=False,
    averageOnTask=False,
    day_start_idx=None,
    day_labels=None,
    useHuber=False,
    idxStop=None,              # int -> red vertical line
    idxRec=None,               # int -> yellow vertical line
    title=None,
):
    """
    Plot data from array shaped (nBands, nRuns, nTasks)

    data_array[band, run, task]

    lbl: array of length nRuns, used to segment & fit by label
    idxStop: int, draw red vertical line
    idxRec: int, draw yellow vertical line
    """

    if len(data_array.shape) != 3:
        data_array = data_array[:, :, np.newaxis]

    n_bands, n_runs, n_tasks = data_array.shape

    color_backgrounLines = 'gainsboro'

    if lbl is not None:
        lbl = np.asarray(lbl)
        assert len(lbl) == n_runs, "lbl must have length n_runs"

    use_single_row = mergeTask or averageOnTask

    n_rows = 1 if use_single_row else n_tasks
    n_cols = n_bands

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(16, 6 * n_rows),
        sharex=True, sharey=True
    )
    axes = np.atleast_2d(axes)

    if title is not None:
        fig.suptitle(title, fontsize=14, x=0.5, y=0.9)

    base_runs = np.arange(n_runs)
    cmap = plt.get_cmap("tab10")

    for idxTask in range(n_rows):
        for idxBand in range(n_cols):

            ax = axes[idxTask, idxBand]

            # ------------------ VERTICAL MARKERS ------------------
            if day_start_idx is not None:
                for k in day_start_idx:
                    ax.axvline(k, color=color_backgrounLines, linewidth=1 ,zorder=0)

            legend_lines = []

            if idxStop is not None:
                for k in idxStop:
                    ax.axvline(k, color="red", linewidth=1.5 ,zorder=1)
                legend_lines.append(mlines.Line2D([], [], color='red', linewidth=1.5, label='Break'))

            if idxRec is not None:
                for k in idxRec:
                    if k in idxStop: k+=0.3
                    ax.axvline(k, color="grey", linewidth=1.5 ,zorder=1)
                legend_lines.append(mlines.Line2D([], [], color='grey', linewidth=1.5, label='Recalibration'))

            # ------------------ SELECT DATA ------------------
            if not mergeTask and not averageOnTask:
                y_base = data_array[idxBand, :, idxTask]
                runs = base_runs
                lbl_local = lbl

            else:
                band_data = data_array[idxBand, :, :]

                if averageOnTask:
                    y_base = np.nanmean(band_data, axis=1)
                    runs = base_runs
                    lbl_local = lbl

                else:
                    # mergeTask: flatten all tasks
                    y_base = band_data.reshape(-1)
                    runs = np.tile(base_runs, n_tasks)
                    if lbl is not None:
                        lbl_local = np.tile(lbl, n_tasks)
                    else:
                        lbl_local = None

            # Remove NaNs
            mask_all = ~np.isnan(y_base)
            y_base = y_base[mask_all]
            runs = runs[mask_all]
            if lbl_local is not None:
                lbl_local = lbl_local[mask_all]

            if len(y_base) == 0:
                ax.set_title(f"Band {idxBand} (NO DATA)")
                continue

            # ------------------ PLOT BY LABEL ------------------
            if lbl_local is not None:
                segments = np.unique(lbl_local)

                for i, seg in enumerate(segments):
                    seg_mask = lbl_local == seg
                    xs = runs[seg_mask]
                    ys = y_base[seg_mask]

                    if i == 3:  color = cmap(i+1 % cmap.N)
                    else:       color = cmap(i % cmap.N)

                    if scatter:
                        ax.scatter(xs, ys, color=color,
                                   edgecolors='k', alpha=0.85),
                                #    label=f"lbl={seg}")
                    else:
                        ax.plot(xs, ys, color=color,
                                label=f"lbl={seg}")

                    # -------- FIT PER LABEL --------
                    if fitLine and len(xs) > 1:
                        # Extend by one run
                        xs_ext = np.append(xs, xs.max() + 1)

                        if useHuber:
                            X = xs.reshape(-1, 1)
                            X_ext = xs_ext.reshape(-1, 1)

                            x_scaler = StandardScaler()
                            y_scaler = StandardScaler()

                            Xs = x_scaler.fit_transform(X)
                            Ys = y_scaler.fit_transform(
                                ys.reshape(-1, 1)
                            ).ravel()

                            model = HuberRegressor(epsilon=2).fit(Xs, Ys)

                            Xs_ext = x_scaler.transform(X_ext)
                            pred_scaled = model.predict(Xs_ext)
                            pred_ext = y_scaler.inverse_transform(
                                pred_scaled.reshape(-1, 1)
                            ).ravel()
                        else:
                            coeffs = np.polyfit(xs, ys, 1)
                            pred_ext = np.poly1d(coeffs)(xs_ext)

                        ax.plot(xs_ext, pred_ext, color=color, linewidth=2)


            else:
                # No labels: single color
                if scatter:
                    ax.scatter(runs, y_base,
                               edgecolors='k', alpha=0.7)
                else:
                    ax.plot(runs, y_base)

                if fitLine and len(runs) > 1:
                    if useHuber:
                        X = runs.reshape(-1, 1)
                        x_scaler = StandardScaler()
                        y_scaler = StandardScaler()
                        Xs = x_scaler.fit_transform(X)
                        Ys = y_scaler.fit_transform(
                            y_base.reshape(-1, 1)
                        ).ravel()
                        model = HuberRegressor(epsilon=2).fit(Xs, Ys)
                        pred_scaled = model.predict(Xs)
                        pred = y_scaler.inverse_transform(
                            pred_scaled.reshape(-1, 1)
                        ).ravel()
                    else:
                        coeffs = np.polyfit(runs, y_base, 1)
                        pred = np.poly1d(coeffs)(runs)

                    ax.plot(runs, pred, linewidth=2)

            # ------------------ AXES ------------------
            if xlim is None:
                ax.set_xlim([0, n_runs - 1])
            else:
                ax.set_xlim(xlim)

            if ylim is None:
                ax.set_ylim([0, np.nanmax(y_base) * 1.05])
            else:
                ax.set_ylim(ylim)

            # Labels
            # if idxTask == n_rows - 1:
            #     ax.set_xlabel("Run index")
            if idxBand == 0:
                ax.set_ylabel(y_col_name)

            if day_start_idx is not None and day_labels is not None:
                ax.set_xticks(day_start_idx)
                ax.set_xticklabels(day_labels, rotation=60,  ha='right')


                # Rotate and move x tick labels diagonally closer to axis
            ax.tick_params(axis='x', pad=0)   # smaller = closer to axis
   


            _, ymax = ax.get_ylim()
            ticks = np.arange(0, ymax + 1e-8, 0.2)
            ax.set_yticks(ticks)
            ax.set_yticklabels([f"{t:.1f}" for t in ticks])

            for k in ticks:
                ax.axhline(k, color=color_backgrounLines, linewidth=1 ,zorder=0)

            ax.tick_params(axis='both', direction='in', length=5)

            # ax.set_title(
            #     f"Band {idxBand}" if use_single_row
            #     else f"Task {idxTask}, Band {idxBand}"
            # )

    # ------------------ LEGEND ------------------
    # handles, labels = axes[-1, -1].get_legend_handles_labels()
    # if handles:
    #     fig.legend(handles, labels, loc='lower center',
    #                ncol=4, frameon=False)
        # ------------------ ADD LEGEND ------------------
    # Add inside the first subplot for simplicity

    handles, labels = ax.get_legend_handles_labels()
    # Include vertical line legend
    handles.extend(legend_lines)
    ax.legend(handles=handles, loc='upper right', frameon=True)

    plt.tight_layout(rect=[0, 0.05, 1, 0.95])

    import matplotlib.transforms as mtransforms
    fig.canvas.draw()

    dx_points = 5   # tweak this (e.g. 6, 10, 15)
    dx = dx_points / 72.0

    for ax in axes.flat:
        offset = mtransforms.ScaledTranslation(
            dx, 0, fig.dpi_scale_trans
        )
        for label in ax.get_xticklabels():
            label.set_transform(label.get_transform() + offset)

    plt.show()


    return fig

def analyze_task_evolution(
    data_array,
    lbl,
    label_start_idx,
    bands=None,
    verbose=True,
    min_tasks_random=5
):
    """
    Statistical test for whether TASKS differ in EVOLUTION
    after controlling for time shift (label start) and label effects.

    Automatically switches to fixed-effect model if number of tasks < min_tasks_random.

    Parameters
    ----------
    data_array : np.ndarray
        Shape (nBands, nRuns, nTasks)

    lbl : array-like
        Length nRuns, label for each run

    label_start_idx : dict or array-like
        Mapping label -> start index

    bands : list or None
        Bands to include. If None, include all.

    verbose : bool
        Print model summaries

    min_tasks_random : int
        Minimum number of tasks to use random-slope mixed model

    Returns
    -------
    results : dict
        Contains fitted models, slopes, and optionally ANOVA table
    """

    import numpy as np
    import pandas as pd
    import statsmodels.formula.api as smf
    import statsmodels.api as sm

    n_bands, n_runs, n_tasks = data_array.shape

    if bands is None:
        bands = list(range(n_bands))

    # ------------------ BUILD LONG-FORM DATA ------------------
    rows = []

    for band in bands:
        for task in range(n_tasks):
            for run in range(n_runs):
                y = data_array[band, run, task]
                if np.isnan(y):
                    continue

                lab = lbl[run]

                # get label start
                if isinstance(label_start_idx, dict):
                    t0 = label_start_idx[lab]
                else:
                    t0 = label_start_idx[lab]

                t_aligned = run - t0

                rows.append({
                    "y": y,
                    "t_aligned": t_aligned,
                    "label": str(lab),
                    "task": str(task),
                    "band": str(band)
                })

    df_long = pd.DataFrame(rows)

    if verbose:
        print(f"Built long-form dataframe with {len(df_long)} rows")
        print(df_long.head())

    results = {"df_long": df_long}

    # ------------------ DECIDE MODEL TYPE ------------------
    if n_tasks >= min_tasks_random:
        # Random-slope mixed model
        if verbose:
            print("\nUsing mixed-effects random-slope model (tasks >= 5)")
        model_slope = smf.mixedlm(
            "y ~ t_aligned * label",
            data=df_long,
            groups=df_long["task"],
            re_formula="~t_aligned"
        ).fit()
        results["model_slope"] = model_slope

        if verbose:
            print(model_slope.summary())

        # Extract per-task slopes
        fixed_slope = model_slope.fe_params["t_aligned"]
        task_random = model_slope.random_effects
        task_slopes = {}
        for task, re in task_random.items():
            task_slopes[task] = fixed_slope + re.get("t_aligned", 0)
        results["task_slopes"] = pd.DataFrame({
            "task": list(task_slopes.keys()),
            "slope": list(task_slopes.values())
        })

    else:
        # Fixed-effect OLS / ANOVA
        if verbose:
            print("\nUsing fixed-effect OLS/ANCOVA model (tasks < 5)")

        # Full model including task interactions
        formula = "y ~ t_aligned * label * task"
        model_ols = smf.ols(formula, data=df_long).fit()
        results["model_ols"] = model_ols

        anova_table = sm.stats.anova_lm(model_ols, typ=2)
        results["anova_table"] = anova_table

        if verbose:
            print("\nANOVA Table:")
            print(anova_table)

        # Extract per-task slopes
        task_slopes = {}
        for task in df_long["task"].unique():
            mask = df_long["task"] == task
            slope, _ = np.polyfit(
                df_long.loc[mask, "t_aligned"],
                df_long.loc[mask, "y"],
                1
            )
            task_slopes[task] = slope
        results["task_slopes"] = pd.DataFrame({
            "task": list(task_slopes.keys()),
            "slope": list(task_slopes.values())
        })

        if verbose:
            print("\nEstimated per-task slopes (after alignment):")
            print(results["task_slopes"].sort_values("task"))

    return results
