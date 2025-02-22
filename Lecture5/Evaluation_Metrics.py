import marimo

__generated_with = "0.11.7"
app = marimo.App(layout_file="layouts/Evaluation_Metrics.slides.json")


@app.cell(hide_code=True)
def _():
    import marimo as mo
    import numpy as np
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    mo.latex(filename=mo.notebook_dir() / "preamble.tex")
    mo.md(r"""# Different Evaluation Metrics""")
    return go, make_subplots, mo, np


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Medical diagnosis: From patient’s perspective

        - A test for COVID-19 has an accuracy of $90\%$, i.e.,
          $$\Pr(\hat{\R{Y}} = \R{Y}) = 0.9$$
            - $\R{Y}$: Indicator of infection.
            - $\hat{\R{Y}}$: Diagnosis of infection.
        - Suppose a person is diagnosed to have the virus, i.e., $\hat{\R{Y}} = 1$.
          - Is it likely ($>50\%$ chance) that the person has the virus? <u>Y/N</u>
          - Is the likelihood $90\%$? <u>Y/N</u>
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.vstack(
        [
            mo.md(r"""## Confusion matrix for binary classification"""),
            mo.image(src=mo.notebook_dir() / "images/cm.dio.svg").center(),
            mo.md(
                r"""
        - TP (True +ve): number of +ve tuples classified as +ve.
        - TN (True -ve): number of -ve tuples classified as -ve.
        - FP (False +ve): number of -ve tuples classified as +ve.  
          (F_______ a________ / Type I error)
        - FN (False -ve): number of +ve tuples classified as -ve.  
          (M______ d________ / Type II error)
        """
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.vstack(
        [
            mo.md(r"""## Accuracy vs Precision"""),
            mo.image(src=mo.notebook_dir() / "images/precision.dio.svg").center(),
            mo.md(
                r"""
        - Accuracy is $\frac{\op{TP} + \op{TN}}{n}$ where $n = \op{TP} + \op{TN} + \op{FP} + \op{FN}$.
        - Precision is $\frac{\op{TP}}{\hat{P}}$ where $\hat{P} = \op{TP} + \op{FP}$.
        - P______________ p______________ v______________ (PPV)
        - Is it possible that accuracy is high but precision is low?
        """
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.vstack(
        [
            mo.md(r"""### Example"""),
            mo.image(src=mo.notebook_dir() / "images/precision_.dio.svg").center(),
            mo.md(
                r"""
                - Accuracy is ____________%.
                - Precision is ____________%.
                - When is accuracy > precision in general?
                """
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.vstack(
        [
            mo.md(r"""## Negative predictive value (NPV)"""),
            mo.image(src=mo.notebook_dir() / "images/NPV.dio.svg").center(),
            mo.md(
                r"""
                - NPV is $\frac{\op{TN}}{\hat{N}}$ where $\hat{N} = \op{TN} + \op{FN} = n - \hat{P}$.
                - Accuracy is $\frac{\op{TP} + \op{TN}}{n} = \frac{\hat{P} \cdot \op{PPV} + \hat{N} \cdot \op{NPV}}{n} = \frac{\hat{P}}{n} \op{PPV} + \frac{\hat{N}}{n} \op{NPV}$.
                - Accuracy > precision iff NPV $\ge$ PPV.
                - Accuracy = precision iff _________________________________________________________
                """
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.vstack(
        [
            mo.md(r"""### Example"""),
            mo.image(src=mo.notebook_dir() / "images/NPV_.dio.svg").center(),
            mo.md(
                r"""
                - Accuracy is _______________%.
                - Precision is _______________%.
                - NPV is ________________%.
                """
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.vstack(
        [
            mo.md(r"""## Medical diagnosis: From Government’s perspective"""),
            mo.md(
                r"""
                - Suppose the government wants to eradicate COVID-19 as it is highly contagious.
                - If a test is $90\%$ accurate, can the government identify $>50\%$ of infected people? <u>Y/N</u>
                """
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.vstack(
        [
            mo.md(r"""## Recall"""),
            mo.image(src=mo.notebook_dir() / "images/recall.dio.svg").center(),
            mo.md(
                r"""
                - Recall is $\frac{\op{TP}}{\op{P}}$ where $\op{P} = \op{TP} + \op{FN}$.
                - S__________ or True positive rate (TPR)
                """
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.vstack(
        [
            mo.md(r"""### Example"""),
            mo.image(src=mo.notebook_dir() / "images/recall_.dio.svg").center(),
            mo.md(
                r"""
                - Accuracy is ____________%.
                - Precision is ____________%.
                - NPV is __________________%.
                - Recall is ___________________________%.
                - When is accuracy $>$ recall?
                """
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.vstack(
        [
            mo.md(r"""## Specificity"""),
            mo.image(src=mo.notebook_dir() / "images/TNR.dio.svg").center(),
            mo.md(
                r"""
                - Specificity is $\frac{\op{TN}}{N}$ where $N = \op{TN} + \op{FP}$.  
                  True negative rate (TNR)
                - Accuracy is 
                  $$\frac{\op{TP} + \op{TN}}{n} = \frac{P \cdot \op{TPR} + N \cdot \op{TNR}}{n} = \frac{P}{n} \op{TPR} + \frac{N}{n} \op{TNR}$$
                - Accuracy > recall iff $\op{TNR} \geq \op{TPR}$.
                - Accuracy = recall iff ______________________________________________________
                """
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.vstack(
        [
            mo.md(r"""## Example"""),
            mo.image(src=mo.notebook_dir() / "images/TNR_.dio.svg").center(),
            mo.md(
                r"""
                - Accuracy is ____________%.
                - Precision is ____________%.
                - NPV is __________________%.
                - Recall is ___________________________%.
                - Specificity is _________________________%.
                """
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.vstack(
        [
            mo.md(r"""## Class imbalance problem"""),
            mo.md(
                r"""
                - Happens when $P \ll N$ (or $N \ll P$).
                - If $P \ll N$, accuracy can be dominated by ____________ over __________________.

                  $$\op{Accuracy} = \frac{{\color{grey}{\op{TP}}} + \op{TN}}{n}
                  = {\color{grey}{\frac{P}{n} \cdot \op{TPR}}} + \frac{N}{n} \cdot \op{TNR} = {\color{grey}{\frac{P}{n} \cdot \op{PPV}}} + \frac{N}{n} \cdot \op{NPV}$$

                - How to evaluate the prediction of positive class?
                """
            ),
            mo.md(
                r"""
                - Cost/benefit analysis
                    - Different per unit cost/benefit assigned to FP, FN, TP, and TN.
                    - Minimize total cost or maximize total benefit.
                      $$\op{Cost} = \op{FP} \cdot \op{Cost}_{\op{FP}} + \op{FN} \cdot \op{Cost}_{\op{FN}} + \op{TP} \cdot \op{Cost}_{\op{TP}} + \op{TN} \cdot \op{Cost}_{\op{TN}}$$
                """
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(fig_m, mo):
    mo.vstack(
        [
            mo.md(r"""## F score"""),
            mo.md(
                r"""$$F_1 := \left( \frac{\left( \op{PPV}^{-1} + \op{TPR}^{-1} \right)}{2} \right)^{-1} = \frac{2 \cdot \op{PPV} \cdot \op{TPR}}{\op{PPV} + \op{TPR}}$$"""
            ),
            mo.as_html(fig_m).center(),
            mo.hstack(
                [
                    mo.md(
                        r"""- Arithmetic mean$=0.7$ implies $\op{PPV,TPR}\geq$_____"""
                    ),
                    mo.md(
                        r"""- Harmonic mean$=0.7$ implies $\op{PPV,TPR}\geq$_____"""
                    ),
                ]
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(go, make_subplots, np):
    def plot_am_hm():
        # Create a meshgrid for x and y in the range [0.001, 1]
        x = np.linspace(0.001, 1, 100)
        y = np.linspace(0.001, 1, 100)
        X, Y = np.meshgrid(x, y)

        # Function to create a surface trace
        def create_surface_trace(X, Y, Z):
            return go.Surface(
                x=X,
                y=Y,
                z=Z,
                colorscale="Viridis",
                colorbar=dict(title="z"),
                opacity=0.4,
                contours_z=dict(
                    show=True,
                    usecolormap=True,
                    highlightcolor="limegreen",
                    project_z=True,
                    start=0,
                    end=1,
                    size=0.1,
                ),
            )

        # Arithmetic mean
        am = (X + Y) / 2
        am_trace = create_surface_trace(X, Y, am)

        # Harmonic mean
        hm = ((X**-1 + Y**-1) / 2) ** -1
        hm_trace = create_surface_trace(X, Y, hm)

        # Create subplots
        fig = make_subplots(
            rows=1,
            cols=2,
            subplot_titles=(
                r"$\text{Arithmetic Mean }z=\frac{x+y}{2}$",
                r"$\text{Harmonic Mean }z=\Big(\frac{x^{-1}+y^{-1}}{2}\Big)^{-1}$",
            ),
            specs=[[{"type": "surface"}, {"type": "surface"}]],
        )

        # Add surface plots to subplots
        fig.add_trace(am_trace, row=1, col=1)
        fig.add_trace(hm_trace, row=1, col=2)
        fig.update_scenes(
            camera=dict(eye=dict(x=1.5, y=-1.5, z=1.5)),
            xaxis_title="x",
            yaxis_title="y",
            zaxis_title="z",
        )

        # Update layout
        fig.update_layout(
            title_text="Why Harmonic means instead of arithmetic mean?",
            width=900,
            height=540,
        )

        return fig


    fig_m = plot_am_hm()
    return fig_m, plot_am_hm


@app.cell(hide_code=True)
def _(fig_F_beta, mo):
    mo.vstack(
        [
            mo.md(r"""## F-beta score"""),
            mo.md(
                r"""
                $$F_{\beta} := \left( \frac{\op{PPV}^{-1} + \beta^2 \op{TPR}^{-1} }{\beta^2 + 1} \right)^{-1} = \frac{(\beta^2 + 1) \cdot \op{PPV} \cdot \op{TPR}}{\beta^2 \cdot \op{PPV} + \op{TPR}} \quad \op{for} \ \beta > 0$$
                """
            ),
            mo.hstack(
                [
                    mo.as_html(fig_F_beta),
                    mo.md(
                        r"""
                - As $\beta \to \infty$, $F_{\beta} \to$ ____
                - As $\beta \to 0$, $F_{\beta} \to$ ____
                """
                    ),
                ],
                align="center"
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(go, np):
    def plot_F_beta():
        # Create a meshgrid for x and y in the range [0.001, 1]
        x = np.linspace(0.001, 1, 100)
        y = np.linspace(0.001, 1, 100)
        X, Y = np.meshgrid(x, y)

        # Function to compute F-beta score
        def F_beta_score(X, Y, beta):
            precision = X
            recall = Y
            return (
                (1 + beta**2)
                * (precision * recall)
                / (beta**2 * precision + recall)
            )

        # Create the initial F-beta score surface plot
        beta = 1.0
        F_beta = F_beta_score(X, Y, beta)

        fig = go.Figure(
            data=[
                go.Surface(
                    x=X,
                    y=Y,
                    z=F_beta,
                    colorscale="Viridis",
                    colorbar=dict(title="F-β"),
                    opacity=0.4,
                )
            ]
        )

        # Update traces for contours
        fig.update_traces(
            contours_z=dict(
                show=True,
                usecolormap=True,
                highlightcolor="limegreen",
                project_z=True,
                start=0,
                end=1,
                size=0.1,
            )
        )

        # Generate logarithmic beta values
        log_beta_values = np.logspace(-2, 2, 19)  # from 0.01 to 100

        # Add slider for beta value
        sliders = [
            {
                "steps": [
                    {
                        "method": "update",
                        "label": f"{beta:.2f}",
                        "args": [
                            {"z": [F_beta_score(X, Y, beta)]},
                            {"title": f"F-β Score Surface Plot (β = {beta:.2f})"},
                        ],
                    }
                    for beta in log_beta_values
                ],
                "active": 9,
                "currentvalue": {"prefix": "β: ", "font": {"size": 20}},
                "pad": {"b": 50},
            }
        ]

        # Update layout
        fig.update_layout(
            autosize=False,
            title=f"F-β Score Surface Plot (β = {beta:.2f})",
            scene=dict(
                xaxis_title="PPV",
                yaxis_title="TPR",
                zaxis_title="F-β Score",
                camera_eye=dict(x=1.3, y=-1.3, z=1.3),
            ),
            width=500,
            height=500,
            margin=dict(l=65, r=50, b=65, t=90),
            sliders=sliders,
        )

        return fig


    fig_F_beta = plot_F_beta()
    return fig_F_beta, plot_F_beta


@app.cell(hide_code=True)
def _(mo):
    mo.vstack(
        [
            mo.md(r"""## Threshold-moving"""),
            mo.image(src=mo.notebook_dir() / "images/proba_clf.dio.svg").center(),
            mo.md(r"""- Apply a threshold $\gamma$ to the output of a [probabilistic classifier](https://en.wikipedia.org/wiki/Probabilistic_classification)."""),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.vstack(
        [
            mo.md(r"""## Area under curve (AUC)"""),
            mo.image(src=mo.notebook_dir() / "images/AUC.dio.svg").center(),
            mo.md(
                r"""
                - Obtain the trade-offs of different performance metrics by varying the threshold.
                - Receiver operation characteristics curve (ROC):
                    - Plot of TPR against FPR (False positive rate=1-TNR)
                    - AUC: ROC area
                - Precision recall curve (PRC):
                    - Plot of precision against recall
                    - AUC: PRC area
                - Which is better, ROC or PRC?
                """
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.vstack(
        [
            mo.md(r"""## References"""),
            mo.md(
                r"""
                - 8.5.1 Metrics for Evaluating Classifier Performance
                - 8.5.6 Comparing Classifiers based on Cost-Benefits and ROC Curves
                """
            ),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
