import marimo

__generated_with = "0.11.9"
app = marimo.App()


@app.cell(hide_code=True)
def _():
    import marimo as mo
    import pandas as pd
    import plotly.graph_objects as go

    mo.latex(filename=mo.notebook_dir() / "preamble.tex")
    mo.md(r"""# Ensemble Methods""")
    return go, mo, pd


@app.cell(hide_code=True)
def _(mo):
    mo.vstack(
        [
            mo.md(r"""## Man vs Machine Rematch"""),
            mo.image(src=mo.notebook_dir() / "images/RF.dio.svg").center(),
        ]
    )
    return


@app.cell(hide_code=True)
def _(man_vs_machine_fig, mo):
    mo.vstack(
        [
            mo.md(r"""## Segment Challenge Results"""),
            mo.md(
                r"""
                $$F_1 := \left( \frac{\left( \op{PPV}^{-1} + \op{TPR}^{-1} \right)}{2} \right)^{-1} = \frac{2 \cdot \op{PPV} \cdot \op{TPR}}{\op{PPV} + \op{TPR}}$$
                """
            ),
            mo.as_html(man_vs_machine_fig).center(),
        ]
    )
    return


@app.cell(hide_code=True)
def _(go, mo, pd):
    def plot_man_vs_machine():

        # Load the data
        rf_data = pd.read_csv(mo.notebook_dir() / "RF.csv")
        human_data = pd.read_csv(mo.notebook_dir() / "human.csv")

        # Create a combined dataframe with an additional column to distinguish the datasets
        rf_data["source"] = "RF"
        human_data["source"] = "Human"
        combined_data = pd.concat([rf_data, human_data])

        # Exclude data points with missing values
        combined_data = combined_data.dropna()

        # Function to filter out dominating points
        def filter_max_accuracy_points(data):
            data = data.sort_values(by="depth")
            filtered_data = []

            for i, row in data.iterrows():
                if not any(
                    (data["depth"] <= row["depth"]) & (data["accuracy"] > row["accuracy"])
                ):
                    filtered_data.append(row)

            return pd.DataFrame(filtered_data)

        # Apply the filtering function for each source
        max_accuracy_points = (
            combined_data.groupby("source")
            .apply(filter_max_accuracy_points, include_groups=False)
            .reset_index(drop=True)
        )

        # Create the scatter plot using go.Scatter
        fig = go.Figure()

        # Add traces for each source
        for source in combined_data["source"].unique():
            source_data = combined_data[combined_data["source"] == source]
            fig.add_trace(
                go.Scatter(
                    x=source_data["depth"],
                    y=source_data["accuracy"],
                    mode="markers+text",
                    text=source_data["name"],
                    name=source,
                    textfont=dict(color="rgba(0,0,0,0)"),  # Make text transparent
                    marker=dict(size=10),
                )
            )

        # Update layout with labels and title
        fig.update_layout(
            title="Man vs Machine", xaxis_title="Tree Depth", yaxis_title="Accuracy"
        )

        # Add hover information
        fig.update_traces(hovertemplate="<b>%{text}</b><br>Accuracy: %{y}<br>Depth: %{x}")

        # Add annotations for the points with the highest accuracy
        for i, row in max_accuracy_points.iterrows():
            fig.add_annotation(
                x=row["depth"],
                y=row["accuracy"],
                text=f"{row['name']}, {row['accuracy']}",
                showarrow=True,
                arrowhead=2,
                ax=20,
                ay=-30,
                bgcolor="rgba(255, 255, 255, 0.6)",
                opacity=1,
                font=dict(size=10),
                hovertext=f"{row['name']}, {row['accuracy']}",
            )

        return fig

    man_vs_machine_fig = plot_man_vs_machine()
    return man_vs_machine_fig, plot_man_vs_machine


@app.cell(hide_code=True)
def _(mo):
    mo.vstack(
        [
            mo.md(r"""## Two heads are better than one"""),
            mo.md(
                r"""
                - [Bing](https://www.bing.com/translator?from=en&to=zh-Hant&text=Two%20heads%20are%20better%20than%20one)/[Baidu](https://fanyi.baidu.com/#en/zh/Two%20heads%20are%20better%20than%20one)/[Google](https://translate.google.com/#view=home&op=translate&sl=auto&tl=zh-TW&text=Two%20heads%20are%20better%20than%20one) translation.
                - The story in [Chinese](http://www.youth.com.tw/db/epaper/es001010/eb0758.htm) and its translation to [English](https://translate.google.com/translate?hl=en&sl=auto&tl=en&u=http%3A%2F%2Fwww.youth.com.tw%2Fdb%2Fepaper%2Fes001010%2Feb0758.htm).
                - Can we combine two poor classifiers into a good classifier?
                - What is the benefit of doing so?
                """
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.vstack(
        [
            mo.image(src=mo.notebook_dir() / "images/ensemble_eg1.dio.svg").center(),
            mo.md(
                r"""
                - Accuracies of $\hat{f}_1$ and $\hat{f}_2$ are both ________%. Are they good?
                - Can we combine them into a better classifier $\hat{f}(x) := g(\hat{f}_1(x), \hat{f}_2(x))$?
                - $\underline{\kern3em}\{\hat{f}_1(x), \hat{f}_2(x)\}$ achieves an accuracy of ______________________%.
                - How does it work in general?
                """
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.vstack(
        [
            mo.md(r"""## Architecture"""),
            mo.image(src=mo.notebook_dir() / "images/arch.dio.svg").center(),
            mo.md(
                r"""
                - Base classifiers $\hat{f}_j$'s are simple but possibly have weak preliminary predictions $\hat{y}_j$'s.
                - Combined classifier $\hat{f}$ uses the combination rule $g$ to merge $\hat{y}_j$'s into a good final prediction $\hat{y}$.
                """
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.vstack(
        [
            mo.md(r"""## Architecture for probabilistic classifiers"""),
            mo.image(src=mo.notebook_dir() / "images/proba.dio.svg").center(),
            mo.md(
                r"""
                - Base classifiers $\hat{f}_j$'s are simple but possibly have weak probability estimates $\hat{P}_{\R{Y}|\RM{X}}^{(j)}(\cdot \mid \M{x})$.
                - Combined classifier $\hat{f}$ uses the combination rule $g$ to merge $\hat{P}_{\R{Y}|\RM{X}}^{(j)}(\cdot \mid \M{x})$'s into a good final prediction $\hat{P}_{\R{Y}|\RM{X}}(\cdot \mid \M{x})$.
                """
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.vstack(
        [
            mo.md(r"""## How to get good performance?"""),
            mo.md(
                r"""
                - Reduce risk by avoiding underfitting and overfitting.
                - For many loss functions $L$ (0-1 loss, sum of squared error, ...):
                  $$\underbrace{\E[L(\R{Y}, \hat{f}_{\R{W}}(\RM{X}))]}_{\text{Risk}} \leq \underbrace{\E[L(\R{Y}, \bar{f}(\RM{X}))]}_{\text{Bias}} + \underbrace{\E[L(\bar{f}(\RM{X}), \hat{f}_{\R{W}}(\RM{X}))]}_{\text{Variance}}$$
                  where
                - $\bar{f} := \M{x} \mapsto \E[\hat{f}_{\R{W}}(\M{x})]$ is the **expected predictor** (W is a random variable. Why?).
                - **Variance** is the dependence of $\hat{f}_{\R{W}}(\RM{X})$ on the data, also known as overfitting/underfitting.
                - **Bias** is the deviation of $\hat{f}(\RM{X})$ from $\R{Y}$, also known as overfitting/underfitting.
                - See the [bias-variance trade-off](https://en.wikipedia.org/wiki/Bias%E2%80%93variance_tradeoff).
                """
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.vstack(
        [
            mo.md(r"""## Bias and variance for probabilistic classifiers"""),
            mo.md(
                r"""
                    - For probabilistic classifiers,
                      $$\underbrace{\E\left[L(P_{\R{Y}|\RM{X}}(\cdot \mid \RM{X}), P_{\hat{\R{Y}}|\RM{X}, \R{W}}(\cdot \mid \RM{X}, \R{W}))\right]}_{\text{Risk}} \leq \underbrace{\E\left[L(P_{\R{Y}|\RM{X}}(\cdot \mid \RM{X}), P_{\hat{\R{Y}}|\RM{X}}(\cdot \mid \RM{X}))\right]}_{\text{Bias}} + \underbrace{I(\hat{\R{Y}}; \R{W} \mid \RM{X})}_{\text{Variance}}$$
                      where
                        - $f_{\R{W}}(\M{x}) := P_{\hat{\R{Y}}|\RM{X}, \R{W}}(\cdot \mid \M{x}, \R{W})$ implies
                          $$\bar{f}(\M{x}) = \E\left[P_{\hat{\R{Y}}|\RM{X}, \R{W}}(\cdot \mid \M{x}, \R{W})\right] = P_{\hat{\R{Y}}|\RM{X}}(\cdot \mid \M{x}),$$
                          called m______________;
                        - $P_{\R{Y}|\RM{X}}(\cdot \mid \RM{X})$ instead of $\R{Y}$ is used as the ground truth;
                        - [Information (or Kullback-Leibler) divergence](https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence) is used as the loss function
                          $$L(Q, P) := D_{KL}(P \parallel Q) := \int_{\mathcal{Y}} dP \log \frac{dP}{dQ};$$
                        - variance becomes the [mutual information](https://en.wikipedia.org/wiki/Mutual_information)
                          $$\E[D_{KL}(P_{\hat{\R{Y}}|\RM{X}, \R{W}}(\cdot \mid \RM{X}, \R{W}) \parallel P_{\hat{\R{Y}}|\RM{X}}(\cdot \mid \RM{X}))] = I(\hat{\R{Y}}; \R{W} \mid \RM{X}) \quad \because I(\RM{X}; \R{W}) = 0.$$
                    """
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.vstack(
        [
            mo.md(r"""## How to reduce variance and bias?"""),
            mo.md(
                r"""
                - Base classifiers should be diverse, i.e., capture as many different pieces of relevant information as possible to reduce ______.
                - The combination rule should reduce variance by smoothing out the noise while aggregating relevant information into the final decision.
                """
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.vstack(
        [
            mo.md(r"""## Bagging (Bootstrap Aggregation) Base classifiers"""),
            mo.image(src=mo.notebook_dir() / "images/bagging.dio.svg").center(),
            mo.md(
                r"""
                - Construct $m$ bootstrap samples.
                - Construct a base classifier for each bootstrap sample.
                """
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.vstack(
        [
            mo.md(r"""#### Bagging (Bootstrap Aggregation) Majority voting"""),
            mo.image(src=mo.notebook_dir() / "images/arch.dio.svg").center(),
            mo.md(
                r"""
                $$\hat{f}(\M{x}) := \arg\max_{\hat{y}} \overbrace{\left( \sum_{j} \mathbb{1} \left( \hat{f}_j(\M{x}) = \hat{y} \right) \right)}^{\left| \left\{ j \mid \hat{f}_j(\M{x}) = \hat{y} \right\} \right| = }$$
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
            mo.image(src=mo.notebook_dir() / "images/bagging_eg1.dio.svg").center(),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.vstack(
        [
            mo.image(src=mo.notebook_dir() / "images/bagging_eg2.dio.svg").center(),
            mo.md(
                r"""
                - Accuracy = _________________________%.
                """
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.vstack(
        [
            mo.md(r"""## Is it always good to follow the majority?"""),
            mo.image(src=mo.notebook_dir() / "images/bagging_eg3.dio.svg").center(),
            mo.md(r"""- Accuracy = _________________________%."""),
            mo.md(r"""- It is beneficial to return 0 more often because _________________________. 
            - How to do this in general?"""),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.vstack(
        [
            mo.md(r"""## Sum rule and threshold moving"""),
            mo.md(
                r"""
                - $\hat{f}(\M{x}) = 1$ iff 
                  $$\frac{1}{2} \left[ \hat{f}_1(\M{x}) + \hat{f}_2(\M{x}) \right] > \underline{\kern5em}$$

                - Binary classification: Choose $\hat{f}(\M{x}) = 1$ iff 
                  $$\frac{1}{m} \sum_{t} \hat{f}_t(\M{x}) > \gamma$$
                  for some chosen threshold $\gamma$.

                - What about multi-class classification?
                """
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.vstack(
        [
            mo.md(r"""#### Bagging (Bootstrap Aggregation) Average of probabilities"""),
            mo.image(src=mo.notebook_dir() / "images/proba.dio.svg").center(),
            mo.md(
                r"""
                $$\hat{f}(\M{x}) := \frac{1}{m} \sum_{t} \hat{f}_t(\M{x}) > \gamma$$
                """
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.vstack(
        [
            mo.md(r"""## Other techniques to diversify base classifiers"""),
            mo.md(
                r"""
                - **Random forest**: Bagging with modified decision tree induction
                    - **Forest-RI**: For each split, consider random i___________________ s___________________ where only $F$ randomly chosen features are considered.
                    - **Forest-RC**: For each split, consider $F$ random l___________________ c___________________ of $L$ randomly chosen features.
                - **Voting** (weka.classifier.meta.vote) and **Stacking** (weka.classifier.meta.stacking): 
                    - Use different classification algorithms.
                - **Adaptive boosting (Adaboost)**:
                    - Each base classifier tries to _______________________________ made by previous base classifiers.
                """
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.vstack(
        [
            mo.md(r"""## Other techniques to combine decisions"""),
            mo.md(
                r"""
                - **Random forest**: Bagging with modified decision tree induction
                    - **Majority voting**
                    - **Average of probabilities**
                - **Voting**
                    - **Majority voting or median**
                    - **Average/product/minimum/maximum probabilities**
                - **Stacking**: Use a meta classifier.
                    - **Adaptive boosting (Adaboost)**: 2003 [Gödel Prize](https://en.wikipedia.org/wiki/G%C3%B6del_Prize) winner
                    - **Weighted majority voting**
                """
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.vstack(
        [
            mo.md(r"""## What is Adaboost?"""),
            mo.md(
                r"""
                - An ensemble method that learns from mistakes:
                  - Combined classifier: Majority voting but with more weight on more accurate base classifier.
                    $$\hat{f}(\M{x}) := \arg\max_{\hat{y}} \sum_{t} w_t \cdot \mathbb{1}((\hat{f}_t)(\M{x}) = \hat{y})$$
                    where 
                    $$w_t := \frac{1}{2} \ln \left( \frac{1 - \text{error}(\hat{f}_t)}{\text{error}(\hat{f}_t)} \right)$$
                    is the amount of say of $\hat{f}_t$ and $\text{error}(\hat{f}_t)$ is the error rate w.r.t. $D_t$. (See the precise formula below.)
                  - Base classifiers: Train $\hat{f}_t$ sequentially in $t$ on $D_t$ obtained by Bagging $(\M{x}_i, \R{Y}_i) \in D$ with
                    $$p_i^{(t)} := \frac{p_i^{(t-1)}}{Z_t} \times \begin{cases} 
                    e^{w_{t-1}}, & \hat{f}_{t-1}(\M{x}_i) \neq \R{Y}_i \text{ (incorrectly classified example)} \\
                    e^{-w_{t-1}}, & \text{otherwise (correctly classified example)}
                    \end{cases}$$
                    starting with $p_i^{(1)} := \frac{1}{|D|}$ and with $Z_t > 0$ chosen so that $\sum_{i} p_i^{(t)} = 1$.
                  - Compute the error rate
                    $$\text{error}(\hat{f}_t) := \sum_{i} p_i^{(t)} \cdot \mathbb{1}((\hat{f}_t)(\M{x}_i) \neq \R{Y}_i)$$
                """
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.vstack(
        [
            mo.md(r"""## Machine vs Machine"""),
            mo.image(src=mo.notebook_dir() / "images/ADB.dio.svg").center(),
        ]
    )
    return


@app.cell(hide_code=True)
def _(machine_vs_machine_fig, mo):
    mo.vstack(
        [
            mo.md(r"""## Segment Challenge: Machine vs Machine"""),
            mo.as_html(machine_vs_machine_fig).center(),
        ]
    )
    return


@app.cell(hide_code=True)
def _(go, mo, pd):
    def plot_machine_vs_machine():

        # Load the data
        rf_data = pd.read_csv(mo.notebook_dir() / "RF.csv")
        adb_data = pd.read_csv(mo.notebook_dir() / "ADB.csv")

        # Create a combined dataframe with an additional column to distinguish the datasets
        rf_data["source"] = "RF"
        adb_data["source"] = "ADB"
        combined_data = pd.concat([rf_data, adb_data])

        # Exclude data points with missing values
        combined_data = combined_data.dropna()

        # Function to filter out dominating points
        def filter_max_accuracy_points(data):
            data = data.sort_values(by="depth")
            filtered_data = []

            for i, row in data.iterrows():
                if not any(
                    (data["depth"] <= row["depth"]) & (data["accuracy"] > row["accuracy"])
                ):
                    filtered_data.append(row)

            return pd.DataFrame(filtered_data)

        # Apply the filtering function for each source
        max_accuracy_points = (
            combined_data.groupby("source")
            .apply(filter_max_accuracy_points, include_groups=False)
            .reset_index(drop=True)
        )

        # Create the scatter plot using go.Scatter
        fig = go.Figure()

        # Add traces for each source
        for source in combined_data["source"].unique():
            source_data = combined_data[combined_data["source"] == source]
            fig.add_trace(
                go.Scatter(
                    x=source_data["depth"],
                    y=source_data["accuracy"],
                    mode="markers+text",
                    text=source_data["name"],
                    name=source,
                    textfont=dict(color="rgba(0,0,0,0)"),  # Make text transparent
                    marker=dict(size=10),
                )
            )

        # Update layout with labels and title
        fig.update_layout(
            title="Machine vs Machine", xaxis_title="Tree Depth", yaxis_title="Accuracy"
        )

        # Add hover information
        fig.update_traces(hovertemplate="<b>%{text}</b><br>Accuracy: %{y}<br>Depth: %{x}")

        # Add annotations for the points with the highest accuracy
        for i, row in max_accuracy_points.iterrows():
            fig.add_annotation(
                x=row["depth"],
                y=row["accuracy"],
                text=f"{row['name']}, {row['accuracy']}",
                showarrow=True,
                arrowhead=2,
                ax=20,
                ay=-30,
                bgcolor="rgba(255, 255, 255, 0.6)",
                opacity=1,
                font=dict(size=10),
                hovertext=f"{row['name']}, {row['accuracy']}",
            )

        return fig


    machine_vs_machine_fig = plot_machine_vs_machine()
    return machine_vs_machine_fig, plot_machine_vs_machine


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## References""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        - Techniques to improve classification accuracy
        - [Witten11] Chapter 8
        - Optional:
            - Breiman, L. (1996). ["Bagging predictors."](https://doi.org/10.1007%2FBF00058655) Machine learning, 24(2), 123-140.
            - Breiman, L. (2001). ["Random forests."](https://doi.org/10.1023%2FA%3A1010933404324) Machine learning, 45(1), 5-32.
            - Freund Y, Schapire R, Abe N. ["A short introduction to boosting."](http://www.yorku.ca/gisweb/eats4400/boost.pdf) Journal-Japanese Society For Artificial Intelligence. 1999 Sep 1;14(771-780):1612.
            - Zhu, H. Zou, S. Rosset, T. Hastie, ["Multi-class AdaBoost"](https://www.intlpress.com/site/pub/files/_fulltext/journals/sii/2009/0002/0003/SII-2009-0002-0003-a008.pdf), 2009.
        """
    )
    return


if __name__ == "__main__":
    app.run()
