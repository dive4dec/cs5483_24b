import marimo

__generated_with = "0.11.6"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo

    mo.md("# Different Evaluation Metrics")
    return (mo,)


@app.cell(hide_code=True)
def _():
    import numpy as np
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    # Create a meshgrid for x and y in the range [0.001, 1]
    x = np.linspace(0.001, 1, 100)
    y = np.linspace(0.001, 1, 100)
    X, Y = np.meshgrid(x, y)

    # Function to create a surface trace
    def create_surface_trace(X, Y, Z):
        return go.Surface(
            x=X, y=Y, z=Z, colorscale="Viridis", colorbar=dict(title="z"), opacity=0.4,
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

    # Arithmetic mean
    am = (X + Y) / 2
    am_trace = create_surface_trace(X, Y, am)

    # Harmonic mean
    hm = ((X**-1 + Y**-1) / 2) ** -1
    hm_trace = create_surface_trace(X, Y, hm)

    # Create subplots
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(
            r"$\text{Arithmetic Mean }z=\frac{x+y}{2}$", 
            r"$\text{Harmonic Mean }z=\Big(\frac{x^{-1}+y^{-1}}{2}\Big)^{-1}$"
        ),
        specs=[[{'type': 'surface'}, {'type': 'surface'}]]
    )

    # Add surface plots to subplots
    fig.add_trace(am_trace, row=1, col=1)
    fig.add_trace(hm_trace, row=1, col=2)
    fig.update_scenes(camera=dict(eye=dict(x=1.4, y=-1.4, z=1.4)),
                      xaxis_title="x",
                      yaxis_title="y",
                      zaxis_title="z")

    # Update layout
    fig.update_layout(
        title_text="3D Surface Plots of Arithmetic and Harmonic Means",
        width=1000,
        height=600,
    )

    fig.show()
    return (
        X,
        Y,
        am,
        am_trace,
        create_surface_trace,
        fig,
        go,
        hm,
        hm_trace,
        make_subplots,
        np,
        x,
        y,
    )


@app.cell(hide_code=True)
def _(X, Y, go, np):
    # Function to compute F-beta score
    def f_beta_score(X, Y, beta):
        precision = X
        recall = Y
        return (1 + beta**2) * (precision * recall) / (beta**2 * precision + recall)

    # Create the initial F-beta score surface plot
    beta = 1.0
    F_beta = f_beta_score(X, Y, beta)

    fig_F_beta = go.Figure(
        data=[
            go.Surface(
                x=X, y=Y, z=F_beta, colorscale="Viridis", colorbar=dict(title="F-β"), opacity=0.4
            )
        ]
    )

    # Update traces for contours
    fig_F_beta.update_traces(
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
                        {"z": [f_beta_score(X, Y, beta)]},
                        {"title": f"F-β Score Surface Plot (β = {beta:.2f})"}
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
    fig_F_beta.update_layout(
        title=r"F-β Score Surface Plot (β = 1.0)",
        autosize=False,
        scene=dict(
            xaxis_title="PPV",
            yaxis_title="TPR",
            zaxis_title="F-β Score",
            camera_eye=dict(x=1.3, y=-1.3, z=1.3),
        ),
        width=600,
        height=600,
        margin=dict(l=65, r=50, b=65, t=90),
        sliders=sliders,
    )

    fig_F_beta.show()
    return F_beta, beta, f_beta_score, fig_F_beta, log_beta_values, sliders


if __name__ == "__main__":
    app.run()
