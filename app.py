"""
Streamlit web interface for Digital Option Strike Shift Timing.

Run with: streamlit run app.py
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from solver import ShiftTimingSolver, Decision


def main():
    st.set_page_config(
        page_title="Strike Shift Timing",
        page_icon="📊",
        layout="wide"
    )

    st.title("Digital Option Strike Shift Timing")
    st.markdown("""
    **Optimal timing for applying strike shifts on digital options.**

    The tool compares the cost of shifting now vs waiting, using a risk-adjusted
    certainty equivalent framework.

    📚 **Learn the theory:** [Understanding Risk-Neutral Pricing](risk_neutrality.html)
    """)

    # Sidebar inputs
    st.sidebar.header("Option Parameters")

    spot = st.sidebar.number_input(
        "Spot Price",
        min_value=0.01,
        value=1.0850,
        step=0.0001,
        format="%.4f",
        help="Current spot price"
    )

    strike = st.sidebar.number_input(
        "Strike Price",
        min_value=0.01,
        value=1.1000,
        step=0.0001,
        format="%.4f",
        help="Digital option strike"
    )

    vol_pct = st.sidebar.number_input(
        "Volatility (%)",
        min_value=0.1,
        max_value=100.0,
        value=6.0,
        step=0.1,
        help="Annualized volatility"
    )
    vol = vol_pct / 100

    expiry_days = st.sidebar.number_input(
        "Days to Expiry",
        min_value=1,
        max_value=365,
        value=30,
        help="Days until option expiry"
    )

    st.sidebar.header("Shift Parameters")

    shift_bps = st.sidebar.number_input(
        "Shift Size (bps)",
        min_value=1.0,
        max_value=100.0,
        value=5.0,
        step=1.0,
        help="Strike shift size in basis points"
    )

    deadline_days = st.sidebar.number_input(
        "Deadline (days before expiry)",
        min_value=1,
        max_value=60,
        value=14,
        help="Must shift by this many days before expiry"
    )

    st.sidebar.header("Model Settings")

    risk_aversion = st.sidebar.slider(
        "Risk Aversion (γ)",
        min_value=0.0,
        max_value=10.0,
        value=1.0,
        step=0.5,
        help="Higher = more conservative. CE = E[C] + (γ/2)·Var[C]/E[C]"
    )

    n_paths = st.sidebar.selectbox(
        "Simulation Paths",
        options=[1000, 5000, 10000, 25000, 50000],
        index=2,
        help="More paths = more accurate but slower"
    )

    # Run solver
    if st.sidebar.button("Run Analysis", type="primary"):
        with st.spinner("Simulating paths..."):
            solver = ShiftTimingSolver(
                spot=spot,
                strike=strike,
                vol=vol,
                expiry_days=expiry_days,
                shift_bps=shift_bps,
                risk_aversion=risk_aversion,
                deadline_days=deadline_days,
                n_paths=n_paths,
                seed=42,
            )
            result = solver.solve()

        # Store in session state
        st.session_state.result = result
        st.session_state.params = {
            'spot': spot, 'strike': strike, 'vol': vol,
            'expiry_days': expiry_days, 'shift_bps': shift_bps
        }

    # Display results if available
    if 'result' in st.session_state:
        result = st.session_state.result
        params = st.session_state.params

        # Decision banner
        if result.decision == Decision.SHIFT_NOW:
            st.success("## 📍 RECOMMENDATION: SHIFT NOW")
        else:
            st.info("## ⏳ RECOMMENDATION: WAIT")

        # Key metrics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "Cost if Shift Now",
                f"{result.cost_now:.4f}",
                help="Gamma × Shift_bps at current spot & time"
            )

        with col2:
            st.metric(
                "Expected Cost (Wait)",
                f"{result.expected_cost_wait:.4f}",
                delta=f"{(result.expected_cost_wait - result.cost_now):.4f}",
                delta_color="inverse"
            )

        with col3:
            st.metric(
                "CE of Waiting",
                f"{result.ce_wait:.4f}",
                help="Certainty equivalent = E[C] + (γ/2)·Var[C]/E[C]"
            )

        with col4:
            st.metric(
                "Days to Deadline",
                f"{result.days_remaining}",
                help="Days remaining before forced shift"
            )

        # Additional stats
        st.markdown("---")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown(f"**Current Gamma:** {result.gamma_now:.6f}")
            moneyness = np.log(params['spot'] / params['strike'])
            st.markdown(f"**Moneyness (ln S/K):** {moneyness:+.4f}")

        with col2:
            std_wait = np.sqrt(result.variance_cost_wait)
            st.markdown(f"**Std Dev (Wait):** {std_wait:.4f}")
            cv = std_wait / result.expected_cost_wait if result.expected_cost_wait > 0 else 0
            st.markdown(f"**CV (Std/Mean):** {cv:.2%}")

        with col3:
            risk_premium = result.ce_wait - result.expected_cost_wait
            st.markdown(f"**Risk Premium:** {risk_premium:.4f}")
            st.markdown(f"**Risk Aversion (γ):** {result.risk_aversion}")

        # Visualization
        st.markdown("---")
        st.subheader("Gamma Path Distribution")

        if result.gamma_paths is not None:
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=(
                    "Gamma Paths Over Time",
                    "Cost Distribution at Deadline",
                    "Cost Paths Over Time",
                    "Gamma vs Spot at Deadline"
                ),
                vertical_spacing=0.12,
                horizontal_spacing=0.1
            )

            # Sample paths for visualization (show 100 max)
            n_show = min(100, result.gamma_paths.shape[0])
            indices = np.random.choice(result.gamma_paths.shape[0], n_show, replace=False)

            # 1. Gamma paths
            for idx in indices:
                fig.add_trace(
                    go.Scatter(
                        x=result.time_grid,
                        y=result.gamma_paths[idx, :],
                        mode='lines',
                        line=dict(width=0.5, color='rgba(100, 100, 255, 0.2)'),
                        showlegend=False
                    ),
                    row=1, col=1
                )

            # Mean gamma path
            mean_gamma = np.mean(result.gamma_paths, axis=0)
            fig.add_trace(
                go.Scatter(
                    x=result.time_grid,
                    y=mean_gamma,
                    mode='lines',
                    line=dict(width=2, color='blue'),
                    name='Mean Gamma'
                ),
                row=1, col=1
            )

            # Current gamma marker
            fig.add_trace(
                go.Scatter(
                    x=[0],
                    y=[result.gamma_now],
                    mode='markers',
                    marker=dict(size=10, color='red'),
                    name='Current'
                ),
                row=1, col=1
            )

            # 2. Cost distribution at deadline
            costs_at_deadline = result.cost_paths[:, -1]
            fig.add_trace(
                go.Histogram(
                    x=costs_at_deadline,
                    nbinsx=50,
                    name='Cost at Deadline',
                    marker_color='lightblue',
                    showlegend=False
                ),
                row=1, col=2
            )

            # Add vertical lines for key values
            fig.add_vline(
                x=result.cost_now,
                line=dict(color='red', width=2, dash='dash'),
                annotation_text='Cost Now',
                row=1, col=2
            )
            fig.add_vline(
                x=result.ce_wait,
                line=dict(color='green', width=2, dash='dash'),
                annotation_text='CE Wait',
                row=1, col=2
            )

            # 3. Cost paths
            for idx in indices:
                fig.add_trace(
                    go.Scatter(
                        x=result.time_grid,
                        y=result.cost_paths[idx, :],
                        mode='lines',
                        line=dict(width=0.5, color='rgba(255, 100, 100, 0.2)'),
                        showlegend=False
                    ),
                    row=2, col=1
                )

            # Mean cost path
            mean_cost = np.mean(result.cost_paths, axis=0)
            fig.add_trace(
                go.Scatter(
                    x=result.time_grid,
                    y=mean_cost,
                    mode='lines',
                    line=dict(width=2, color='red'),
                    name='Mean Cost'
                ),
                row=2, col=1
            )

            # Cost now line
            fig.add_hline(
                y=result.cost_now,
                line=dict(color='blue', width=1, dash='dot'),
                annotation_text='Cost Now',
                row=2, col=1
            )

            # 4. Gamma vs Spot scatter at deadline
            # We need spot values at deadline - reconstruct from solver
            # For now, use gamma values which correlate with spot
            fig.add_trace(
                go.Scatter(
                    x=result.gamma_paths[:, -1],
                    y=result.cost_paths[:, -1],
                    mode='markers',
                    marker=dict(size=3, color='purple', opacity=0.3),
                    name='Gamma vs Cost at Deadline',
                    showlegend=False
                ),
                row=2, col=2
            )

            fig.update_xaxes(title_text="Days from Now", row=1, col=1)
            fig.update_yaxes(title_text="Gamma", row=1, col=1)
            fig.update_xaxes(title_text="Cost", row=1, col=2)
            fig.update_yaxes(title_text="Frequency", row=1, col=2)
            fig.update_xaxes(title_text="Days from Now", row=2, col=1)
            fig.update_yaxes(title_text="Cost", row=2, col=1)
            fig.update_xaxes(title_text="Gamma at Deadline", row=2, col=2)
            fig.update_yaxes(title_text="Cost at Deadline", row=2, col=2)

            fig.update_layout(
                height=700,
                showlegend=True,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )

            st.plotly_chart(fig, use_container_width=True)

        # Interpretation
        st.markdown("---")
        st.subheader("Interpretation")

        if result.decision == Decision.SHIFT_NOW:
            st.markdown(f"""
            **Why shift now?**

            The current cost ({result.cost_now:.4f}) is lower than the risk-adjusted
            cost of waiting ({result.ce_wait:.4f}).

            Even though the expected cost if you wait ({result.expected_cost_wait:.4f})
            might be similar, the uncertainty (variance) makes waiting less attractive
            given your risk aversion setting (γ = {result.risk_aversion}).
            """)
        else:
            st.markdown(f"""
            **Why wait?**

            The risk-adjusted cost of waiting ({result.ce_wait:.4f}) is lower than
            shifting now ({result.cost_now:.4f}).

            There's a good chance spot will move away from the strike, reducing gamma
            and hence the shift cost. The potential savings outweigh the risk of
            higher costs.
            """)

    else:
        st.info("👈 Set parameters and click 'Run Analysis' to start.")

        # Show formula
        st.markdown("---")
        st.subheader("How it works")

        st.markdown(r"""
        **Cost of shift:**
        $$\text{Cost} = \Gamma(S, K, \tau) \times \text{shift\_bps}$$

        where $\Gamma$ is the vanilla option gamma:
        $$\Gamma = \frac{N'(d_1)}{S \cdot \sigma \cdot \sqrt{\tau}}$$

        **Certainty Equivalent:**
        $$CE = E[\text{Cost}] + \frac{\gamma}{2} \cdot \frac{Var[\text{Cost}]}{E[\text{Cost}]}$$

        **Decision Rule:**
        - If $\text{Cost}_{now} < CE_{wait}$: **SHIFT NOW**
        - Otherwise: **WAIT**
        """)


if __name__ == "__main__":
    main()
