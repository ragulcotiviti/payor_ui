"""
📈  Trends – Trend analysis for payors across years.
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from typing import List, Dict, Tuple

def get_payors_with_complete_data(df: pd.DataFrame) -> pd.DataFrame:
    """Get payors that have data for all years."""
    # Create unique payor-state combinations
    df['payor_state_key'] = df['payor_name'] + ' (' + df['payor_state'] + ')'
    
    # Group by payor_state_key and count unique years
    year_counts = df.groupby('payor_state_key')['year'].nunique()
    complete_payors = year_counts[year_counts >= 3].index.tolist()
    
    # Filter dataframe to only include payors with complete data
    complete_df = df[df['payor_state_key'].isin(complete_payors)].copy()
    complete_df = complete_df.sort_values(['payor_state_key', 'year'])
    
    return complete_df

def calculate_trend_metrics(values: List[float]) -> Dict:
    """Calculate trend metrics for a series of values."""
    if len(values) < 2:
        return {"trend": "Insufficient data", "change_pct": 0, "direction": "stable"}
    
    # Calculate year-over-year changes
    changes = []
    for i in range(1, len(values)):
        if values[i-1] != 0:
            change = ((values[i] - values[i-1]) / values[i-1]) * 100
            changes.append(change)
    
    if not changes:
        return {"trend": "No change", "change_pct": 0, "direction": "stable"}
    
    avg_change = np.mean(changes)
    total_change = ((values[-1] - values[0]) / values[0]) * 100 if values[0] != 0 else 0
    
    # Determine trend direction
    if avg_change > 5:
        direction = "increasing"
    elif avg_change < -5:
        direction = "decreasing"
    else:
        direction = "stable"
    
    return {
        "trend": direction,
        "avg_change_pct": avg_change,
        "total_change_pct": total_change,
        "direction": direction
    }

def create_trend_chart(df: pd.DataFrame, payor_state_key: str, feature: str) -> go.Figure:
    """Create an interactive trend chart for a specific payor-state combination and feature."""
    payor_data = df[df['payor_state_key'] == payor_state_key].copy()
    
    if payor_data.empty:
        return None
    
    # Aggregate data by year to avoid overlapping points
    payor_data = payor_data.groupby('year')[feature].mean().reset_index()
    payor_data = payor_data.sort_values('year')
    
    # Prepare data for plotting
    years = payor_data['year'].tolist()
    values = payor_data[feature].tolist()
    
    # Calculate trend metrics
    trend_metrics = calculate_trend_metrics(values)
    
    # Create the line chart
    fig = go.Figure()
    
    # Determine hover template and y-axis formatting based on feature type
    if 'share' in feature.lower():
        hover_template = '<b>%{x}</b><br>' + f'{feature}: %{{y:.2%}}<br>' + '<extra></extra>'
        trend_hover_template = '<b>Trend</b><br>' + f'{feature}: %{{y:.2%}}<br>' + '<extra></extra>'
        yaxis_format = '.2%'
    else:
        hover_template = '<b>%{x}</b><br>' + f'{feature}: %{{y:,.0f}}<br>' + '<extra></extra>'
        trend_hover_template = '<b>Trend</b><br>' + f'{feature}: %{{y:,.0f}}<br>' + '<extra></extra>'
        yaxis_format = ',.0f'
    
    # Add line trace
    fig.add_trace(go.Scatter(
        x=years,
        y=values,
        mode='lines+markers',
        name=payor_state_key,
        line=dict(color='#636EFA', width=3),
        marker=dict(size=8, color='#636EFA'),
        hovertemplate=hover_template
    ))
    
    # Add trend line (linear regression)
    if len(years) >= 2:
        z = np.polyfit(range(len(years)), values, 1)
        p = np.poly1d(z)
        trend_line = p(range(len(years)))
        
        fig.add_trace(go.Scatter(
            x=years,
            y=trend_line,
            mode='lines',
            name='Trend Line',
            line=dict(color='#EF553B', width=2, dash='dash'),
            hovertemplate=trend_hover_template
        ))
    
    # Update layout
    fig.update_layout(
        title=f"{payor_state_key} - {feature.replace('_', ' ').title()} Trend",
        xaxis_title="Year",
        yaxis_title=feature.replace('_', ' ').title(),
        height=400,
        showlegend=True,
        hovermode='x unified',
        xaxis=dict(
            tickmode='array',
            tickvals=[2021, 2022, 2023],
            ticktext=['2021', '2022', '2023'],
            type='category'
        ),
        yaxis=dict(
            tickformat=yaxis_format
        )
    )
    
    return fig, trend_metrics

def format_value(value: float, feature: str) -> str:
    """Format value based on feature type."""
    if 'cost' in feature.lower() or 'premium' in feature.lower() or 'amount' in feature.lower():
        return f"${value:,.0f}"
    elif 'share' in feature.lower():
        return f"{value:.2%}"
    elif 'count' in feature.lower() or 'population' in feature.lower():
        return f"{value:,.0f}"
    else:
        return f"{value:,.2f}"

def page(df_raw=None) -> None:
    st.header(" Trend Analysis")
    
    if df_raw is None:
        st.info("Please upload data first to view trends.")
        return
    
    # Get payors with complete data (all 3 years)
    complete_df = get_payors_with_complete_data(df_raw)
    
    if complete_df.empty:
        st.warning("No payors found with data for all 3 years. Please upload data with complete year coverage.")
        return
    
    # Display summary
    unique_payors = complete_df['payor_name'].nunique()
    years_available = sorted(complete_df['year'].unique())
    
    # Create three columns for selection
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Payor name selection
        payor_names = sorted(complete_df['payor_name'].unique())
        selected_payor_name = st.selectbox(
            "Select Payor",
            options=payor_names,
            help="Choose a payor to analyze trends"
        )
    
    with col2:
        # State selection (filtered based on selected payor)
        if selected_payor_name:
            available_states = sorted(complete_df[complete_df['payor_name'] == selected_payor_name]['payor_state'].unique())
            selected_state = st.selectbox(
                "Select State",
                options=available_states,
                help="Choose a state for the selected payor"
            )
        else:
            selected_state = None
    
    with col3:
        # Feature selection
        numeric_features = complete_df.select_dtypes(include=[np.number]).columns.tolist()
        # Remove year, ranking columns, and payor_code
        feature_options = [f for f in numeric_features if f not in ['year', 'rank', 'FINAL_RANK', 'RANK_SCORE', 'payor_code']]
        
        # Set default feature to premium if available, otherwise use first available feature
        default_feature = "premium" if "premium" in feature_options else feature_options[0] if feature_options else None
        
        selected_feature = st.selectbox(
            "Select Feature",
            options=feature_options,
            index=feature_options.index(default_feature) if default_feature and default_feature in feature_options else 0,
            help="Choose a metric to analyze"
        )
    
    if selected_payor_name and selected_state and selected_feature:
        # Create the payor_state_key for filtering
        selected_payor_key = f"{selected_payor_name} ({selected_state})"
        
        # Get payor data
        payor_data = complete_df[complete_df['payor_state_key'] == selected_payor_key].sort_values('year')
        
        if not payor_data.empty:
            # Check if we have data for all years
            available_years = sorted(payor_data['year'].unique())
            all_years = sorted(complete_df['year'].unique())
            
            if len(available_years) >= 3:
                # Create trend chart
                fig, trend_metrics = create_trend_chart(complete_df, selected_payor_key, selected_feature)
                
                if fig:
                    # Display the chart
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Display trend insights
                    st.markdown("### Trend Insights")
                    
                    # Create metrics display
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        # Determine color based on trend direction
                        trend_direction = trend_metrics["direction"]
                        trend_bg = "rgba(239, 68, 68, 0.15)" if trend_direction == "decreasing" else "rgba(34, 197, 94, 0.15)" if trend_direction == "increasing" else "rgba(156, 163, 175, 0.15)"
                        trend_border = "rgba(239, 68, 68, 0.4)" if trend_direction == "decreasing" else "rgba(34, 197, 94, 0.4)" if trend_direction == "increasing" else "rgba(156, 163, 175, 0.4)"
                        trend_arrow = "↓" if trend_direction == "decreasing" else "↑" if trend_direction == "increasing" else "→"
                        st.markdown(f"""
                        <div style="background: rgba(255,255,255,0.05); padding: 1rem; border-radius: 8px; margin: 1rem 0;">
                            <div style="font-size: 0.875rem; color: #9CA3AF; margin-bottom: 0.5rem;">Trend Direction</div>
                            <div style="font-size: 1.5rem; font-weight: 600; color: white; margin-bottom: 0.5rem;">{trend_metrics["direction"].title()}</div>
                            <div style="background: {trend_bg}; border: 1px solid {trend_border}; color: white; font-size: 0.875rem; padding: 0.5rem 0.75rem; border-radius: 4px; display: inline-block; font-weight: 500;">{trend_arrow} {trend_metrics['avg_change_pct']:.1f}% avg change</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col2:
                        total_change = trend_metrics['total_change_pct']
                        # Determine color based on total change direction
                        total_bg = "rgba(239, 68, 68, 0.15)" if total_change < 0 else "rgba(34, 197, 94, 0.15)" if total_change > 0 else "rgba(156, 163, 175, 0.15)"
                        total_border = "rgba(239, 68, 68, 0.4)" if total_change < 0 else "rgba(34, 197, 94, 0.4)" if total_change > 0 else "rgba(156, 163, 175, 0.4)"
                        total_arrow = "↓" if total_change < 0 else "↑" if total_change > 0 else "→"
                        st.markdown(f"""
                        <div style="background: rgba(255,255,255,0.05); padding: 1rem; border-radius: 8px; margin: 1rem 0;">
                            <div style="font-size: 0.875rem; color: #9CA3AF; margin-bottom: 0.5rem;">Total Change</div>
                            <div style="font-size: 1.5rem; font-weight: 600; color: white; margin-bottom: 0.5rem;">{total_change:.1f}%</div>
                            <div style="background: {total_bg}; border: 1px solid {total_border}; color: white; font-size: 0.875rem; padding: 0.5rem 0.75rem; border-radius: 4px; display: inline-block; font-weight: 500;">{total_arrow} over entire period</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col3:
                        # Get aggregated data for first and last values
                        agg_data = payor_data.groupby('year')[selected_feature].mean().reset_index().sort_values('year')
                        values = agg_data[selected_feature].tolist()
                        if len(values) >= 2:
                            first_value = values[0]
                            last_value = values[-1]
                            # Determine color based on whether current value is higher or lower than first value
                            current_bg = "rgba(239, 68, 68, 0.15)" if last_value < first_value else "rgba(34, 197, 94, 0.15)" if last_value > first_value else "rgba(156, 163, 175, 0.15)"
                            current_border = "rgba(239, 68, 68, 0.4)" if last_value < first_value else "rgba(34, 197, 94, 0.4)" if last_value > first_value else "rgba(156, 163, 175, 0.4)"
                            current_arrow = "↓" if last_value < first_value else "↑" if last_value > first_value else "→"
                            st.markdown(f"""
                            <div style="background: rgba(255,255,255,0.05); padding: 1rem; border-radius: 8px; margin: 1rem 0;">
                                <div style="font-size: 0.875rem; color: #9CA3AF; margin-bottom: 0.5rem;">Current Value</div>
                                <div style="font-size: 1.5rem; font-weight: 600; color: white; margin-bottom: 0.5rem;">{format_value(last_value, selected_feature)}</div>
                                <div style="background: {current_bg}; border: 1px solid {current_border}; color: white; font-size: 0.875rem; padding: 0.5rem 0.75rem; border-radius: 4px; display: inline-block; font-weight: 500;">{current_arrow} from {format_value(first_value, selected_feature)}</div>
                            </div>
                            """, unsafe_allow_html=True)
                    
                    # Detailed trend analysis
                    st.markdown("#### Detailed Analysis")
                    
                    # Show year-by-year changes using aggregated data
                    years = agg_data['year'].tolist()
                    values = agg_data[selected_feature].tolist()
                    
                    if len(years) >= 2:
                        changes = []
                        for i in range(1, len(years)):
                            if values[i-1] != 0:
                                change_pct = ((values[i] - values[i-1]) / values[i-1]) * 100
                                changes.append({
                                    'period': f"{years[i-1]} → {years[i]}",
                                    'change': change_pct,
                                    'value': values[i]
                                })
                        
                        if changes:
                            st.markdown("**Year-over-Year Changes:**")
                            for change in changes:
                                change_color = "🟢" if change['change'] > 0 else "🔴" if change['change'] < 0 else "🟡"
                                st.markdown(f"{change_color} **{change['period']}**: {change['change']:+.1f}% → {format_value(change['value'], selected_feature)}")
                    
                    # Trend interpretation
                    st.markdown("#### Trend Interpretation")
                    if trend_metrics['direction'] == 'increasing':
                        st.info(f" **Growing Trend**: {selected_payor_name} in {selected_state} shows a consistent increase in {selected_feature.replace('_', ' ')} over the analyzed period.")
                    elif trend_metrics['direction'] == 'decreasing':
                        st.warning(f" **Declining Trend**: {selected_payor_name} in {selected_state} shows a consistent decrease in {selected_feature.replace('_', ' ')} over the analyzed period.")
                    else:
                        st.success(f" **Stable Trend**: {selected_payor_name} in {selected_state} maintains relatively stable {selected_feature.replace('_', ' ')} levels.")
            else:
                # Show available data even if incomplete
                st.warning(f" **Incomplete Data**: {selected_payor_name} in {selected_state} only has data for {len(available_years)} year(s): {', '.join(map(str, available_years))}")
                st.info(f" **Available Years**: {', '.join(map(str, available_years))} out of {', '.join(map(str, all_years))}")
                
                # Still show the chart with available data
                if len(available_years) >= 2:
                    fig, trend_metrics = create_trend_chart(complete_df, selected_payor_key, selected_feature)
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                        st.info("📈 **Limited Trend Analysis**: Chart shows available data, but trend analysis may be incomplete due to missing years.")
                else:
                    st.error(" **Insufficient Data**: Need at least 2 years of data for trend analysis.")
        else:
            st.error(f" **No Data Found**: {selected_payor_name} in {selected_state} has no data in the dataset.")
    else:
        st.info("Please select a payor, state, and feature to view trends.")