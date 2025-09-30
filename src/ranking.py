"""
🏅  Ranking / clustering page.
"""
from __future__ import annotations
import streamlit as st
import pandas as pd
import numpy as np
# from openai import OpenAI
import openai
import plotly.graph_objects as go
import plotly.express as px
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from src.upload import add_rankings, _clean
from streamlit_plotly_events import plotly_events

# # # ──────────────────────────────────────────────────────────────
# # # OpenAI API setup
# openai = OpenAI(api_key="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")
# # Azure OpenAI client
# ──────────────────────────────────────────────────────────────
client = openai.AzureOpenAI(
    api_key        = "936856630b764210913d9a8fd6c8212b",
    azure_endpoint = "https://ironclad-openai-001.openai.azure.com/",
    api_version    = "2024-02-15-preview",
)
DEPLOYMENT_CHAT = "gpt‑4o"    
st.set_page_config(layout="wide", page_title="Payor Strategy Explorer")

# ──────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────
PALETTE = {0: "darkred", 1: "darkgreen", 2: "darkblue"}

NUM = [
    "premium", "population", "unique_member_count", "unique_provider_count",
    "unique_claim_count", "unique_claimant_count", "market_share",
]
BIN = ["type_L", "type_H", "is_commercial", "is_medicare", "is_medicaid"]

# ──────────────────────────────────────────────────────────────
# Ranking Algorithm from payorbenchmarking-poc
# ──────────────────────────────────────────────────────────────
# ------------------------------------------------------------------
# Ranking algorithm
# ------------------------------------------------------------------
def data_ranking(df: pd.DataFrame) -> pd.DataFrame:
    ranked_parts: list[pd.DataFrame] = []

    for yr, grp in df.groupby("year"):
        grp = grp.copy()
        if {"medical_cost", "administrative_cost"}.issubset(grp.columns):
            mc_col, ac_col = "medical_cost", "administrative_cost"
        else:                 # nothing to rank this year
            ranked_parts.append(grp)
            continue

        grp["MEDICAL_COST_RANK"]        = grp[mc_col].rank(method="max", ascending=True).astype(int)
        grp["ADMINISTRATIVE_COST_RANK"] = grp[ac_col].rank(method="min", ascending=True).astype(int)
        grp["RANK_SCORE"]               = grp["MEDICAL_COST_RANK"] * grp["ADMINISTRATIVE_COST_RANK"]
        grp["FINAL_RANK"]               = grp["RANK_SCORE"].rank(method="min").astype(int)
        ranked_parts.append(grp.sort_values("FINAL_RANK"))

    return pd.concat(ranked_parts, ignore_index=True)


# ──────────────────────────────────────────────────────────────
# Payor Comparison Functions from payorbenchmarking-poc
# ──────────────────────────────────────────────────────────────
def find_higher_ranked_payor(df, selected_payor_name):
    """
    Finds the payor ranked immediately higher than the selected payor.
    Returns None if the selected payor is rank 1.
    """
    try:
        selected_payor = df[df['payor_name'] + ' - ' + df['payor_state'] == selected_payor_name].iloc[0]
        rank = selected_payor['rank']
        next_rank = rank - 1

        if next_rank < 1:
            return None # No one is ranked higher than rank 1

        next_payor_df = df[df['rank'] == next_rank]
        return next_payor_df.iloc[0] if not next_payor_df.empty else None
    except (IndexError, KeyError):
        return None

# @st.cache_data(show_spinner=False)
# def generate_insights(payor1, payor2, max_characters=500):
#     """
#     Generates insights by comparing two payors using their full data.
#     """
#     # Check if payor2 exists (better ranked payor)
#     if payor2 is None:
#         prompt = f"""
#         {payor1['payor_name']} has demonstrated the following behaviors:
#         - Premium: ${payor1.get('premium', 0):,.0f}
#         - Population: {payor1.get('population', 0):,.0f}
#         - Market Share: {payor1.get('market_share', 0):.2%}
#         - Medical Cost: ${payor1.get('Medical Cost (MC)', payor1.get('MC', 0)):,.0f}
#         - Administrative Cost: ${payor1.get('Administrative Cost (AC)', payor1.get('AC', 0)):,.0f}
#         - Final Rank: {payor1.get('rank', 'N/A')}
#         
#         This payor is currently ranked #1 in their cluster, which means they are performing at the highest level among their peers. 
#         Provide insights in the following format:
#         
#         **Key Strengths:**
#         • [First strength]
#         • [Second strength]
#         • [Third strength]
#         
#         **Strategic Analysis:**
#         • [First strategic insight]
#         • [Second strategic insight]
#         
#         **Recommendations for {payor1['payor_name']}:**
#         • [First recommendation to maintain position]
#         • [Second recommendation for continued success]
#         
#         Keep each bullet point concise and actionable. Focus on their competitive advantages and market positioning.
#         """
#     else:
#         prompt = f"""
#         {payor1['payor_name']} has demonstrated the following behaviors:
#         - Premium: ${payor1.get('premium', 0):,.0f}
#         - Population: {payor1.get('population', 0):,.0f}
#         - Market Share: {payor1.get('market_share', 0):.2%}
#         - Medical Cost: ${payor1.get('Medical Cost (MC)', payor1.get('MC', 0)):,.0f}
#         - Administrative Cost: ${payor1.get('Administrative Cost (AC)', payor1.get('AC', 0)):,.0f}
#         - Final Rank: {payor1.get('rank', 'N/A')}
#         
#         your better ranked payor, on the other hand, shows these behaviors:
#         - Premium: ${payor2.get('premium', 0):,.0f}
#         - Population: {payor2.get('population', 0):,.0f}
#         - Market Share: {payor2.get('market_share', 0):.2%}
#         - Medical Cost: ${payor2.get('Medical Cost (MC)', payor2.get('MC', 0)):,.0f}
#         - Administrative Cost: ${payor2.get('Administrative Cost (AC)', payor2.get('AC', 0)):,.0f}
#         - Final Rank: {payor2.get('rank', 'N/A')}
#         
#         Based on the data provided, analyze these two payors and provide insights in the following format:
#         
#         **Key Differences:**
#         • [First key difference]
#         • [Second key difference]
#         • [Third key difference]
#         
#         **Strategic Analysis:**
#         • [First strategic insight]
#         • [Second strategic insight]
#         
#         **Recommendations for {payor1['payor_name']}:**
#         • [First recommendation]
#         • [Second recommendation]
#         • [Third recommendation]
#         
#         Keep each bullet point concise and actionable. Focus on the most important insights and recommendations.
#         """
#     
#     try:
#         response = openai.chat.completions.create(
#             model="gpt-4o",
#             messages=[
#                 {"role": "system", "content": "You are a helpful AI assistant for financial analysis in the healthcare sector."},
#                 {"role": "user", "content": prompt},
#             ],
#             max_tokens=max_characters,
#             temperature=0.7,
#             timeout=30  # 30 second timeout for insights generation
#         )
#         return response.choices[0].message.content.strip()
#     except Exception as e:
#         return f"Error generating insights: {e}"


@st.cache_data(show_spinner=False)
def generate_insights(payor1, payor2, max_characters=500):
    """
    Generates insights by comparing two payors using their full data.
    """
    # Check if payor2 exists (better ranked payor)
    if payor2 is None:
        prompt = f"""
        {payor1['payor_name']} has demonstrated the following behaviors:
        - Premium: ${payor1.get('premium', 0):,.0f}
        - Population: {payor1.get('population', 0):,.0f}
        - Market Share: {payor1.get('market_share', 0):.2%}
        - Medical Cost: ${payor1.get('Medical Cost (MC)', payor1.get('MC', 0)):,.0f}
        - Administrative Cost: ${payor1.get('Administrative Cost (AC)', payor1.get('AC', 0)):,.0f}
        - Final Rank: {payor1.get('rank', 'N/A')}
        
        This payor is currently ranked #1 in their cluster, which means they are performing at the highest level among their peers. 
        Provide insights in the following format:
        
        **Key Strengths:**
        • [First strength]
        • [Second strength]
        • [Third strength]
        
        **Strategic Analysis:**
        • [First strategic insight]
        • [Second strategic insight]
        
        **Recommendations for {payor1['payor_name']}:**
        • [First recommendation to maintain position]
        • [Second recommendation for continued success]
        
        Keep each bullet point concise and actionable. Focus on their competitive advantages and market positioning.
        """
    else:
        prompt = f"""
        {payor1['payor_name']} has demonstrated the following behaviors:
        - Premium: ${payor1.get('premium', 0):,.0f}
        - Population: {payor1.get('population', 0):,.0f}
        - Market Share: {payor1.get('market_share', 0):.2%}
        - Medical Cost: ${payor1.get('Medical Cost (MC)', payor1.get('MC', 0)):,.0f}
        - Administrative Cost: ${payor1.get('Administrative Cost (AC)', payor1.get('AC', 0)):,.0f}
        - Final Rank: {payor1.get('rank', 'N/A')}
        
        your better ranked payor, on the other hand, shows these behaviors:
        - Premium: ${payor2.get('premium', 0):,.0f}
        - Population: {payor2.get('population', 0):,.0f}
        - Market Share: {payor2.get('market_share', 0):.2%}
        - Medical Cost: ${payor2.get('Medical Cost (MC)', payor2.get('MC', 0)):,.0f}
        - Administrative Cost: ${payor2.get('Administrative Cost (AC)', payor2.get('AC', 0)):,.0f}
        - Final Rank: {payor2.get('rank', 'N/A')}
        
        Based on the data provided, analyze these two payors and provide insights in the following format:
        
        **Key Differences:**
        • [First key difference]
        • [Second key difference]
        • [Third key difference]
        
        **Strategic Analysis:**
        • [First strategic insight]
        • [Second strategic insight]
        
        **Recommendations for {payor1['payor_name']}:**
        • [First recommendation]
        • [Second recommendation]
        • [Third recommendation]
        
        Keep each bullet point concise and actionable. Focus on the most important insights and recommendations.
        """
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a helpful AI assistant for financial analysis in the healthcare sector."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=max_characters,
            temperature=0.7,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error generating insights: {e}"
# ──────────────────────────────────────────────────────────────
# PCA Biplot with feature loadings
# ──────────────────────────────────────────────────────────────
def create_pca_biplot(X, labels, df_meta, year, feature_names, features_to_display=None):
    """
    Performs PCA and creates a single interactive biplot with loadings.
    """
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X)

    # Combine original metadata with PCA results for hovering
    meta = df_meta.copy()
    meta['PC1'] = X_pca[:, 0]
    meta['PC2'] = X_pca[:, 1]
    meta['cluster'] = labels.astype(str)

    # Create the interactive scatter plot (removed payor_name from hover_data for privacy)
    fig = px.scatter(
        meta,
        x='PC1',
        y='PC2',
        color='cluster',
        hover_data=['payor_state', 'unique_member_count', 'rank'],
        title=f"PCA Biplot with Feature Loadings · {year}"
    )

    # Remove PC1/PC2 axis labels and ticks
    fig.update_layout(
        xaxis_title=None,
        yaxis_title=None,
        xaxis_showticklabels=False,
        yaxis_showticklabels=False
    )

    # Add feature loadings (restore arrows/annotations)
    loadings = pca.components_.T * np.sqrt(pca.explained_variance_)

    if features_to_display is None:
        features_to_display = [f for f in feature_names if f not in ["PC1", "PC2"]]

    for i, feature in enumerate(feature_names):
        if feature in features_to_display and feature not in ["PC1", "PC2"]:
            fig.add_annotation(
                ax=0, ay=0, axref="x", ayref="y",
                x=loadings[i, 0], y=loadings[i, 1], xref="x", yref="y",
                showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=2, arrowcolor="#636EFA"
            )
            fig.add_annotation(
                x=loadings[i, 0] * 1.15, y=loadings[i, 1] * 1.15, ax=0, ay=0,
                xref="x", yref="y", text=feature, showarrow=False
            )

    return fig, X_pca

# ──────────────────────────────────────────────────────────────
# Clustering pipeline
# ──────────────────────────────────────────────────────────────
def _prep() -> ColumnTransformer:
    return ColumnTransformer([
        ("num", StandardScaler(), NUM),
        ("bin", "passthrough", BIN),
        ("state", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["payor_state"]),
    ])

@st.cache_data(show_spinner=False)
def _cluster(df: pd.DataFrame) -> pd.DataFrame:
    # Filter to only include columns that exist
    available_num = [col for col in NUM if col in df.columns]
    available_bin = [col for col in BIN if col in df.columns]
    
    if not available_num:
        st.error("No numeric columns available for clustering")
        return df
        
    # Create feature matrix with available columns
    feature_cols = available_num + available_bin + ["payor_state"]
    preprocess = _prep()
    X = preprocess.fit_transform(df[feature_cols])
    
    km = KMeans(n_clusters=3, n_init="auto", random_state=42).fit(X)
    pca = PCA(n_components=2, random_state=42).fit_transform(X)
    out = df.copy()
    out["cluster"] = km.labels_
    out["pca1"], out["pca2"] = pca[:, 0], pca[:, 1]
    return out

# ──────────────────────────────────────────────────────────────
# GPT-based cluster labeling
# ──────────────────────────────────────────────────────────────
# Predefined cluster labels to prevent hallucination and ensure uniqueness
CLUSTER_LABELS = {
    0: "High Premium Segment",
    1: "Large Network Providers",
    2: "Comprehensive Insurers",
    3: "High Utilization Segment",
    4: "Complex Product Mix",
    5: "Market Leaders",
    6: "Regional Specialists",
    7: "Cost Efficient Segment"
}

@st.cache_data(show_spinner=False)
def _calculate_cluster_centroids(df: pd.DataFrame) -> dict:
    """Calculate K-means centroids for each feature across all clusters."""
    centroids = {}
    
    # Features to calculate centroids for
    feature_cols = ['premium', 'population', 'unique_member_count', 'unique_provider_count', 
                   'unique_claim_count', 'market_share']
    
    for feature in feature_cols:
        if feature in df.columns:
            # Calculate centroids for this feature across all clusters
            feature_centroids = {}
            for cluster_id in range(3):  # Assuming 3 clusters
                cluster_data = df[df['cluster'] == cluster_id][feature]
                if not cluster_data.empty:
                    feature_centroids[cluster_id] = cluster_data.mean()
                else:
                    feature_centroids[cluster_id] = 0
            
            centroids[feature] = feature_centroids
    
    # Add geographic coverage centroids
    if 'payor_state' in df.columns:
        geo_centroids = {}
        for cluster_id in range(3):
            cluster_data = df[df['cluster'] == cluster_id]['payor_state']
            if not cluster_data.empty:
                geo_centroids[cluster_id] = cluster_data.nunique()
            else:
                geo_centroids[cluster_id] = 0
        centroids['geographic_coverage'] = geo_centroids
    
    return centroids

def _label_with_uniqueness(df_sub: pd.DataFrame, cid: int, centroids: dict = None, used_labels: set = None) -> str:
    """Generate unique cluster label based on actual cluster characteristics for the specific year."""
    if used_labels is None:
        used_labels = set()
    
    # Calculate cluster centroids for validation
    cluster_centroids = {}
    if centroids:
        for feature, feature_centroids in centroids.items():
            if cid in feature_centroids:
                cluster_centroids[feature] = feature_centroids[cid]
    
    # Validate feature dominance using centroids
    dominant_features = []
    if cluster_centroids:
        # Find which features are dominant for this cluster compared to others
        for feature, cluster_value in cluster_centroids.items():
            if feature in centroids:
                other_clusters = [centroids[feature][i] for i in range(3) if i != cid]
                if other_clusters and cluster_value > max(other_clusters):
                    dominant_features.append(feature)
    
    # DATA VALIDATION: Verify cluster characteristics match the label
    cluster_stats = _calculate_cluster_statistics(df_sub)
    
    # Get all available labels
    all_labels = ['High Premium Segment', 'Large Network Providers', 'Comprehensive Insurers', 'High Utilization Segment', 'Complex Product Mix', 'Market Leaders', 'Regional Specialists', 'Cost Efficient Segment']
    
    # Filter out already used labels
    available_labels = [label for label in all_labels if label not in used_labels]
    
    # Fallback: analyze cluster characteristics to select most appropriate label
    prompt = (
        "PRIMARY ANALYSIS: Use the centroid data and column analysis to determine the cluster name.\n\n"
        "CLUSTER CENTROIDS (MOST IMPORTANT):\n"
        f"Cluster {cid} dominant features: {dominant_features}\n"
        f"Cluster {cid} centroids: {cluster_centroids}\n\n"
        "CLUSTER STATISTICS (VERIFICATION):\n"
        f"Cluster {cid} actual averages: {cluster_stats}\n\n"
        "Available features to analyze:\n"
        "- population: Total population served\n"
        "- unique_member_count: Number of unique members enrolled\n"
        "- premium: Total premium revenue\n"
        "- unique_provider_count: Number of unique healthcare providers in network\n"
        "- unique_claim_count: Number of unique claims processed\n"
        "- market_share: Market share percentage\n"
        "- payor_state: Geographic coverage (number of states)\n\n"
        "Instructions:\n"
        "1. FIRST: Analyze the dominant features from centroids - these define the cluster's primary characteristics\n"
        "2. SECOND: Verify the cluster statistics match the dominant features\n"
        "3. THIRD: Ensure the selected label is justified by BOTH centroid dominance AND actual data averages\n"
        "4. FOURTH: Select from AVAILABLE labels only (avoid duplicates)\n\n"
        "Label selection based on DOMINANT CENTROID FEATURES (HIGH PRIORITY ORDER with MODERATE thresholds):\n"
        "PRIORITY 1 - High Premium Segment: When premium is dominant AND premium centroid > $3,000,000,000 (3 billion)\n"
        "PRIORITY 2 - Large Network Providers: When unique_provider_count is dominant AND unique_provider_count centroid > 30,000 providers\n"
        "PRIORITY 3 - High Utilization Segment: When unique_claim_count is dominant AND unique_claim_count centroid > 300,000 claims\n"
        "PRIORITY 4 - Market Leaders: When market_share is dominant AND market_share centroid > 20%\n"
        "PRIORITY 5 - Regional Specialists: When market_share is dominant AND market_share centroid > 12% BUT geographic_coverage centroid < 10 states\n"
        "PRIORITY 6 - Cost Efficient Segment: When premium is NOT dominant AND premium centroid < $100,000,000\n"
        "PRIORITY 7 - Comprehensive Insurers: When 3+ features are dominant AND premium centroid > $1,500,000,000 AND population centroid > 8,000,000 AND unique_member_count centroid > 1,500,000\n"
        "PRIORITY 8 - Complex Product Mix: When BOTH unique_member_count AND unique_claim_count are dominant AND unique_member_count centroid > 3,000,000 AND unique_claim_count centroid > 150,000\n\n"
        f"AVAILABLE LABELS (choose from these only): {available_labels}\n\n"
        "CRITICAL: The dominant features from centroids should be the PRIMARY factor in determining the cluster name.\n"
        "VERIFICATION: Ensure the selected label is justified by actual cluster data averages.\n"
        "IMPORTANT: Generate labels based on THIS YEAR'S data characteristics, not predefined patterns.\n"
        "UNIQUENESS: Select only from the available labels to avoid duplicates.\n"
        "Return ONLY the label name, nothing else.\n\n"
        f"Cluster data (first 5 rows):\n{df_sub.head().to_markdown(index=False)}"
    )
    
    try:
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,  # Low temperature for consistent selection
            max_tokens=50,
            timeout=10  # 10 second timeout
        )
        label = response.choices[0].message.content.strip()
        
        # Clean up the response and validate it's from our available list
        label = label.replace('"', '').replace("'", "").strip()
        
        if label in available_labels:
            # VERIFY the label is justified by data
            if _verify_label_justification(label, cluster_stats, dominant_features):
                return label
            else:
                # If not justified, select a more appropriate label from available ones
                return _select_best_label_from_available(df_sub, available_labels)
        else:
            # If response is not in our available list, select based on cluster characteristics
            return _select_best_label_from_available(df_sub, available_labels)
            
    except Exception as e:
        # Fallback to characteristic-based selection from available labels
        return _select_best_label_from_available(df_sub, available_labels)

def _select_best_label_from_available(df_sub: pd.DataFrame, available_labels: list) -> str:
    """Select the best label from available labels based on cluster characteristics."""
    # Calculate cluster characteristics
    avg_premium = df_sub['premium'].mean() if 'premium' in df_sub.columns else 0
    avg_population = df_sub['population'].mean() if 'population' in df_sub.columns else 0
    avg_market_share = df_sub['market_share'].mean() if 'market_share' in df_sub.columns else 0
    avg_claim_count = df_sub['unique_claim_count'].mean() if 'unique_claim_count' in df_sub.columns else 0
    avg_provider_count = df_sub['unique_provider_count'].mean() if 'unique_provider_count' in df_sub.columns else 0
    num_states = df_sub['payor_state'].nunique() if 'payor_state' in df_sub.columns else 1
    
    # Priority-based selection from available labels
    if "High Premium Segment" in available_labels and avg_premium > 800000000:
        return "High Premium Segment"
    elif "Large Network Providers" in available_labels and avg_provider_count > 30000:
        return "Large Network Providers"
    elif "High Utilization Segment" in available_labels and avg_claim_count > 300000:
        return "High Utilization Segment"
    elif "Market Leaders" in available_labels and avg_market_share > 0.20:
        return "Market Leaders"
    elif "Regional Specialists" in available_labels and avg_market_share > 0.12 and num_states < 10:
        return "Regional Specialists"
    elif "Cost Efficient Segment" in available_labels and avg_premium < 100000000:
        return "Cost Efficient Segment"
    elif "Comprehensive Insurers" in available_labels and avg_premium > 1500000000 and avg_population > 8000000:
        return "Comprehensive Insurers"
    elif "Complex Product Mix" in available_labels and avg_population > 3000000 and avg_claim_count > 150000:
        return "Complex Product Mix"
    else:
        # Return first available label as fallback
        return available_labels[0] if available_labels else "Comprehensive Insurers"

def _label(df_sub: pd.DataFrame, cid: int, centroids: dict = None) -> str:
    """Generate cluster label based on actual cluster characteristics for the specific year."""
    
    # Calculate cluster centroids for validation
    cluster_centroids = {}
    if centroids:
        for feature, feature_centroids in centroids.items():
            if cid in feature_centroids:
                cluster_centroids[feature] = feature_centroids[cid]
    
    # Validate feature dominance using centroids
    dominant_features = []
    if cluster_centroids:
        # Find which features are dominant for this cluster compared to others
        for feature, cluster_value in cluster_centroids.items():
            if feature in centroids:
                other_clusters = [centroids[feature][i] for i in range(3) if i != cid]
                if other_clusters and cluster_value > max(other_clusters):
                    dominant_features.append(feature)
    
    # DATA VALIDATION: Verify cluster characteristics match the label
    cluster_stats = _calculate_cluster_statistics(df_sub)

    # Fallback: analyze cluster characteristics to select most appropriate label
    prompt = (
        "Based on the cluster data below, select the **exactly one** most appropriate label from this list:\n"
        "['Premium Titans', 'Diverse Coverage', 'Network Titans', 'Comprehensive Insurers', 'High Utilizers', 'Complex Coverage', 'Market Leaders', 'Regional Players', 'Cost Efficient', 'Growth Focused']\n\n"
        "Rules:\n"
        "- Select the label that BEST describes the companies in this cluster\n"
        "- Consider premium levels, market coverage, network size, and business characteristics\n"
        "- Return ONLY the label name, nothing else\n\n"
        "PRIMARY ANALYSIS: Use the centroid data and column analysis to determine the cluster name.\n\n"
        "CLUSTER CENTROIDS (MOST IMPORTANT):\n"
        f"Cluster {cid} dominant features: {dominant_features}\n"
        f"Cluster {cid} centroids: {cluster_centroids}\n\n"
        "CLUSTER STATISTICS (VERIFICATION):\n"
        f"Cluster {cid} actual averages: {cluster_stats}\n\n"
        "Available features to analyze:\n"
        "- population: Total population served\n"
        "- unique_member_count: Number of unique members enrolled\n"
        "- premium: Total premium revenue\n"
        "- unique_provider_count: Number of unique healthcare providers in network\n"
        "- unique_claim_count: Number of unique claims processed\n"
        "- market_share: Market share percentage\n"
        "- payor_state: Geographic coverage (number of states)\n\n"
        "Instructions:\n"
        "1. FIRST: Analyze the dominant features from centroids - these define the cluster's primary characteristics\n"
        "2. SECOND: Verify the cluster statistics match the dominant features\n"
        "3. THIRD: Ensure the selected label is justified by BOTH centroid dominance AND actual data averages\n"
        "4. FOURTH: If data doesn't justify a label, select a more appropriate one\n\n"
        "Label selection based on DOMINANT CENTROID FEATURES (HIGH PRIORITY ORDER with MODERATE thresholds):\n"
        "PRIORITY 1 - High Premium Segment: When premium is dominant AND premium centroid > $3,000,000,000 (3 billion)\n"
        "PRIORITY 2 - Large Network Providers: When unique_provider_count is dominant AND unique_provider_count centroid > 30,000 providers\n"
        "PRIORITY 3 - High Utilization Segment: When unique_claim_count is dominant AND unique_claim_count centroid > 300,000 claims\n"
        "PRIORITY 4 - Market Leaders: When market_share is dominant AND market_share centroid > 20%\n"
        "PRIORITY 5 - Regional Specialists: When market_share is dominant AND market_share centroid > 12% BUT geographic_coverage centroid < 10 states\n"
        "PRIORITY 6 - Cost Efficient Segment: When premium is NOT dominant AND premium centroid < $100,000,000\n"
        "PRIORITY 7 - Comprehensive Insurers: When 3+ features are dominant AND premium centroid > $1,500,000,000 AND population centroid > 8,000,000 AND unique_member_count centroid > 1,500,000\n"
        "PRIORITY 8 - Complex Product Mix: When BOTH unique_member_count AND unique_claim_count are dominant AND unique_member_count centroid > 3,000,000 AND unique_claim_count centroid > 150,000\n\n"
        "AVAILABLE LABELS: ['High Premium Segment', 'Large Network Providers', 'Comprehensive Insurers', 'High Utilization Segment', 'Complex Product Mix', 'Market Leaders', 'Regional Specialists', 'Cost Efficient Segment']\n\n"
        "CRITICAL: The dominant features from centroids should be the PRIMARY factor in determining the cluster name.\n"
        "VERIFICATION: Ensure the selected label is justified by actual cluster data averages.\n"
        "IMPORTANT: Generate labels based on THIS YEAR'S data characteristics, not predefined patterns.\n"
        "Return ONLY the label name, nothing else.\n\n"
        f"Cluster data:\n{df_sub.to_markdown(index=False)}"
    )
    
    try:
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,  # Low temperature for consistent selection
            max_tokens=50
        )
        label = response.choices[0].message.content.strip()
        
        # Clean up the response and validate it's from our list
        label = label.replace('"', '').replace("'", "").strip()
        valid_labels = ['High Premium Segment', 'Large Network Providers', 'Comprehensive Insurers', 'High Utilization Segment', 'Complex Product Mix', 'Market Leaders', 'Regional Specialists', 'Cost Efficient Segment', 'Growth Oriented']
        
        if label in valid_labels:
            # VERIFY the label is justified by data
            if _verify_label_justification(label, cluster_stats, dominant_features):
                return label
            else:
                # If not justified, select a more appropriate label
                return _select_best_label(df_sub, valid_labels)
        else:
            # If response is not in our list, select based on cluster characteristics
            return _select_best_label(df_sub, valid_labels)
            
    except Exception as e:
        # Fallback to characteristic-based selection
        return _select_best_label(df_sub, ['High Premium Segment', 'Large Network Providers', 'Comprehensive Insurers', 'High Utilization Segment', 'Complex Product Mix', 'Market Leaders', 'Regional Specialists', 'Cost Efficient Segment', 'Growth Oriented'])

def _select_best_label(df_sub: pd.DataFrame, valid_labels: list) -> str:
    """Select the best label based on cluster characteristics."""
    # Calculate cluster characteristics
    avg_premium = df_sub['premium'].mean() if 'premium' in df_sub.columns else 0
    avg_population = df_sub['population'].mean() if 'population' in df_sub.columns else 0
    avg_market_share = df_sub['market_share'].mean() if 'market_share' in df_sub.columns else 0
    num_states = df_sub['payor_state'].nunique() if 'payor_state' in df_sub.columns else 1
    
    # Selection logic based on characteristics
    if avg_premium > 1000000000:  # High premium
        return "High Premium Segment"
    elif avg_population > 10000000:  # Large network
        return "Large Network Providers"
    elif avg_market_share > 0.1:  # High market share
        return "Market Leaders"
    elif avg_premium < 100000000:  # Low cost
        return "Cost Efficient Segment"
    elif num_states < 5:  # Regional focus
        return "Regional Specialists"
    else:
        return "Comprehensive Insurers"  # Default

@st.cache_data(show_spinner=False)
def _calculate_cluster_statistics(df_sub: pd.DataFrame) -> dict:
    """Calculate actual statistics for a cluster to verify label justification."""
    stats = {}
    
    # Calculate averages for key features
    numeric_features = ['premium', 'population', 'unique_member_count', 'unique_provider_count', 'unique_claim_count', 'market_share']
    
    for feature in numeric_features:
        if feature in df_sub.columns:
            stats[feature] = df_sub[feature].mean()
    
    # Calculate geographic coverage
    if 'payor_state' in df_sub.columns:
        stats['geographic_coverage'] = df_sub['payor_state'].nunique()
    
    return stats

def _verify_label_justification(label: str, cluster_stats: dict, dominant_features: list) -> bool:
    """Verify that the selected label is justified by the actual cluster data."""
    
    # Define verification thresholds for each label
    verification_thresholds = {
        'High Premium Segment': {
            'premium': 3000000000,  # $3B
            'dominant_features': ['premium']
        },
        'Large Network Providers': {
            'unique_provider_count': 30000,  # 30K providers
            'dominant_features': ['unique_provider_count']
        },
        'Market Leaders': {
            'market_share': 0.20,  # 20%
            'dominant_features': ['market_share']
        },
        'High Utilization Segment': {
            'unique_claim_count': 300000,  # 300K claims
            'dominant_features': ['unique_claim_count']
        },
        'Regional Specialists': {
            'market_share': 0.12,  # 12%
            'geographic_coverage': 10,  # < 10 states
            'dominant_features': ['market_share']
        },
        'Cost Efficient Segment': {
            'premium': 100000000,  # < $100M
            'dominant_features': []  # Premium should NOT be dominant
        },
        'Complex Product Mix': {
            'unique_member_count': 3000000,  # 3M members
            'unique_claim_count': 150000,  # 150K claims
            'dominant_features': ['unique_member_count', 'unique_claim_count']
        },
        'Comprehensive Insurers': {
            'premium': 1500000000,  # $1.5B
            'population': 8000000,  # 8M population
            'unique_member_count': 1500000,  # 1.5M members
            'dominant_features': []  # 3+ features should be dominant
        }
    }
    
    if label not in verification_thresholds:
        return False
    
    thresholds = verification_thresholds[label]
    
    # Check if dominant features match
    if 'dominant_features' in thresholds and thresholds['dominant_features']:
        if not all(feature in dominant_features for feature in thresholds['dominant_features']):
            return False
    
    # Check if actual values meet thresholds
    for feature, threshold in thresholds.items():
        if feature != 'dominant_features' and feature in cluster_stats:
            if feature == 'geographic_coverage':
                # For geographic coverage, check if it's LESS than threshold (Regional Specialists)
                if label == 'Regional Specialists':
                    if cluster_stats[feature] >= threshold:
                        return False
                else:
                    if cluster_stats[feature] < threshold:
                        return False
            else:
                if cluster_stats[feature] < threshold:
                    return False
    
    return True

@st.cache_data(show_spinner=False)
def _get_dominant_features_for_cluster(df: pd.DataFrame, cluster_id: int, centroids: dict) -> list:
    """Get the dominant features for a specific cluster based on centroids."""
    dominant_features = []
    
    if not centroids:
        return dominant_features
    
    # Check which features are dominant for this cluster compared to others
    for feature, feature_centroids in centroids.items():
        if cluster_id in feature_centroids:
            cluster_value = feature_centroids[cluster_id]
            other_clusters = [feature_centroids[i] for i in range(3) if i != cluster_id]
            
            if other_clusters and cluster_value > max(other_clusters):
                dominant_features.append(feature)
    
    return dominant_features

def _format_dominant_features_text(dominant_features: list, cluster_name: str) -> str:
    """Format dominant features and cluster name explanation as sentences."""
    if not dominant_features:
        return "This cluster shows balanced characteristics across all features.\nCluster name reflects overall performance patterns."
    
    # Create meaningful feature descriptions
    feature_descriptions = {
        'premium': 'premium revenue',
        'population': 'population size',
        'unique_member_count': 'member enrollment',
        'unique_provider_count': 'provider network size',
        'unique_claim_count': 'claim volume',
        'market_share': 'market share',
        'geographic_coverage': 'geographic coverage'
    }
    
    descriptions = [feature_descriptions.get(feature, feature) for feature in dominant_features]
    
    # First line: features sentence (consistent length)
    if len(descriptions) == 1:
        features_line = f"This cluster excels in {descriptions[0]}."
    elif len(descriptions) == 2:
        features_line = f"This cluster excels in {descriptions[0]} and {descriptions[1]}."
    else:
        features_line = f"This cluster excels in {', '.join(descriptions[:-1])} and {descriptions[-1]}."
    
    # Second line: cluster name explanation sentence (data-driven)
    if cluster_name == 'High Premium Segment':
        explanation_line = f"This segment reflects the highest {feature_descriptions.get('premium', 'premium revenue')} among all segments."
    elif cluster_name == 'Market Leaders':
        explanation_line = f"This segment reflects dominant {feature_descriptions.get('market_share', 'market share')} position."
    elif cluster_name == 'Large Network Providers':
        explanation_line = f"This segment reflects the largest {feature_descriptions.get('unique_provider_count', 'healthcare provider networks')}."
    elif cluster_name == 'High Utilization Segment':
        explanation_line = f"This segment reflects the highest {feature_descriptions.get('unique_claim_count', 'claim processing volumes')}."
    elif cluster_name == 'Regional Specialists':
        explanation_line = f"This segment reflects strong local {feature_descriptions.get('market_share', 'market concentration')}."
    elif cluster_name == 'Cost Efficient Segment':
        explanation_line = f"This segment reflects the most cost-effective {feature_descriptions.get('premium', 'premium structures')}."
    elif cluster_name == 'Complex Product Mix':
        explanation_line = f"This segment reflects diverse {feature_descriptions.get('unique_member_count', 'member')} and {feature_descriptions.get('unique_claim_count', 'claim')} portfolios."
    elif cluster_name == 'Comprehensive Insurers':
        explanation_line = f"This segment reflects balanced high performance across multiple {', '.join(descriptions)}."
    else:
        explanation_line = f"This segment reflects the dominant {', '.join(descriptions)} characteristics."
    
    return f"{features_line}\n{explanation_line}"

def _summary(df_sub: pd.DataFrame, cid: int) -> str:
    prompt = (
        f"Summarize Cluster {cid+1} behavior in ≤2 sentences (no bullets). "
        "Highlight medical cost vs administrative cost patterns.\n\n"
        f"{df_sub.to_markdown(index=False)}"
    )
    try:
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a helpful data analyst."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"⚠ GPT error: {e}"

# ──────────────────────────────────────────────────────────────
# Main Ranking Page
# ──────────────────────────────────────────────────────────────
def page(df_all: pd.DataFrame) -> None:
    st.header("Ranking")

    # Initialize selected payor in session state (moved to top)
    if "selected_payor_name" not in st.session_state:
        st.session_state.selected_payor_name = None

    # Always define selected_payor and next_payor at page level
    selected_payor = None
    next_payor = None

    # Apply ranking algorithm first
    df_ranked = data_ranking(df_all)
    if not df_ranked.empty:
        df_all = df_ranked  # Use ranked data for further processing

    # Step 1: Let user pick year and highlight payors
    years = sorted(df_all["year"].dropna().unique())
    if not years:
        st.error("No year data found in the uploaded file.")
        return
        
    # Get all unique payor names for highlighting (available immediately)
    all_payor_names = sorted(df_all['payor_name'].unique())
    
    # Create two columns for year selection and payor highlighting
    col1, col2 = st.columns(2)
    
    with col1:
        year = st.selectbox("Choose year", ["Choose from options"] + years, index=0)
    
    with col2:
        hi = st.multiselect("Highlight payors", all_payor_names)
        # Store highlighted payors in session state for deep-dive view
        st.session_state.highlighted_payors = hi
    
    # Check if year is selected before proceeding
    if year == "Choose from options":
        st.info("Please select a year to continue.")
        return
    
    # Step 2: Clean the data for the selected year
    try:
        df = _clean(df_all, year)
        
        if df.empty:
            st.error("No data remaining after cleaning for this year.")
            return
            
    except Exception as e:
        st.error(f"Error cleaning data: {e}")
        return

    # Step 3: Clustering
    dfc = _cluster(df)
    dfc = add_rankings(dfc)

    # Step 3: GPT cluster labels with centroid validation and uniqueness
    # Calculate centroids for validation (needed for both labeling and summaries)
    centroids = _calculate_cluster_centroids(dfc)
    
    if "cluster_labels" not in st.session_state:
        # Generate unique labels for each cluster with caching
        used_labels = set()
        cluster_labels = {}
        
        with st.spinner("Generating cluster labels..."):
            progress_bar = st.progress(0)
            for i in range(3):
                cluster_data = dfc[dfc["cluster"] == i]
                label = _label_with_uniqueness(cluster_data, i, centroids, used_labels)
                cluster_labels[i] = label
                used_labels.add(label)
                progress_bar.progress((i + 1) / 3)
            progress_bar.empty()
        
        st.session_state.cluster_labels = cluster_labels
    labels = st.session_state.cluster_labels
    dfc["cluster_label"] = dfc["cluster"].map(labels)

    # Initialize session state for deep-dive
    if "sel_cluster" not in st.session_state:
        st.session_state.sel_cluster = None
    if "beh_cache" not in st.session_state:
        st.session_state.beh_cache = {}

    # Deep-dive view or cluster landscape
    if st.session_state.sel_cluster is None:
        # Cluster Landscape View
        st.subheader("Cluster landscape")
        
        # Create anonymized dataframe for display (protect competitor information)
        dfc_display = dfc.copy()
        selected_payor_name_only = None
        if st.session_state.selected_payor_name:
            selected_payor_name_only = st.session_state.selected_payor_name.split(' - ')[0]  # Extract just the name part
        
        # Remove payor_name from display completely for privacy
        dfc_display = dfc_display.drop(columns=['payor_name'], errors='ignore')
        
        # Create payor names for dropdown and highlighting
        payor_names = dfc['payor_name'] + ' - ' + dfc['payor_state']
        
        # Use the highlighted payors from session state
        dim, up = dfc[~dfc['payor_name'].isin(hi)], dfc[dfc['payor_name'].isin(hi)]

        # Original cluster plot with dim/bright functionality
        fig = go.Figure()
        
        # Add dim points
        for i in range(3):
            sub = dim[dim["cluster"] == i]
            fig.add_trace(go.Scatter(
                x=sub["pca1"], y=sub["pca2"],
                mode="markers",
                marker=dict(size=6, opacity=0.6, color=PALETTE[i]),
                text=sub.apply(lambda r: (
                    f"State: {r['payor_state']}<br>"
                    f"Premium: ${r['premium']:,.0f}<br>"
                    f"Market Share: {r['market_share']:.2%}<br>"
                    f"Population: {r['population']:,.0f}<br>"
                    f"Final Rank: {r.get('rank', 'N/A')}"
                ), axis=1),
                hoverinfo="text", name=labels[i],
                customdata=sub.apply(lambda r: f"Payor #{r.get('rank', 'N/A')} - {r['payor_state']}", axis=1),
            ))

        # Add bright points (highlighted payors - show payor_name)
        for i in range(3):
            sub = up[up["cluster"] == i]
            if sub.empty:
                continue
            fig.add_trace(go.Scatter(
                x=sub["pca1"], y=sub["pca2"],
                mode="markers",
                marker=dict(
                    size=12, 
                    color=PALETTE[i], 
                    line=dict(width=2, color="white"),
                    opacity=1.0
                ),
                text=sub.apply(lambda r: (
                    f"Payor: {r['payor_name']}<br>"
                    f"State: {r['payor_state']}<br>"
                    f"Premium: ${r['premium']:,.0f}<br>"
                    f"Market Share: {r['market_share']:.2%}<br>"
                    f"Population: {r['population']:,.0f}<br>"
                    f"Final Rank: {r.get('rank', 'N/A')}"
                ), axis=1),
                hoverinfo="text", showlegend=False, name=labels[i],
                customdata=sub.apply(lambda r: f"{r['payor_name']} - {r['payor_state']}", axis=1),
            ))

        fig.update_layout(
            title="Cluster Landscape",
            xaxis_title=None, yaxis_title=None,
            xaxis_showticklabels=False, yaxis_showticklabels=False,
            legend_title="Clusters"
        )
        
        st.plotly_chart(fig, use_container_width=True)

        # Cluster summary cards with modern UI (no graphs, no icons)
        st.subheader("Cluster Summaries")
        
        # Create a more engaging layout with metrics and visual elements
        col1, col2, col3 = st.columns(3)
        
        # for i in range(3):
        #     sub = dfc_display[dfc_display["cluster"] == i]
            
        #     with col1 if i == 0 else col2 if i == 1 else col3:
        #         with st.container(border=True):
        #             # Header with color accent - same styling as provided code but with consistent colors
        #             st.markdown(f"""
        #             <div style="background: linear-gradient(90deg, {PALETTE[i]}20, transparent); 
        #                         padding: 1rem; border-radius: 8px; margin: -1rem -1rem 1rem -1rem;">
        #                 <h3 style="color: {PALETTE[i]}; margin: 0;">{labels[i]}</h3>
        #             </div>
        #             """, unsafe_allow_html=True)
                    
        #             # Simple payor count only
        #             st.markdown(f"<span style='color:#555;'>Payors: <b>{len(sub)}</b></span>", unsafe_allow_html=True)
                    
        #             # Action button with better styling
        #             if st.button("🔍 Explore Cluster", key=f"btn_{i}", use_container_width=True):
        #                 st.session_state.sel_cluster = i
        #                 st.rerun()

        for i in range(3):
            sub = dfc_display[dfc_display["cluster"] == i]
            cluster_colors = ["#EF553B", "#00CC96","#636EFA"]
            
            # Get dominant features for this cluster
            dominant_features = _get_dominant_features_for_cluster(dfc, i, centroids)
            dominant_text = _format_dominant_features_text(dominant_features, labels[i])
            
            with col1 if i == 0 else col2 if i == 1 else col3:
                with st.container(border=True):
                    st.markdown(f"""
                        <div style="
                            background: linear-gradient(90deg, {cluster_colors[i]}33, transparent);
                            padding: 0.9rem 1.2rem;
                            border-radius: 8px;
                            margin-bottom: 1rem;
                            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
                        ">
                            <h3 style="
                                color: {cluster_colors[i]};
                                margin: 0;
                                font-weight: 600;
                                font-size: 1.15rem;
                                letter-spacing: 0.3px;
                            ">{labels[i]}</h3>
                        </div>
                    """, unsafe_allow_html=True)

                    st.markdown(f"<span style='color:#555;'>Payors: <b>{len(sub)}</b></span>", unsafe_allow_html=True)
                    
                    # Dominant features information
                    st.markdown(f"<span style='color:#666; font-size: 0.9em;'>{dominant_text}</span>", unsafe_allow_html=True)

                    if st.button("🔍 Explore Cluster", key=f"btn_{i}", use_container_width=True):
                        st.session_state.sel_cluster = i
                        st.rerun()


    else:
        # Deep-dive view
        cid = st.session_state.sel_cluster
        sub = dfc[dfc["cluster"] == cid]
        label = labels[cid]
        
        if st.button("← Back"):
            st.session_state.sel_cluster = None
            st.rerun()
        
        st.header(f"Detailed view · {label}")

        # Create PCA biplot for this cluster
        available_num = [col for col in NUM if col in dfc.columns]
        available_bin = [col for col in BIN if col in dfc.columns]
        feature_cols = available_num + available_bin + ["payor_state"]
        preprocess = _prep()
        X = preprocess.fit_transform(dfc[feature_cols])
        feature_names = preprocess.get_feature_names_out()
        
        # Payor Comparison Section (only in deep-dive view)
        st.sidebar.title("Payor Comparison")
        st.sidebar.info("💡 **Select a payor below to compare with the higher ranked payor**")
        
        # Create payor selection dropdown - only show highlighted payors from this specific cluster
        highlighted_payors_in_cluster = dfc[
            (dfc['payor_name'].isin(st.session_state.get('highlighted_payors', []))) & 
            (dfc['cluster'] == cid)
        ]['payor_name'] + ' - ' + dfc[
            (dfc['payor_name'].isin(st.session_state.get('highlighted_payors', []))) & 
            (dfc['cluster'] == cid)
        ]['payor_state']
        
        selected_payor_name = st.sidebar.selectbox(
            "Select Payor to Compare", 
            ["-- Select a payor --"] + sorted(highlighted_payors_in_cluster.unique()),
            key="payor_selection",
            index=0
        )
        
        # Update session state if a payor is selected
        if selected_payor_name and selected_payor_name != "-- Select a payor --":
            st.session_state.selected_payor_name = selected_payor_name
        elif selected_payor_name == "-- Select a payor --":
            st.session_state.selected_payor_name = None
        
        # Sidebar for feature selection (moved after Payor Comparison)
        # st.sidebar.title("Feature Loadings")
        # features_to_show = st.sidebar.multiselect(
        #     "Select features to display on biplot:",
        #     options=[f for f in feature_names if f not in ["PC1", "PC2"]],
        #     default=[],
        #     placeholder="choose from the option"
        # )
        features_to_show = []  # Ensure downstream code still works
        
        # Identify payors to highlight
        payors_to_highlight = []
        if 'selected_payor_name' in st.session_state and st.session_state.selected_payor_name:
            selected_payor_series = dfc[dfc['payor_name'] + ' - ' + dfc['payor_state'] == st.session_state.selected_payor_name]
            if not selected_payor_series.empty:
                selected_payor = selected_payor_series.iloc[0]
                next_payor = find_higher_ranked_payor(dfc, st.session_state.selected_payor_name)
                payors_to_highlight.append(st.session_state.selected_payor_name)
                if next_payor is not None:
                    next_payor_identifier = next_payor['payor_name'] + ' - ' + next_payor['payor_state']
                    payors_to_highlight.append(next_payor_identifier)

        # Cluster plot for this specific cluster with payor comparison
        fig = go.Figure()
        
        # Add all points for this cluster
        cluster_data = dfc[dfc["cluster"] == cid]
        fig.add_trace(go.Scatter(
            x=cluster_data["pca1"], y=cluster_data["pca2"],
            mode="markers",
            marker=dict(size=8, color=PALETTE[cid], line=dict(width=1, color="black")),
            text=cluster_data.apply(lambda r: (
                f"State: {r['payor_state']}<br>"
                f"Premium: ${r['premium']:,.0f}<br>"
                f"Market Share: {r['market_share']:.2%}<br>"
                f"Population: {r['population']:,.0f}<br>"
                f"Final Rank: {r.get('rank', 'N/A')}"
            ), axis=1),
            hoverinfo="text", name=label,
            customdata=cluster_data.apply(lambda r: f"Payor #{r.get('rank', 'N/A')} - {r['payor_state']}", axis=1),
            showlegend=True
        ))

        # Add highlighted payors with different symbols
        if payors_to_highlight:
            for i, payor_id in enumerate(payors_to_highlight):
                payor_data = dfc[dfc['payor_name'] + ' - ' + dfc['payor_state'] == payor_id]
                if not payor_data.empty:
                    payor = payor_data.iloc[0]
                    
                    # Determine if this is the selected payor or better ranked payor
                    if payor_id == st.session_state.selected_payor_name:
                        # Selected payor - circle with white border, use cluster color
                        symbol = "circle"
                        color = PALETTE[cid]  # Use cluster color
                        name = "Selected Payor"
                        hover_text = f"Selected Payor<br>State: {payor['payor_state']}<br>Premium: ${payor['premium']:,.0f}<br>Market Share: {payor['market_share']:.2%}<br>Population: {payor['population']:,.0f}<br>Medical Cost: ${payor.get('Medical Cost (MC)', payor.get('MC', 0)):,.0f}<br>Administrative Cost: ${payor.get('Administrative Cost (AC)', payor.get('AC', 0)):,.0f}<br>Final Rank: {payor.get('rank', 'N/A')}"
                    else:
                        # Better ranked payor - star with white border
                        symbol = "star"
                        color = "rgba(255, 215, 0, 0.8)"  # Gold color
                        name = "Better Ranked Payor"
                        hover_text = f"Better Ranked Payor<br>State: {payor['payor_state']}<br>Premium: ${payor['premium']:,.0f}<br>Market Share: {payor['market_share']:.2%}<br>Population: {payor['population']:,.0f}<br>Medical Cost: ${payor.get('Medical Cost (MC)', payor.get('MC', 0)):,.0f}<br>Administrative Cost: ${payor.get('Administrative Cost (AC)', payor.get('AC', 0)):,.0f}<br>Final Rank: {payor.get('rank', 'N/A')}"
                    
                    fig.add_trace(go.Scatter(
                        x=[payor["pca1"]], y=[payor["pca2"]],
                        mode="markers",
                        marker=dict(
                            symbol=symbol,
                            color=color,
                            size=16,
                            line=dict(width=2, color="white")
                        ),
                        name=name,
                        hoverinfo="text",
                        text=[hover_text],
                        showlegend=True
                    ))

        fig.update_layout(
            title=f"{label} Cluster",
            xaxis_title=None, yaxis_title=None,
            xaxis_showticklabels=False, yaxis_showticklabels=False,
            legend_title="Clusters",
            legend=dict(
                itemsizing='constant',
                itemwidth=30
            )
        )
        
        # Make the plot clickable with proper event handling
        st.plotly_chart(fig, use_container_width=True, key="clickable_plot")

        # Payor Comparison Section
        if selected_payor is not None:
            st.markdown("---")
            st.markdown("### Payor Comparison")
            
            # Generate insights
            st.markdown("#### Generated Insights")
            with st.spinner("Generating comparison insights..."):
                insights = generate_insights(selected_payor, next_payor)
            
            # Display insights with beautiful formatting
            st.markdown("""
            <style>
            .insights-container {
                background: rgba(255, 255, 255, 0.05);
                border-radius: 12px;
                padding: 20px;
                margin: 15px 0;
                border-left: 4px solid #DC2626;
                backdrop-filter: blur(10px);
                border: 1px solid rgba(255, 255, 255, 0.1);
            }
            .insights-section {
                margin: 20px 0;
            }
            .insights-section h4 {
                color: #DC2626;
                font-size: 18px;
                font-weight: 600;
                margin-bottom: 12px;
                border-bottom: 2px solid #DC2626;
                padding-bottom: 5px;
            }
            .insights-bullet {
                background: rgba(255, 255, 255, 0.03);
                border-radius: 8px;
                padding: 12px 15px;
                margin: 8px 0;
                border-left: 3px solid #636EFA;
                border: 1px solid rgba(255, 255, 255, 0.1);
                font-size: 14px;
                line-height: 1.5;
                color: #E5E7EB;
            }
            .insights-bullet:hover {
                transform: translateX(3px);
                transition: transform 0.2s ease;
                background: rgba(255, 255, 255, 0.08);
                border-color: rgba(255, 255, 255, 0.2);
            }
            </style>
            """, unsafe_allow_html=True)
            
            # Parse and format the insights
            if insights:
                # Split insights into sections
                sections = insights.split('**')
                formatted_insights = '<div class="insights-container">'
                
                for i, section in enumerate(sections):
                    if section.strip():
                        if section.startswith('Key Differences:') or section.startswith('Strategic Analysis:') or section.startswith('Recommendations for'):
                            # This is a section header
                            section_name = section.replace(':', '').replace('**', '')
                            formatted_insights += f'<div class="insights-section"><h4>{section_name}</h4>'
                        elif '•' in section:
                            # This contains bullet points
                            bullets = section.split('•')
                            for bullet in bullets:
                                if bullet.strip():
                                    formatted_insights += f'<div class="insights-bullet">• {bullet.strip()}</div>'
                            formatted_insights += '</div>'
                
                formatted_insights += '</div>'
                st.markdown(formatted_insights, unsafe_allow_html=True)
            else:
                st.info("No insights generated. Please try again.")

        # Side-by-side payor comparison cards (only in deep-dive view)
        # Get original data for both payors (before clustering)
        selected_original = None
        next_original = None
        if selected_payor is not None:
            selected_original = df_all[(df_all['payor_name'] == selected_payor['payor_name']) & 
                                    (df_all['payor_state'] == selected_payor['payor_state']) &
                                    (df_all['year'] == year)].iloc[0] if len(df_all[(df_all['payor_name'] == selected_payor['payor_name']) & 
                                                                                  (df_all['payor_state'] == selected_payor['payor_state']) & 
                                                                                  (df_all['year'] == year)]) > 0 else None
        if next_payor is not None:
            next_original = df_all[(df_all['payor_name'] == next_payor['payor_name']) & 
                                 (df_all['payor_state'] == next_payor['payor_state']) &
                                 (df_all['year'] == year)].iloc[0] if len(df_all[(df_all['payor_name'] == next_payor['payor_name']) & 
                                                                               (df_all['payor_state'] == next_payor['payor_state']) & 
                                                                               (df_all['year'] == year)]) > 0 else None
        
        # Use Streamlit columns for proper side-by-side layout
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Selected Payor")
            if selected_payor is not None and selected_original is not None:
                with st.container():
                    st.markdown(f"**{selected_payor['payor_name']}**")
                    st.markdown(f"**State:** {selected_payor['payor_state']}")
                    # Ensure rank is displayed as integer if possible
                    rank_val = selected_payor.get('rank', 'N/A')
                    if isinstance(rank_val, float) and rank_val.is_integer():
                        rank_val = int(rank_val)
                    st.markdown(f"**Rank:** {rank_val}")
                    
                    # Display key features only
                    if 'premium' in selected_original:
                        st.markdown(f"**Premium:** ${selected_original['premium']:,.0f}")
                    if 'population' in selected_original:
                        st.markdown(f"**Population:** {selected_original['population']:,.0f}")
                    if 'market_share' in selected_original:
                        st.markdown(f"**Market Share:** {selected_original['market_share']:.2%}")
                    if 'medical_cost' in selected_original or 'MC' in selected_original:
                        medical_cost = selected_original.get('medical_cost', selected_original.get('MC', 0))
                        st.markdown(f"**Medical Cost:** ${medical_cost:,.0f}")
                    if 'administrative_cost' in selected_original or 'AC' in selected_original:
                        admin_cost = selected_original.get('administrative_cost', selected_original.get('AC', 0))
                        st.markdown(f"**Administrative Cost:** ${admin_cost:,.0f}")
            else:
                st.info("No payor selected for comparison")
        
        with col2:
            st.markdown("### Better Ranked Payor")
            if next_payor is not None and next_original is not None:
                with st.container():
                    # Check if payors belong to the same group
                    selected_group = selected_payor['payor_name'].split()[0] if selected_payor['payor_name'] else ""
                    next_group = next_payor['payor_name'].split()[0] if next_payor['payor_name'] else ""
                    
                    # Show "ANONYMOUS GRP" if they don't belong to the same group
                    if selected_group != next_group:
                        display_name = "ANONYMOUS GRP"
                    else:
                        display_name = next_payor['payor_name']
                    
                    st.markdown(f"**{display_name}**")
                    st.markdown(f"**State:** {next_payor['payor_state']}")
                    # Ensure rank is displayed as integer if possible
                    rank_val = next_payor.get('rank', 'N/A')
                    if isinstance(rank_val, float) and rank_val.is_integer():
                        rank_val = int(rank_val)
                    st.markdown(f"**Rank:** {rank_val}")
                    
                    # Display key features only
                    if 'premium' in next_original:
                        st.markdown(f"**Premium:** ${next_original['premium']:,.0f}")
                    if 'population' in next_original:
                        st.markdown(f"**Population:** {next_original['population']:,.0f}")
                    if 'market_share' in next_original:
                        st.markdown(f"**Market Share:** {next_original['market_share']:.2%}")
                    if 'medical_cost' in next_original or 'MC' in next_original:
                        medical_cost = next_original.get('medical_cost', next_original.get('MC', 0))
                        st.markdown(f"**Medical Cost:** ${medical_cost:,.0f}")
                    if 'administrative_cost' in next_original or 'AC' in next_original:
                        admin_cost = next_original.get('administrative_cost', next_original.get('AC', 0))
                        st.markdown(f"**Administrative Cost:** ${admin_cost:,.0f}")
            else:
                st.info("No better ranked payor available in this cluster")
        
        # Add spacing before radar chart
        st.markdown("")
        st.markdown("")
        st.markdown("")
        st.markdown("")
        
        # Radar chart for comparison (only if both payors exist)
        if selected_payor is not None and next_payor is not None:
            # Dynamic feature selection for radar chart
            st.sidebar.markdown("---")
            st.sidebar.markdown("### Radar Chart Features")
            
            # Get all numeric columns from the original Excel data
            all_numeric_cols = df_all.select_dtypes(include=[np.number]).columns.tolist()
            # Remove ranking fields, payor code, and year
            all_numeric_cols = [col for col in all_numeric_cols if col not in ["rank", "rank_score", "final_rank", "medical_cost_rank", "administrative_cost_rank", "MC_rank", "AC_rank", "RANK_SCORE", "FINAL_RANK", "MEDICAL_COST_RANK", "ADMINISTRATIVE_COST_RANK", "payor_code", "year"]]
            
            # Default features if none selected
            default_features = ["premium", "population", "market_share", "medical_cost", "administrative_cost"]
            default_features = [f for f in default_features if f in all_numeric_cols]
            
            selected_radar_features = st.sidebar.multiselect(
                "Select features for radar chart:",
                options=all_numeric_cols,
                default=default_features[:5] if default_features else all_numeric_cols[:5],
                help="Choose which features to display in the radar chart comparison"
            )
            
            if selected_radar_features:
                # Get values for selected features
                values1 = []
                values2 = []
                categories = []
                
                for feature in selected_radar_features:
                    val1 = selected_original.get(feature, 0) if selected_original is not None else 0
                    val2 = next_original.get(feature, 0) if next_original is not None else 0
                    # Handle NaN values
                    if pd.isna(val1):
                        val1 = 0
                    if pd.isna(val2):
                        val2 = 0
                    values1.append(float(val1))
                    values2.append(float(val2))
                    categories.append(feature.replace('_', ' ').title())
                
                if values1 and values2:
                    # Alternative normalization approaches
                    # normalization_method = st.sidebar.selectbox(
                    #     "Normalization Method:",
                    #     ["Percentile Rank", "Z-Score", "Min-Max", "Log Scale", "Relative to Max"],
                    #     index=0,
                    #     help="Choose how to normalize the radar chart values"
                    # )
                    
                    # if normalization_method == "Percentile Rank":
                    #     # Use percentile ranking within the cluster only
                    #     cluster_data = dfc[dfc["cluster"] == cid][selected_radar_features].dropna()
                    #     normalized_values1 = []
                    #     normalized_values2 = []
                    #     for i, feature in enumerate(selected_radar_features):
                    #         feature_data = cluster_data[feature].dropna()
                    #         if len(feature_data) > 0:
                    #             val1_percentile = (feature_data < values1[i]).mean() * 100
                    #             val2_percentile = (feature_data < values2[i]).mean() * 100
                    #             normalized_values1.append(val1_percentile / 100)  # Convert to 0-1
                    #             normalized_values2.append(val2_percentile / 100)
                    #         else:
                    #             normalized_values1.append(0.5)
                    #             normalized_values2.append(0.5)
                    
                    # elif normalization_method == "Z-Score":
                    #     # Standardize using z-scores within the cluster only
                    #     cluster_data = dfc[dfc["cluster"] == cid][selected_radar_features].dropna()
                    #     normalized_values1 = []
                    #     normalized_values2 = []
                    #     for i, feature in enumerate(selected_radar_features):
                    #         feature_data = cluster_data[feature].dropna()
                    #         if len(feature_data) > 1:
                    #             mean_val = feature_data.mean()
                    #             std_val = feature_data.std()
                    #             if std_val > 0:
                    #                 z1 = (values1[i] - mean_val) / std_val
                    #                 z2 = (values2[i] - mean_val) / std_val
                    #                 # Convert z-scores to 0-1 range (assuming normal distribution)
                    #                 normalized_values1.append(max(0, min(1, (z1 + 3) / 6)))
                    #                 normalized_values2.append(max(0, min(1, (z2 + 3) / 6)))
                    #             else:
                    #                 normalized_values1.append(0.5)
                    #                 normalized_values2.append(0.5)
                    #         else:
                    #             normalized_values1.append(0.5)
                    #             normalized_values2.append(0.5)
                    
                    # elif normalization_method == "Log Scale":
                    #     # Apply log transformation then min-max scaling within cluster
                    #     cluster_data = dfc[dfc["cluster"] == cid][selected_radar_features].dropna()
                    #     normalized_values1 = []
                    #     normalized_values2 = []
                    #     for i, feature in enumerate(selected_radar_features):
                    #         feature_data = cluster_data[feature].dropna()
                    #         if len(feature_data) > 0:
                    #             # Add small constant to avoid log(0)
                    #             epsilon = 1e-8
                    #             log_values1 = np.log(values1[i] + epsilon)
                    #             log_values2 = np.log(values2[i] + epsilon)
                    #             log_feature_data = np.log(feature_data + epsilon)
                    #             max_log = log_feature_data.max()
                    #             min_log = log_feature_data.min()
                                
                    #             if max_log == min_log:
                    #                 normalized_values1.append(0.5)
                    #                 normalized_values2.append(0.5)
                    #             else:
                    #                 normalized_values1.append((log_values1 - min_log) / (max_log - min_log))
                    #                 normalized_values2.append((log_values2 - min_log) / (max_log - min_log))
                    #         else:
                    #             normalized_values1.append(0.5)
                    #             normalized_values2.append(0.5)
                    
                    # elif normalization_method == "Relative to Max":
                    #     # Scale relative to the maximum value within the cluster only
                    #     cluster_data = dfc[dfc["cluster"] == cid][selected_radar_features].dropna()
                    #     normalized_values1 = []
                    #     normalized_values2 = []
                    #     for i, feature in enumerate(selected_radar_features):
                    #         feature_data = cluster_data[feature].dropna()
                    #         max_val = feature_data.max() if len(feature_data) > 0 else 1
                    #         if max_val > 0:
                    #             normalized_values1.append(min(1, values1[i] / max_val))
                    #             normalized_values2.append(min(1, values2[i] / max_val))
                    #         else:
                    #             normalized_values1.append(0.5)
                    #             normalized_values2.append(0.5)
                    
                    # else:  # Min-Max (original method)
                    #     all_values = values1 + values2
                    #     max_val = max(all_values) if all_values else 1
                    #     min_val = min(all_values) if all_values else 0
                        
                    #     if max_val == min_val:
                    #         normalized_values1 = [0.5] * len(values1)
                    #         normalized_values2 = [0.5] * len(values2)
                    #     else:
                    #         normalized_values1 = [(val - min_val) / (max_val - min_val) for val in values1]
                    #         normalized_values2 = [(val - min_val) / (max_val - min_val) for val in values2]

                    # Always use log transform with min-max normalization
                    cluster_data = dfc[dfc["cluster"] == cid][selected_radar_features].dropna()
                    normalized_values1 = []
                    normalized_values2 = []

                    for i, feature in enumerate(selected_radar_features):
                        feature_data = cluster_data[feature].dropna()
                        if len(feature_data) > 0:
                            epsilon = 1e-8
                            log_values1 = np.log(values1[i] + epsilon)
                            log_values2 = np.log(values2[i] + epsilon)
                            log_feature_data = np.log(feature_data + epsilon)
                            max_log = log_feature_data.max()
                            min_log = log_feature_data.min()

                            if max_log == min_log:
                                normalized_values1.append(0.5)
                                normalized_values2.append(0.5)
                            else:
                                normalized_values1.append((log_values1 - min_log) / (max_log - min_log))
                                normalized_values2.append((log_values2 - min_log) / (max_log - min_log))
                        else:
                            normalized_values1.append(0.5)
                            normalized_values2.append(0.5)

                                        
                    fig = go.Figure()
                    # Add customdata for original values and set hovertemplate
                    fig.add_trace(go.Scatterpolar(
                        r=normalized_values1,
                        theta=categories,
                        fill='toself',
                        name='Selected',
                        line_color="#636EFA",
                        customdata=np.array(values1).reshape(-1, 1),
                        hovertemplate='<b>%{theta}</b><br>Selected: %{customdata[0]:,.2f}<extra></extra>'
                    ))
                    fig.add_trace(go.Scatterpolar(
                        r=normalized_values2,
                        theta=categories,
                        fill='toself',
                        name='Better Ranked',
                        line_color="#EF553B",
                        customdata=np.array(values2).reshape(-1, 1),
                        hovertemplate='<b>%{theta}</b><br>Better Ranked: %{customdata[0]:,.2f}<extra></extra>'
                    ))
                    fig.update_layout(
                        polar=dict(
                            radialaxis=dict(
                                visible=True,
                                showgrid=False,  # Hide radial grid lines
                                showticklabels=False,  # Hide tick labels (0, 0.2, 0.4, etc.)
                                showline=False  # Hide the axis line
                            ),
                            angularaxis=dict(
                                tickfont=dict(size=12),  # Reduce font size for category labels
                            ),
                        ),
                        showlegend=True,
                        height=350,
                        margin=dict(l=0, r=0, t=30, b=80),  # Increase bottom margin
                        title=dict(
                            text="Feature Comparison Radar Chart",
                            font=dict(size=18, color="#E5E7EB")
                        )
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No valid features selected for radar chart")
            else:
                st.info("Please select features for the radar chart comparison")







    # Custom CSS for modern UI
    st.markdown("""
    <style>
    /* Modern button styling */
    .stButton > button {
        background: #6B7280;
        color: white;
        border: none;
        border-radius: 8px;
        padding: 8px 20px;
        font-weight: 500;
        transition: all 0.2s ease;
    }
    .stButton > button:hover {
        background: #4B5563;
        color: white;
        transform: translateY(-1px);
    }
    
    /* Plotly chart styling */
    .js-plotly-plot .plotly .main-svg {
        border-radius: 8px;
    }
    
    /* Container styling */
    .stContainer {
        border-radius: 8px;
    }
    
    /* Metric styling */
    .metric-container {
        background: #F9FAFB;
        border-radius: 8px;
        padding: 1rem;
        border: 1px solid #E5E7EB;
    }
    
    /* Comparison cards */
    .comparison-card {
        background: #F9FAFB;
        border-radius: 8px;
        padding: 1rem;
        border: 1px solid #E5E7EB;
        margin: 0.5rem 0;
    }
    </style>
    """, unsafe_allow_html=True)

def show_custom_loader():
    st.markdown(
        '''
        <div class="loader-container">
            <div class="loader"></div>
            <div style="margin-top:10px;color:#636EFA;font-weight:600;">Processing, please wait...</div>
        </div>
        <style>
        .loader-container { display: flex; flex-direction: column; align-items: center; margin: 2rem 0; }
        .loader {
            border: 6px solid #f3f3f3;
            border-top: 6px solid #636EFA;
            border-radius: 50%;
            width: 48px;
            height: 48px;
            animation: spin 1s linear infinite;
        }
        @keyframes spin {
            0% { transform: rotate(0deg);}
            100% { transform: rotate(360deg);}
        }
        </style>
        ''',
        unsafe_allow_html=True
    )