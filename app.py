import streamlit as st
from src import upload, ranking, trends

# Set wide layout and page title
st.set_page_config(layout="wide", page_title="Payor Benchmarking")

# Add banner CSS to sidebar
st.sidebar.markdown("""
<style>
/* Banner styling for sidebar */
.sidebar-banner {
    background: linear-gradient(135deg, #4A148C 0%, #6A1B9A 100%);
    border-radius: 10px;
    padding: 1rem;
    margin-bottom: 1rem;
    box-shadow: 0 5px 15px rgba(74, 20, 140, 0.3);
    color: white;
    text-align: center;
}

.sidebar-logo {
    font-size: 1.5rem;
    font-weight: bold;
    margin-bottom: 0.3rem;
    color: #E1BEE7;
}

.sidebar-tagline {
    font-size: 0.8rem;
    color: #F8BBD9;
}

.pattern-dots {
    display: flex;
    justify-content: center;
    gap: 3px;
    margin-top: 0.5rem;
}

.pattern-dot {
    width: 4px;
    height: 4px;
    border-radius: 50%;
    background: white;
    animation: pulse 2s infinite;
}

.pattern-dot:nth-child(odd) {
    background: #E1BEE7;
}

.pattern-dot:nth-child(even) {
    background: #F8BBD9;
}

@keyframes pulse {
    0% { transform: scale(1); }
    50% { transform: scale(1.2); }
    100% { transform: scale(1); }
}

/* Consistent button styling for sidebar (for Upload button) */
.sidebar .stButton > button {
    background: #6B7280 !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 8px 20px !important;
    font-weight: 500 !important;
    transition: all 0.2s ease !important;
}

.sidebar .stButton > button:hover {
    background: #4B5563 !important;
    color: white !important;
    transform: translateY(-1px) !important;
}

/* === NEW CSS TO MAKE NAVIGATION BUTTONS TRANSPARENT === */
/* Targets the 2nd and 3rd instances of stButton in the sidebar */
.sidebar .stButton:nth-of-type(2) > button,
.sidebar .stButton:nth-of-type(3) > button {
    background: transparent !important;
    color: inherit !important; /* Use sidebar's default text color */
    text-align: left;
    font-weight: normal !important;
    padding: 0.75rem 0.5rem !important;
}

.sidebar .stButton:nth-of-type(2) > button:hover,
.sidebar .stButton:nth-of-type(3) > button:hover {
    background: rgba(255, 255, 255, 0.1) !important; /* Subtle hover */
    transform: none !important; /* Disable lift effect on hover */
}

/* === ACTIVE PAGE HIGHLIGHTING === */
/* Highlight active page with red border and text */
.sidebar .stButton:nth-of-type(2) > button.active-page,
.sidebar .stButton:nth-of-type(3) > button.active-page {
    background: rgba(220, 38, 38, 0.1) !important; /* Light red background */
    color: #DC2626 !important; /* Red text */
    border-left: 3px solid #DC2626 !important; /* Red left border */
    font-weight: 600 !important;
}

.sidebar .stButton:nth-of-type(2) > button.active-page:hover,
.sidebar .stButton:nth-of-type(3) > button.active-page:hover {
    background: rgba(220, 38, 38, 0.15) !important; /* Slightly darker red on hover */
    color: #DC2626 !important;
}


/* === END OF ACTIVE PAGE HIGHLIGHTING === */

/* Ensure selectbox elements maintain standard size */
.stSelectbox > div > div {
    font-size: 1rem !important;
}

.stSelectbox label {
    font-size: 1rem !important;
    font-weight: 600 !important;
}
</style>
""", unsafe_allow_html=True)

# Add COTIVITI logo image to sidebar
try:
    st.sidebar.image(r"images_cotiviti.jpeg", use_container_width=True)
except:
    # Fallback to text banner if image not found
    st.sidebar.markdown("""
    <div class="sidebar-banner">
        <div class="sidebar-logo">COTIVITI</div>
        <div class="sidebar-tagline">The Power of Perspective</div>
        <div class="pattern-dots">
            <div class="pattern-dot"></div>
            <div class="pattern-dot"></div>
            <div class="pattern-dot"></div>
            <div class="pattern-dot"></div>
            <div class="pattern-dot"></div>
            <div class="pattern-dot"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Sidebar navigation
st.sidebar.title("Navigation")

st.markdown("<h1 style='text-align: center;'>Payor Benchmarking <span style='font-size: 0.5em; vertical-align: middle;'>v0.01</span></h1>", unsafe_allow_html=True)

# Show upload page if no data has been uploaded yet
if "df_raw" not in st.session_state:
    upload.page()
else:
    # --- Sidebar ---
    if "page" not in st.session_state:
        st.session_state.page = "home" # A default state

    if st.sidebar.button("📁 Upload New Data", use_container_width=True):
        keys_to_clear = ["df_raw", "page", "df_all", "cluster_labels", "sel_cluster", "beh_cache"]
        for key in keys_to_clear:
            st.session_state.pop(key, None)
        st.rerun()

    st.sidebar.markdown("---")

    # Determine which page is currently active
    current_page = st.session_state.get("page", "home")
    
    # Create navigation buttons with active highlighting
    if current_page == "ranking":
        st.sidebar.markdown("""
        <div style="background: rgba(220, 38, 38, 0.1); color: #DC2626; border-left: 3px solid #DC2626; font-weight: 600; padding: 0.75rem 0.5rem; border-radius: 8px; margin: 0.5rem 0; text-align: center;">
            Ranking
        </div>
        """, unsafe_allow_html=True)
    else:
        if st.sidebar.button("Ranking", use_container_width=True):
            st.session_state.page = "ranking"
            st.rerun()
    
    if current_page == "trends":
        st.sidebar.markdown("""
        <div style="background: rgba(220, 38, 38, 0.1); color: #DC2626; border-left: 3px solid #DC2626; font-weight: 600; padding: 0.75rem 0.5rem; border-radius: 8px; margin: 0.5rem 0; text-align: center;">
            Trends
        </div>
        """, unsafe_allow_html=True)
    else:
        if st.sidebar.button("Trends", use_container_width=True):
            st.session_state.page = "trends"
            st.rerun()

    st.sidebar.markdown("---")

    # --- Main Page Content ---
    if st.session_state.page == "ranking":
        ranking.page(st.session_state.df_raw)
    elif st.session_state.page == "trends":
        trends.page(st.session_state.df_raw)
    elif st.session_state.page == "home":
        st.info("Select a section from the sidebar to begin.")