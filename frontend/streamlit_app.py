import pandas as pd
import streamlit as st
import plotly.express as px
from collections import defaultdict

from menu import menu_with_redirect
from prepare_quran_dataset.construct.database import MoshafPool


def get_total_moshf_hours(moshaf_pool: MoshafPool) -> float:
    hours = 0.0
    for moshaf in moshaf_pool:
        hours += moshaf.total_duraion_minutes / 60.0
    return round(hours, 2)


def get_total_size_gb(moshaf_pool: MoshafPool) -> float:
    total_size = 0.0
    for moshaf in moshaf_pool:
        total_size += moshaf.total_size_mb / 1024.0
    return round(total_size, 2)


def get_reciter_to_hours(moshaf_pool: MoshafPool) -> dict[str, float]:
    """Returns a dict for {"reciter": "total hours"}
    """
    reciter_to_hours = defaultdict(lambda: 0.0)
    for moshaf in moshaf_pool:
        key = f'{moshaf.reciter_id} / {moshaf.reciter_arabic_name}'
        reciter_to_hours[key] += moshaf.total_duraion_minutes / 60.0
    for k in reciter_to_hours:
        reciter_to_hours[k] = round(reciter_to_hours[k], 2)
    return reciter_to_hours


def dashboard():
    # Sample data for display (replace with actual values)
    number_of_moshaf = len(st.session_state.moshaf_pool)
    number_of_reciters = len(st.session_state.reciter_pool)
    total_hours = get_total_moshf_hours(st.session_state.moshaf_pool)
    total_size_gb = get_total_size_gb(st.session_state.moshaf_pool)

    # Custom CSS for number styling with different title colors
    st.markdown("""
    <style>
    .stat-box {
        font-size: 24px;
        font-weight: bold;
        color: #ffffff;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
    }
    .title-moshaf {
        color: #FF5733;  /* Orange */
    }
    .title-reciters {
        color: #33C3FF;  /* Light Blue */
    }
    .title-hours {
        color: #9D33FF;  /* Purple */
    }
    .title-size {
        color: #33FF8A;  /* Green */
    }
    </style>
    """, unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    col3, col4 = st.columns(2)
    # Display statistics with styling
    with col1:
        st.markdown(
            f"<div class='stat-box'><span class='title-moshaf'>Number of Moshaf</span><br>{number_of_moshaf}</div>", unsafe_allow_html=True)

    with col2:
        st.markdown(
            f"<div class='stat-box'><span class='title-reciters'>Number of Reciters</span><br>{number_of_reciters}</div>", unsafe_allow_html=True)

    with col3:
        st.markdown(
            f"<div class='stat-box'><span class='title-hours'>Total Hours</span><br>{total_hours}</div>", unsafe_allow_html=True)

    with col4:
        st.markdown(
            f"<div class='stat-box'><span class='title-size'>Total Size (GB)</span><br>{total_size_gb}</div>", unsafe_allow_html=True)

    # Create an interactive Plotly bar chart for reciters and total hours
    reciters_data = get_reciter_to_hours(st.session_state.moshaf_pool)
    reciters_data = {
        'Reciter': list(reciters_data.keys()),
        'Total Hours': list(reciters_data.values())
    }
    df = pd.DataFrame(reciters_data)
    st.subheader("Reciters and Total Hours")
    fig = px.bar(df, x='Reciter', y='Total Hours', title="Total Hours by Reciter",
                 hover_data=['Total Hours'], color='Total Hours',
                 labels={'Total Hours': 'Total Hours'}, width=800)

    # Display the chart in Streamlit
    st.plotly_chart(fig)


def main():
    menu_with_redirect()
    st.title('Recitation Database Manager')
    dashboard()


if __name__ == '__main__':
    main()
