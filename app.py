import streamlit as st

# Setup Page
home_page = st.Page(
    "home.py",
    title="Home",
    icon=":material/home:",
    default=True,
)
model_1_page = st.Page(
    "test_cnn.py",
    title="Test using CNN",
    icon=":material/image_search:",
)
model_2_page = st.Page(
    "test_gnn.py",
    title="Test using GNN",
    icon=":material/image_search:",
)
info_cnn = st.Page(
    "about_cnn.py",
    title="About Model CNN",
    icon=":material/info:",
)
info_gnn = st.Page(
    "about_gnn.py",
    title="About Model GNN",
    icon=":material/info:",
)

# Navigasi Pages
pg = st.navigation(
    {
        "Home Page": [home_page],
        "Model Using": [model_1_page, model_2_page],
        "Information": [info_cnn, info_gnn],
    }
)

# Run Page
pg.run()