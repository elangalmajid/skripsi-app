import streamlit as st


# Setup Page
home_page = st.Page(
    "home.py",
    title="Home",
    icon=":material/home:",
    default=True,
)
model_1_page = st.Page(
    "test_resnet50.py",
    title="Test using ResNet50",
    icon=":material/mystery:",
)
model_2_page = st.Page(
    "test_vgg16.py",
    title="Test using VGG16",
    icon=":material/symptoms:",
)
model_3_page = st.Page(
    "test_gnn.py",
    title="Test using VGG16",
    icon=":material/symptoms:",
)
info_resnet50 = st.Page(
    "about_resnet50.py",
    title="About Model Resnet50",
    icon=":material/info:",
)
info_vgg16 = st.Page(
    "about_vgg16.py",
    title="About Model VGG16",
    icon=":material/info:",
)

# Navigasi Pages
pg = st.navigation(
    {
        "Home Page": [home_page],
        "Model Using": [model_1_page, model_2_page, model_3_page],
        "Information": [info_resnet50, info_vgg16],
    }
)

# Logo EyeCheck.AI
#st.logo("eye_check.png")

# Run Page
pg.run()
