import streamlit as st

st.title("Home")

st.subheader("Aplikasi Klasifikasi Tingkat Kematangan Sangrai Biji Kopi")

st.markdown("""
    Aplikasi ini membantu mengklasifikasikan biji kopi menggunakan model deep learning Convolutional Neural Network (CNN) dan Graph Neural Network (GNN). Aplikasi ini dirancang untuk memberikan hasil yang cepat dan akurat, memungkinkan pengguna untuk melakukan klasifikasi biji kopi secara mandiri. Dengan antarmuka yang ramah pengguna dan aksesibilitas melalui platform web bisa memudahkan pengguna untuk mengunggah foto biji kopi dan mendapatkan hasil prediksi dalam waktu singkat.

    **Jenis Tingkat Kematangan Sangrai Biji Kopi**
    - Green
    - Light
    - Medium
    - Dark
""")

st.image("image/jenis_kopi.jpeg", caption="Jenis Tingkat Kematangan Sangrai Biji Kopi", use_container_width=True)

st.markdown("""
    **Tujuan Aplikasi**
    
    Aplikasi ini bertujuan untuk membantu pengguna mengklasifikasikan biji kopi yang disangrai secara mandiri. Diharapkan dapat membantu pengguna untuk mempelajari penyangraian biji kopi.
    
    **Keunggulan Aplikasi**
            
    - Menggunakan 2 Model Machine Learning Terbaik (CNN dan GNN) 
            
    - Aksesibilitas Melalui Web 
    
    Aplikasi ini dirancang untuk diakses dengan mudah melalui platform web menggunakan Streamlit, sehingga pengguna dapat mengunggah foto biji kopi dan mendapatkan hasil prediksi dengan cepat.   
""")

# Footer
def footer():
    st.markdown("""
        <style>
        .footer {
            position: fixed;
            left: 0;
            bottom: 0;
            width: 100%;
            color: #F7F1F0;
            text-align: right;
            padding: 5px;
            font-size: small;
        }
        </style>
        <div class='footer'>© 2026 Elang Al Majid - 210002</div>
    """, unsafe_allow_html=True)

# Display footer
footer()