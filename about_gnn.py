import streamlit as st

st.title("Informasi Tentang Model GNN")

st.markdown("""
    Model Graph Neural Networks (GNN) menggunakan arsitektur **MobileViG**. 
    Berikut adalah penjelasan mendalam mengenai model dan arsitektur.
""")


st.subheader("Arsitektur MobileViG")

st.markdown("""
    **MobileViG** atau Mobile Vision GNN, merupakan model Graph Neural Network yang dirancang khusus untuk efisiensi pada perangkat mobile 
    dengan tetap mempertahankan performa tinggi. Arsitektur ini menggunakan pendekatan graph-based untuk memproses citra dengan merepresentasikan 
    gambar sebagai graph, di mana setiap node merepresentasikan patch dari gambar dan edge merepresentasikan relasi spasial antar patch. 
    Model ini menggabungkan efisiensi MobileNet dengan kekuatan GNN dalam menangkap relasi struktural, sehingga cocok untuk aplikasi computer vision 
    pada perangkat dengan resource terbatas.
""")

st.markdown("""
    **Kelebihan MobileViG**
    - Efisiensi Komputasi Tinggi, MobileViG dirancang untuk mobile deployment dengan jumlah parameter yang lebih sedikit dan operasi yang lebih efisien, 
    sehingga dapat berjalan dengan baik pada perangkat dengan resource terbatas.
    - Menangkap Relasi Struktural, Dengan pendekatan graph-based, model ini dapat menangkap hubungan spasial dan struktural antar region dalam gambar 
    dengan lebih baik dibanding CNN konvensional.
    - Performa Tinggi dengan Model Ringan, Meskipun ringan, MobileViG mampu mencapai akurasi yang kompetitif dengan model yang lebih besar berkat 
    representasi graph yang kaya informasi.
""")

st.markdown("""
    **Kekurangan MobileViG**
    - Kompleksitas Implementasi, Arsitektur graph-based lebih kompleks untuk diimplementasikan dan di-debug dibanding arsitektur CNN standar.
    - Kurang Populer dan Dukungan Komunitas, Sebagai arsitektur yang relatif baru, MobileViG memiliki lebih sedikit resources, pretrained models, 
    dan dukungan komunitas dibanding arsitektur mapan seperti ResNet atau EfficientNet.
    - Sensitivitas terhadap Konstruksi Graph, Performa model sangat bergantung pada bagaimana graph dibangun dari citra, memerlukan tuning yang cermat 
    pada parameter seperti jumlah neighbors dan metode sampling.
""")

st.subheader("Grafik Accuracy & Loss GNN")
st.image("image/gnn_graphic.png", caption="Grafik Accuracy & Loss GNN", use_container_width=True)

st.subheader("Confussion Matrix GNN")
st.image("image/gnn_conf_matrix.jpeg", caption="Confussion Matrix GNN", use_container_width=True)

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