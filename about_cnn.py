import streamlit as st

st.title("Informasi Tentang Model CNN")

st.markdown("""
    Model Convolutional Neural Networks (CNN) menggunakan arsitektur **ResNet50**. 
    Berikut adalah penjelasan mendalam mengenai model dan arsitektur.
""")


st.subheader("Arsitektur ResNet50")

st.markdown("""
    **ResNet50** atau Residual Network dengan 50 layer, merupakan model deep learning yang dirancang untuk menangani permasalahan 
    vanishing gradient yang sering muncul pada jaringan dalam. Dengan adanya residual connections, ResNet50 dapat mempertahankan 
    aliran informasi sehingga mampu belajar dari fitur-fitur gambar yang lebih dalam dan kompleks. Model ini terkenal dalam klasifikasi 
    gambar dan sering digunakan dalam aplikasi untuk mendeteksi pola unik pada citra.
""")

st.markdown("""
    **Kelebihan ResNet50**
    - Mengatasi Vanishing Gradient, Dengan residual connections, ResNet50 mampu mencegah hilangnya informasi di jaringan yang sangat dalam, 
    sehingga menghasilkan model yang lebih stabil selama pelatihan.
    - Akurasi Tinggi pada Citra Kompleks, ResNet50 dapat mengenali detail gambar dengan baik, sehingga efektif dalam mendeteksi variasi kecil pada gambar biji kopi.
    - Kinerja Lebih Baik di Jaringan Dalam, Model ini dapat mencapai kedalaman yang besar tanpa kehilangan akurasi, sehingga mampu mempelajari fitur yang kompleks.
""")
st.markdown("""
    **Kekurangan ResNet50**
    - Waktu Komputasi yang Lama, Dengan 50 layer, ResNet50 membutuhkan waktu pelatihan yang lebih lama dibanding model yang lebih ringan.
    - Memerlukan Memori yang Lebih Besar, Karena arsitekturnya yang dalam, model ini membutuhkan GPU dan memori yang lebih tinggi.
    - Overhead Komputasi Residual Connections, Meskipun efektif, residual connections menambah kompleksitas yang sedikit memperlambat inferensi.
""")

st.subheader("Grafik Training & Validation Accuracy CNN")
st.image("image/cnn_training_validation_accuracy.png", caption="Grafik Training & Validation Accuracy CNN", use_container_width=True)

st.subheader("Grafik Training & Validation Loss CNN")
st.image("image/cnn_training_validation_loss.png", caption="Grafik Training & Validation Loss CNN", use_container_width=True)

st.subheader("Confussion Matrix CNN")
st.image("image/cnn_final_confusion_matrix.png", caption="Confussion Matrix CNN", use_container_width=True)

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