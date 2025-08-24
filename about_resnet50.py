import streamlit as st

st.title("Informasi Tentang Model CNN")

st.markdown("""
    Aplikasi **Eye👁️Check.AI** menggunakan Convolutional Neural Networks (CNN), khususnya dua arsitektur populer yaitu **ResNet50** dan **VGG16**. 
    Kedua model ini dikenal dengan performa yang unggul dalam mengklasifikasikan citra medis. Berikut adalah penjelasan mendalam mengenai masing-masing model 
    serta alasan mengapa keduanya dipilih dalam aplikasi ini.
""")


st.subheader("Arsitektur ResNet50")

st.markdown("""
    **ResNet50** atau Residual Network dengan 50 layer, merupakan model deep learning yang dirancang untuk menangani permasalahan 
    vanishing gradient yang sering muncul pada jaringan dalam. Dengan adanya residual connections, ResNet50 dapat mempertahankan 
    aliran informasi sehingga mampu belajar dari fitur-fitur gambar yang lebih dalam dan kompleks. Model ini terkenal dalam klasifikasi 
    gambar dan sering digunakan dalam aplikasi medis untuk mendeteksi pola unik pada citra.
""")

st.markdown("""
    **Kelebihan ResNet50**
    - Mengatasi Vanishing Gradient, Dengan residual connections, ResNet50 mampu mencegah hilangnya informasi di jaringan yang sangat dalam, 
    sehingga menghasilkan model yang lebih stabil selama pelatihan.
    - Akurasi Tinggi pada Citra Kompleks, ResNet50 dapat mengenali detail gambar dengan baik, sehingga efektif dalam mendeteksi variasi kecil pada penyakit mata.
    - Kinerja Lebih Baik di Jaringan Dalam, Model ini dapat mencapai kedalaman yang besar tanpa kehilangan akurasi, sehingga mampu mempelajari fitur yang kompleks.
""")
st.markdown("""
    **Kekurangan ResNet50**
    - Waktu Komputasi yang Lama, Dengan 50 layer, ResNet50 membutuhkan waktu pelatihan yang lebih lama dibanding model yang lebih ringan, seperti VGG16.
    - Memerlukan Memori yang Lebih Besar, Karena arsitekturnya yang dalam, model ini membutuhkan GPU dan memori yang lebih tinggi.
    - Overhead Komputasi Residual Connections, Meskipun efektif, residual connections menambah kompleksitas yang sedikit memperlambat inferensi.
""")

st.subheader("Grafik Training & Validation Accuracy ResNet50")
st.image("resnet_training_validation_accuracy.png", caption="Grafik Training & Validation Accuracy ResNet50", use_container_width=True)

st.subheader("Grafik Training & Validation Loss Resnet50")
st.image("resnet_training_validation_loss.png", caption="Grafik Training & Validation Loss ResNet50", use_container_width=True)

st.subheader("Confussion Matrix ResNet50")
st.image("resnet_final_confusion_matrix.png", caption="Confussion Matrix ResNet50", use_container_width=True)

st.subheader("Classification Report ResNet50")
st.image("classification_resnet50.png", caption="Classification Report ResNet50", use_container_width=True)

# Footer
def footer():
    st.markdown("<div class='footer'>© 2024 Muhammad Giat - 210013 - Eye👁️Check.AI </div>", unsafe_allow_html=True)

# Display footer
footer()