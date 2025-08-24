import streamlit as st

st.title("Informasi Tentang Model CNN")

st.markdown("""
    Aplikasi **Eye👁️Check.AI** menggunakan Convolutional Neural Networks (CNN), khususnya dua arsitektur populer yaitu **ResNet50** dan **VGG16**. 
    Kedua model ini dikenal dengan performa yang unggul dalam mengklasifikasikan citra medis. Berikut adalah penjelasan mendalam mengenai masing-masing model 
    serta alasan mengapa keduanya dipilih dalam aplikasi ini.
""")


st.subheader("Arsitektur VGG16")

st.markdown("""
    **VGG16**, yang dikembangkan oleh Visual Geometry Group, terdiri dari 16 lapisan dengan arsitektur yang sederhana namun efektif. 
    Menggunakan filter kecil berukuran 3x3, VGG16 sangat tepat dalam menangkap fitur-fitur penting pada setiap bagian citra. 
    Kemampuan model ini untuk melakukan klasifikasi yang akurat telah menjadikannya pilihan utama dalam berbagai tugas pengenalan gambar.
""")

st.markdown("""
    **Kelebihan VGG16**
    - Arsitektur Sederhana dan Efektif, VGG16 menggunakan filter 3x3 yang kecil, membuatnya mudah untuk diimplementasikan dan dipahami, serta cukup efisien pada data medis.
    - Cepat dalam Inferensi, Dengan lapisan yang lebih sedikit dibanding ResNet50, VGG16 lebih cepat dalam prediksi, ideal untuk aplikasi yang membutuhkan respons cepat.
    - Deteksi Fitur yang Stabil, VGG16 memberikan hasil konsisten untuk citra sederhana dengan pola yang relatif seragam, seperti gambar mata dalam aplikasi ini.
""")
st.markdown("""
    **Kekurangan VGG16**
    - Cenderung Overfitting pada Data Terbatas, Dengan arsitektur yang lebih dangkal, VGG16 terkadang mengalami overfitting jika tidak dilakukan regularisasi yang cukup pada data yang terbatas.
    - Kurang Optimal untuk Fitur Kompleks, VGG16 kesulitan menangkap detail rumit yang lebih baik ditangani oleh model yang lebih dalam seperti ResNet50.
    - Konsumsi Memori Relatif Tinggi, Meski memiliki lebih sedikit layer, penggunaan lapisan fully connected di akhir jaringan menyebabkan penggunaan memori yang lebih besar.
""")

st.subheader("Grafik Training & Validation Accuracy VGG16")
st.image("vgg16_training_validation_accuracy.png", caption="Grafik Training & Validation Accuracy VGG16", use_container_width=True)

st.subheader("Grafik Training & Validation Loss VGG16")
st.image("vgg16_training_validation_loss.png", caption="Grafik Training & Validation Loss VGG16", use_container_width=True)

st.subheader("Confussion Matrix VGG16")
st.image("vgg16_final_confusion_matrix.png", caption="Confussion Matrix VGG16", use_container_width=True)

st.subheader("Classification Report VGG16")
st.image("classification_vgg16.png", caption="Classification Report VGG16", use_container_width=True)

# Footer
def footer():
    st.markdown("<div class='footer'>© 2024 Muhammad Giat - 210013 - Eye👁️Check.AI </div>", unsafe_allow_html=True)

# Display footer
footer()