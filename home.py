import streamlit as st

st.title("Eye👁️Check.AI")

st.subheader("Selamat datang di aplikasi klasifikasi citra penyakit mata")

st.markdown("""
    **Eye👁️Check.AI** adalah aplikasi berbasis web yang membantu mengklasifikasikan penyakit mata menggunakan model deep learning Convolutional Neural Network (CNN). Aplikasi ini dirancang untuk memberikan hasil yang cepat dan akurat, memungkinkan pengguna untuk melakukan pengecekan mata secara mandiri. Dengan antarmuka yang ramah pengguna dan aksesibilitas melalui platform web, EyeCheck.AI memudahkan pengguna untuk mengunggah foto mata dan mendapatkan hasil prediksi dalam waktu singkat.

    **Fokus Klasifikasi Penyakit**
    - Bulging Eyes (Exophthalmos)
    - Katarak (Cataracts)
    - Crossed Eyes (Strabismus)
    - Uveitis
""")

st.image("jenis_penyakit_mata.jpg", caption="Fokus Klasifikasi Jenis Penyakit Mata", use_container_width=True)

st.markdown("""
    **Tujuan Aplikasi**
    
    Aplikasi ini bertujuan untuk membantu pengguna mengklasifikasikan penyakit mata secara dini. Deteksi dini ini memungkinkan penanganan lebih cepat, mengurangi risiko kebutaan, dan mengoptimalkan pengobatan.
    
    **Keunggulan Aplikasi**
    - Menggunakan Model 2 CNN Terbaik  
    
    EyeCheck.AI menggunakan 2 arsitektur CNN canggih yang telah diuji coba, yaitu ResNet50 dan VGG16, yang dikenal dengan akurasinya dalam klasifikasi citra medis.
    - Aksesibilitas Melalui Web 
    
    Aplikasi ini dirancang untuk diakses dengan mudah melalui platform web menggunakan Streamlit, sehingga pengguna dapat mengunggah foto mata dan mendapatkan hasil prediksi dengan cepat.   

    **Pentingnya Kesehatan Mata**
    
    Kesehatan mata memiliki peran vital dalam menjaga kualitas hidup kita sehari-hari. Gangguan pada penglihatan dapat memengaruhi berbagai aspek kehidupan, seperti kemampuan untuk bekerja, belajar, dan menjalani aktivitas sehari-hari secara mandiri. Deteksi dini menjadi kunci utama dalam menjaga kesehatan mata dan mencegah risiko kebutaan yang serius. Menurut data dari Kementerian Kesehatan, lebih dari 80% kasus kebutaan sebenarnya bisa dicegah jika penyakit mata dapat terdeteksi lebih awal dan ditangani dengan tepat.
""")

st.image("mata_sehat.jpg", caption="Gambaran Mata Sehat", use_container_width=True)

st.markdown("""
    **Eye👁️Check.AI** hadir sebagai solusi inovatif untuk membantu mengklasifikasikan secara dini beberapa penyakit mata. Aplikasi ini sangat berguna bagi mereka yang memiliki keterbatasan akses ke dokter spesialis mata, memberikan kemudahan bagi pengguna di daerah terpencil atau yang menghadapi kendala biaya dan waktu dalam melakukan pemeriksaan langsung. Selain itu, EyeCheck.AI turut meningkatkan kesadaran masyarakat akan pentingnya kesehatan mata, mengingat beberapa penyakit mata tidak menunjukkan gejala signifikan pada tahap awal. Dengan menggunakan teknologi berbasis deep learning, EyeCheck.AI memungkinkan Anda untuk lebih peduli dan proaktif terhadap kesehatan mata Anda.
    
    
    **Selamat menggunakan Eye👁️Check.AI!**
""")

# Footer
def footer():
    st.markdown("<div class='footer'>© 2024 Muhammad Giat - 210013 - Eye👁️Check.AI </div>", unsafe_allow_html=True)

# Display footer
footer()
