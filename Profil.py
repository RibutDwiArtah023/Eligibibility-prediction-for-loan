# import plotly.express as px
# from sklearn.utils.validation import joblib

import streamlit as st
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn import neighbors, datasets
from sklearn.preprocessing import MinMaxScaler

import pickle

from sklearn import metrics

# intial template
# px.defaults.template = "plotly_dark"
# px.defaults.color_continuous_scale = "reds"


st.title('Prediksi Kelayakan Untuk Pinjaman Rumah')
st.write("========================================================================================")
st.write("""
Selamat datang di aplikasi Prediksi Kelayakan Untuk Pinjaman Rumah
\nAplikasi ini untuk memprediksi Kelayakan Pinjaman Rumah, apakah pelanggan layak untuk melakukan pinjaman rumah atau tidak layak untuk melakukan pinjaman rumah
""")

df= pd.read_csv("https://raw.githubusercontent.com/RibutDwiArtah023/Eligibibility-prediction-for-loan/master/Loan_Data.csv")

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Profil","Deskripsi Data", "Import Data", "Preprocessing", "Modelling", "Implementation"])

# create content
with tab1:
    st.write("""
    <h5>Profil</h5>
    """, unsafe_allow_html=True)
    st.container()
    st.write(''' 
                \nSaya Ribut Dwi Artah dengan nomor induk mahasiswa 200411100023.  
                \nKali ini saya akan mencoba melakukan penambangan data dan memprediksi kelayakan untuk pinjaman rumah.
                \nApa maksud dari prediksi kelayakan untuk pinjaman rumah?
                \nPerusahaan Dream Housing Finance menangani segala jenis pinjaman rumah. Mereka hadir di semua daerah perkotaan, semi perkotaan dan pedesaan.
                Pelanggan pertama-tama mengajukan pinjaman rumah, setelah itu perusahaan memvalidasi kelayakan pelanggan untuk pinjaman.
                Perusahaan ingin mengotomatiskan proses kelayakan pinjaman (real time) berdasarkan detail pelanggan yang diberikan saat mengisi formulir aplikasi online.
                Rincian tersebut adalah Jenis Kelamin, Status Perkawinan, Pendidikan, Jumlah Tanggungan, Pendapatan, Jumlah Pinjaman, Riwayat Kredit dan lain-lain. 
                Untuk mengotomatiskan proses ini, mereka telah menyediakan kumpulan data untuk mengidentifikasi segmen pelanggan yang memenuhi syarat untuk jumlah 
                pinjaman sehingga mereka dapat secara khusus menargetkan pelanggan tersebut.
                \nUntuk pertanyaan lebih lanjut bisa menghubungi saya melalui :
                \nEmail : ributdwi01@gmail.com 
                \nGithub : RibutDwiArtah023
                \nInstagram : ributdwii_ 
                \nTerimakasih sudah mengunjungi aplikasi Prediksi Kelayakan Pinjaman Rumah ini. 
                \nSemoga bermanfaat, see you ^^
            ''')
with tab2:
    st.write("""
    <h5>Deskripsi Data</h5>
    """, unsafe_allow_html=True)

    st.write("ELIGIBILITY PREDICTION FOR LOAN DATASET")
    st.write("Dataset yang digunakan mengandung data yang bertipe kategorikal dan numerik")
    st.write("Fitur - fitur yang terdapat dalam dataset : ")
    st.write("""
    <ol>
    <li>Gender : Jenis Kelamin (Laki-laki = 1, Perempuan = 0)</li>
    <li>Married : Status Perkawinan (Iya = 1, Tidak = 0)</li>
    <li>Dependents : Jumlah Tanggungan, yaitu jumlah orang yang di tanggung hidupnya oleh pemohon</li>
    <li>Education : Pendidikan, yaitu status pendidikan dari pemohon (Lulus = 0, Tidak Lulus = 1)</li>
    <li>Self_Employed : Wiraswasta, yaitu kategori pekerjaan dari pemohon (Jika wiraswasta = 1 dan jika bukan wiraswasta = 0)</li>
    <li>ApplicantIncome : Pendapatan Pemohon</li>
    <li>CoapplicantIncome : Pendapatan Pemohon Bersama</li>
    <li>LoanAmount : Jumlah Pinjaman (dalam ribuan)</li>
    <li>Loan_Amount_Term : Jangka Waktu Jumlah Pinjaman (dalam beberapa bulan)</li>
    <li>Credit_History : Riwayat Kredit untuk memenuhi pedoman (Jika pernah = 1, jika tidak pernah = 0)</li>
    <li>Property_Area : Area Properti (yang terdiri dari Rural = 0, Semi Urban = 1, Urban = 2)</li>
    <li>Loan_Status : Hasil status pinjaman, (Y = 1 untuk layak melakukan peminjaman rumah, dan N = 0 untuk tidak layak melakukan peminjaman rumah</li>
    </ol>
    """,unsafe_allow_html=True)

    st.write("""
    <h6>Sumber Data</h6>
    """, unsafe_allow_html=True)
    st.write("Link Dataset : https://www.kaggle.com/datasets/devzohaib/eligibility-prediction-for-loan")
    st.write("Link Repository Github : https://github.com/RibutDwiArtah023/Eligibibility-prediction-for-loan")


with tab3:
    st.write("""
    <h5>Import Data</h5>
    """, unsafe_allow_html=True)
    st.container()
    uploaded_files = st.file_uploader("Upload file CSV", accept_multiple_files=True)
    for uploaded_file in uploaded_files:
        # uplod file
        df = pd.read_csv(uploaded_file)
        st.write(" **Nama File Anda :** ", uploaded_file.name)
        st.write(df)

with tab4:
    st.write("""
    <h5>Preprocessing</h5>
    """, unsafe_allow_html=True)
    st.write('''Preprocessing adalah proses menyiapkan data dasar atau inti sebelum melakukan proses lainnya.
    Pada dasarnya data preprocessing dapat dilakukan dengan membuang data yang tidak sesuai atau mengubah data menjadi bentuk yang lebih mudah untuk diproses oleh sistem.
    Proses pembersihan meliputi penghilangan duplikasi data, pengisian atau penghapusan data yang hilang, pembetulan data yang tidak konsisten, dan pembetulan salah ketik.
    Seperti namanya, normalisasi dapat diartikan secara sederhana sebagai proses menormalkan data dari hal-hal yang tidak sesuai''')
    st.container()
    from sklearn.preprocessing import LabelEncoder
    labelencoder = LabelEncoder()
    df['Gender'] = labelencoder.fit_transform(df['Gender'])
    df['Married'] = labelencoder.fit_transform(df['Married'])
    df['Education'] = labelencoder.fit_transform(df['Education'])
    df['Self_Employed'] = labelencoder.fit_transform(df['Self_Employed'])
    df['Property_Area'] = labelencoder.fit_transform(df['Property_Area'])
    df['Loan_Status'] = labelencoder.fit_transform(df['Loan_Status'])
    scaler = st.radio(
    "Pilih metode normalisasi data",
    ('Tanpa Scaler', 'MinMax Scaler'))
    if scaler == 'Tanpa Scaler':
        st.write("Dataset Tanpa Scaler : ")
        df_drop_column=df.drop(['Loan_ID','Dependents'], axis=1)
        df_depebdents=df["Dependents"].replace('3+', '3')
        df_new= pd.concat([df_depebdents,df_drop_column], axis=1)
    elif scaler == 'MinMax Scaler':
        st.write("Dataset setelah Scaling dengan MinMax Scaler: ")
        scaler = MinMaxScaler()
        df_for_scaler = pd.DataFrame(df, columns = ['ApplicantIncome',	'CoapplicantIncome',	'LoanAmount',	'Loan_Amount_Term'])
        df_for_scaler = scaler.fit_transform(df_for_scaler)
        df_for_scaler = pd.DataFrame(df_for_scaler,columns = ['ApplicantIncome',	'CoapplicantIncome',	'LoanAmount',	'Loan_Amount_Term'])
        df_drop_column_for_minmaxscaler=df.drop(['Loan_ID','Dependents', 'ApplicantIncome',	'CoapplicantIncome',	'LoanAmount',	'Loan_Amount_Term'], axis=1)
        df_depebdents=df["Dependents"].replace('3+', '3')
        df_new = pd.concat([df_depebdents,df_for_scaler,df_drop_column_for_minmaxscaler], axis=1)
    df_new.dropna(inplace=True,axis=0)
    st.write(df_new)

with tab5:
    st.write("""
    <h5>Modelling</h5>
    """, unsafe_allow_html=True)
    st.container()
    X=df_new.iloc[:,0:11].values
    y=df_new.iloc[:,11].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,stratify=y, random_state=42)
    algoritma = st.radio(
    "Pilih model yang anda inginkan untuk cek akurasi",
    ('KNN','Naive Bayes','Random Forest','Ensemble Stacking'))
    if algoritma=='KNN':
        model = KNeighborsClassifier(n_neighbors=3)
        filename='knn.pkl'
        st.write("""
        <h5>Pengertian</h5>
        """, unsafe_allow_html=True)
        st.write('''Algoritma KNN (K-Nearest Neighbour) mengasumsikan bahwa sesuatu yang mirip akan ada dalam jarak yang berdekatan atau bertetangga. 
        Artinya data-data yang cenderung serupa akan dekat satu sama lain. KNN menggunakan semua data yang tersedia dan 
        mengklasifikasikan data atau kasus baru berdasarkan ukuran kesamaan atau fungsi jarak. Data baru kemudian ditugaskan ke kelas 
        tempat sebagian besar data tetangga berada.''')
    elif algoritma=='Naive Bayes':
        model = GaussianNB()
        filename='naivebayes.pkl'
        st.write("""
        <h5>Pengertian</h5>
        """, unsafe_allow_html=True)
        st.write(''' Metode yang juga dikenal sebagai Naive Bayes Classifier ini menerapkan teknik supervised klasifikasi objek di masa depan dengan menetapkan
        label kelas ke instance/catatan menggunakan probabilitas bersyarat. Probabilitas bersyarat adalah ukuran peluang suatu peristiwa yang terjadi berdasarkan 
        peristiwa lain yang telah (dengan asumsi, praduga, pernyataan, atau terbukti) terjadi Rumus: P(A???B) = P(B???A)P(A)P(B). Adapun salah satu jenis naive bayes adalah gausian. 
        Distribusi Gaussian adalah asumsi pendistribusian nilai kontinu yang terkait dengan setiap fitur berisi nilai numerik. Ketika diplot, akan muncul kurva berbentuk lonceng
        yang simetris tentang rata-rata nilai fitur.''')
    elif algoritma=='Random Forest':
        model = RandomForestClassifier(n_estimators = 100)
        filename='randomforest.pkl'
        st.write("""
        <h5>Pengertian</h5>
        """, unsafe_allow_html=True)
        st.write(''' Random forest adalah kombinasi dari  masing ??? masing tree yang baik kemudian dikombinasikan ke dalam satu model. Random Forest bergantung pada sebuah
        nilai vector random dengan distribusi yang sama pada semua pohon yang masing masing decision tree memiliki kedalaman yang maksimal. Random forest adalah classifier 
        yang terdiri dari classifier yang berbentuk pohon {h(x, ?? k ), k = 1, . . .} dimana ??k adalah random vector yang diditribusikan secara independen dan masing masing 
        tree pada sebuah unit kan memilih class yang paling popular pada input x. Berikut ini karakteristik akurasi pada random forest.''')
    elif algoritma=='Ensemble Stacking':
        estimators = [
            ('rf_1', RandomForestClassifier(n_estimators=10, random_state=42)),
            ('knn_1', KNeighborsClassifier(n_neighbors=10))             
        ]
        model = StackingClassifier(estimators=estimators, final_estimator=GaussianNB())
        filename='stacking.pkl'
        st.write("""
        <h5>Pengertian</h5>
        """, unsafe_allow_html=True)
        st.write(''' Metode ensemble adalah algoritma dalam pembelajaran mesin (machine learning) dimana algoritma ini sebagai pencarian solusi prediksi terbaik dibandingkan dengan
        algoritma yang lain karena metode ensemble ini menggunakan beberapa algoritma pembelajaran untuk pencapaian solusi prediksi yang lebih baik daripada algoritma yang bisa diperoleh 
        dari salah satu pembelajaran algoritma kosituen saja. Stacking merupakan cara untuk mengkombinasi beberapa model, dengan konsep meta learner yang dipakai setelah bagging dan boosting. 
        Tidak seperti bagging dan boosting, stacking memungkinkan mengkombinasikan model dari tipe yang berbeda.''')
    model.fit(X_train, y_train)
    Y_pred = model.predict(X_test) 
    score=metrics.accuracy_score(y_test,Y_pred)
    loaded_model = pickle.load(open(filename, 'rb'))
    st.write("""
        <h5>Hasil Akurasi</h5>
        """, unsafe_allow_html=True)
    st.write(f"Akurasi : {score*100} %")

with tab6:
    st.write("""
    <h5>Implementation</h5>
    """, unsafe_allow_html=True)
    Gender=st.selectbox(
        'Pilih Jenis Kelamin',
        ('Laki-laki','Perempuan')
    )
    if Gender=='Laki-laki':
        Gender=1
    elif Gender=='Perempuan':
        Gender=0
    Married=st.selectbox(
        'Pilih Status Pernikahan',
        ('Iya','Tidak')
    )
    if Married=='Iya':
        Married=1
    elif Married=='Tidak':
        Married=0
    Dependents=st.number_input("Jumlah tanggungan : ")
    Education=st.selectbox(
        'Pilih Status Pendidikan',
        ('Lulus','Tidak Lulus')
    )
    if Education=='Lulus':
        Education=0
    elif Education=='Tidak Lulus':
        Education=1
    Self_Employed=st.selectbox(
        'Pilih Status Pekerjaan',
        ('Wiraswasta','Bukan Wiraswasta')
    )
    if Self_Employed=='Wiraswasta':
        Self_Employed=1
    elif Self_Employed=='Bukan Wiraswasta':
        Self_Employed=0
    ApplicantIncome=st.number_input("Pendapatan Pemohon : ")
    CoapplicantIncome=st.number_input("Pendapatan Pemohon Bersama : ")
    LoanAmount=st.number_input("Jumlah Pinjaman : ")
    Loan_Amount_Term=st.number_input("Jangka Waktu Jumlah Pinjaman : ")
    Credit_History=st.selectbox(
        'Riwayat Kredit',
        ('Pernah','Tidak Pernah')
    )
    if Credit_History=='Pernah':
        Credit_History=1
    elif Credit_History=='Tidak Pernah':
        Credit_History=0
    Property_Area=st.selectbox(
        'Kawasan Properti',
        ('Urban','Semi Urban','Rural')
    )
    if Property_Area=='Rural':
        Property_Area=0
    elif Property_Area=='Semi Urban':
        Property_Area=1
    elif Property_Area=='Urban':
        Property_Area=2
    prediksi=st.button("Prediksi")
    if prediksi:
        if scaler == 'Tanpa Scaler':
            dataArray = [Dependents, Gender, Married, Education, Self_Employed, ApplicantIncome, CoapplicantIncome, LoanAmount, Loan_Amount_Term, Credit_History, Property_Area]
        else:
            ApplicantIncome_proceced = (ApplicantIncome - df['ApplicantIncome'].min(axis=0)) / (df['ApplicantIncome'].max(axis=0) - df['ApplicantIncome'].min(axis=0))
            CoapplicantIncome_proceced = (CoapplicantIncome - df['CoapplicantIncome'].min(axis=0)) / (df['CoapplicantIncome'].max(axis=0) - df['CoapplicantIncome'].min(axis=0))
            LoanAmount_proceced = (LoanAmount - df['LoanAmount'].min(axis=0)) / (df['LoanAmount'].max(axis=0) - df['LoanAmount'].min(axis=0))
            Loan_Amount_Term_proceced = (Loan_Amount_Term - df['Loan_Amount_Term'].min(axis=0)) / (df['Loan_Amount_Term'].max(axis=0) - df['Loan_Amount_Term'].min(axis=0))
            dataArray = [Dependents, ApplicantIncome_proceced,	CoapplicantIncome_proceced,	LoanAmount_proceced,	Loan_Amount_Term_proceced,	Gender,	Married,	Education,	Self_Employed,	Credit_History,	Property_Area]
        pred = loaded_model.predict([dataArray])
        if int(pred[0])==0:
            st.error(f"Hasil Prediksi : Tidak layak meminjam")
        elif int(pred[0])==1:
            st.success(f"Hasil Prediksi : Layak meminjam")