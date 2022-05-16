import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


def process_data(data, all_nba_dat=None, games=10, dummies=False):
    """
    Memproses data mentah dari statistika pemain dan menambahkan kolom All-NBA

    :parameter data:  Data pemain yang akan dimasukan kolom All-NBA
    :parameter all_nba_dat: Data pemain yang mendapatkan All-NBA
    :parameter dummies: Berupa boolean, apabila True akan mengkonversi kolom posisi menjadi dummy variable
    :parameter games: Jumlah minimal pertandingan yang dimainkan oleh pemain

    :return: Dataframe dari data pemain yang sudah memiliki kolom All-NBA
    """

    # Memilih data pemain yang bermain setidaknya jumlah nilai parameter games
    data = data.loc[data['g'] >= games]

    # Memasukan data All-NBA jika diperlukan
    if all_nba_dat is not None:
        # Membaca data pemain yang mendapatkan All-NBA
        all_nba = pd.read_csv(all_nba_dat)

        # Membuat kolom All-NBA di data yang diinginkan
        data['isAllNBA'] = 0

        # Melakukan looping terhadap tahun-tahun yang ada di data
        for i in data['year'].unique().tolist():
            # Melakukan looping untuk nama-nama pemain yang mendapatkan All-NBA di tahun tersebut
            for j in all_nba.loc[all_nba['Year'] == i]['Name'].unique().tolist():
                # Mengubah nilai kolom isAllNBA menjadi satu apabila pemain mendapatkan penghargaan
                data.loc[(data['year'] == i) & (data['player'] == j), 'isAllNBA'] = 1

    # Mengubah pemain yang memainkan beberapa posisi menjadi hanya posisi utama
    data['pos'] = data['pos'].str.extract(r'([PCS][GF]?)')

    # Membuat dictionary untuk menggantikan nilai posisi
    to_replace = {'PG': 'G',
                  'SG': 'G',
                  'PF': 'F',
                  'SF': 'F'}

    # Menggantikan nilai posisi dengan G atau F (posisi C sudah sesuai format)
    for pos in to_replace.keys():
        data['pos'] = data['pos'].str.replace(pos, to_replace[pos])

    # Membuang data-data yang hilang
    data.dropna(inplace=True)

    # Membuang kolom tahun, umur, dan tim
    data.drop(['tm', 'age', 'year'], axis=1, inplace=True)

    # Mengubah posisi menjadi dummy variable apabila nilai dummies adalah True
    if dummies:
        data = pd.concat([data, pd.get_dummies(data['pos'],
                                               prefix='pos',
                                               drop_first=True)], axis=1).drop(['pos'], axis=1)
    return data


def initial_scaler(data, exclude):
    """
    Melakukan scaling data menggunakan standard scaler
    Hanya untuk data TRAINING
    :param data: Data yang ingin discaling
    :param exclude: Kolom yang ada di data yang tidak ingin discaling

    :return: Data yang sudah discaling
    """

    # Membuat objek scaler
    std_scaler = StandardScaler()

    # Mengambil kolom-kolom numerik
    numerics = data.select_dtypes(np.number).columns

    # Melakukan pengecekan data yang akan tidak dimasukkan
    if exclude is not None:
        numerics = numerics.drop(exclude)

    std_scaler.fit(data[numerics])

    # Melakukan standard scaling ke kolom-kolom numerik yang diinginkan
    data[numerics] = std_scaler.transform(data[numerics])

    return data, std_scaler


def scaler(scale, data, exclude):
    """
    Melakukan scaling data menggunakan standard scaler
    :param scale: Scaler yang digunakan untuk scaling di awal
    :param data: Data yang ingin discaling
    :param exclude: Kolom yang ada di data yang tidak ingin discaling

    :return: Data yang sudah discaling
    """

    # Mengambil kolom-kolom numerik
    numerics = data.select_dtypes(np.number).columns

    # Melakukan pengecekan data yang akan tidak dimasukkan
    if exclude is not None:
        numerics = numerics.drop(exclude)

    # Melakukan standard scaling ke kolom-kolom numerik yang diinginkan
    data[numerics] = scale.transform(data[numerics])

    return data
