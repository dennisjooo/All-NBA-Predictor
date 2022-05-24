import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def prob_threshold_plot(data, hue_tar, threshold=0.5, size=(10, 10), save_name='something'):
    """
    Fungsi ini akan membuat scatter plot dari hasil probabilitas prediksi relatif terhadap yang hasil asli.

    :param data: Data yang ingin dibuat grafiknya
    :param hue_tar: Kategori dari hasil asli
    :param threshold: Threshold dari probabilitas yang diinginkan
    :param size: Besar dari grafik yang diinginkan
    :param save_name: Menentukan nama dari gambar yang akan disimpan
    """

    # Menyiapkan figur grafik
    plt.figure(figsize=size)

    # Membuat scatterplot
    sns.scatterplot(x=range(1, len(data) + 1),
                    y=data,
                    hue=hue_tar, alpha=0.5)

    # Membuat garis threshold
    plt.axhline(y=threshold, color='green', alpha=0.5, linewidth=3)

    # Pengaturan grafik
    plt.title('Probability and Threshold', fontsize=10, y=1.02)
    plt.ylabel('Probability')
    sns.despine()

    # Menunjukan hasil grafik
    plt.savefig(f'Graph\\{save_name}.png', transparent=True)
    plt.show()


def eval_metrics(actual, predicted):
    """
    Fungsi ini akan menghitung metrik-metrik yang akan digunakan untuk mengevaluasi model

    :param actual: Nilai asli dari variabel dependen
    :param predicted: Nilai prediksi dari variabel dependen
    :return: Sebuah dictionary yang berisi nilai metrik yang dicari
    """

    # Membuat confusion matrix dari hasil asli dan prediksi
    cm = pd.crosstab(actual, predicted)

    # Membuat dictionary dari metrik-metrik yang diinginkan
    temp = {
        'tp': cm.iloc[1, 1],
        'tn': cm.iloc[0, 0],
        'fn': cm.iloc[1, 0],
        'fp': cm.iloc[0, 1],
        'accuracy': accuracy_score(actual, predicted),
        'recall': recall_score(actual, predicted),
        'precision': precision_score(actual, predicted),
        'f1': f1_score(actual, predicted)
    }
    return temp


def cm_heatmap(actual, predicted, size=(8, 8),save_name='Something'):
    """
    Fungsi ini akna membuat heatmap confusion matrix dari nilai asli dan prediksi variabel dependen

    :param actual: Nilai asli variabel dependen
    :param predicted: Nilai prediksi variabel dependen
    :param size: Besar dari grafik yang diinginkan
    :param save_name: Menentukan nama dari gambar yang akan disimpan
    """

    # Mencari confusion matrix dari nilai asli dan prediksi
    cm = pd.crosstab(actual, predicted)

    # Membuat figur grafik
    plt.figure(figsize=size)

    # Membuat heatmap dari confusion matrix
    sns.heatmap(cm, annot=True, fmt='g')

    # Plot settings
    plt.xlabel('Prediction')
    plt.ylabel('Actual')

    # Menunjukan hasil
    plt.savefig(f'Graph\\{save_name}.png',transparent=True)
    plt.show()


def optimal_threshold(actual, predicted, n_candidates, metric='accuracy', bound=(0, 100)):
    """
    Fungsi ini akan mencari threshold yang terbaik berdasarkan metrik yang dipilih.

    :param actual: Nilai label asli untuk data
    :param predicted: Nilai probabilitas dari prediksi model
    :param n_candidates: Jumlah dari interval threshold yang ingin dicoba
    :param metric: Netrik yang dipilih untuk mengoptimasi threshold
    :param bound: Rentang threshold yang diinginkan

    :return: Mengembalikan threshold yang diinginkan
    """

    # Membuat dictionary dari pilihan metrik evaluasi
    choices = {'accuracy': accuracy_score,
               'recall': recall_score,
               'precision': precision_score,
               'f1': f1_score}

    # Membuat daftar angka yang sesuai dengan bound dan n_candidates
    candidates = np.linspace(bound[0], bound[1], n_candidates) / 100

    # Menginisiasikan nilai terbaik threshold dan hasil dari metrik
    best_threshold = 0
    score = 0

    # Melakukan looping terhadap daftar angka yang sudah dibuat
    for threshold in candidates:
        # Menggunakan dictionary choice untuk menarik dan menghitung metrik
        # np.where digunakan untuk membulatkan probabilitas prediksi
        current_score = choices[metric](actual, np.where(predicted > threshold, 1, 0))

        # Melakukan perbandingan nilai metrik sekarang dengan nilai metrik sebelumnya
        if current_score >= score:
            # Mengubah nilai metrik terbaik jika perlu
            score = current_score
            # Mengubah threshold terbaik jika perlu
            best_threshold = threshold

    # Menampilkan hasil skor dan threshold terbaik
    print(f'Threshold terbaik adalah {best_threshold} dengan skor {metric} {score * 100}%')

    return best_threshold


def create_compare_frame(train_metrics, dev_metrics):
    """
    Fungsi ini hanya bisa digunakan di penelitian ini saja. Tujuan dari fungsi ini adalah membuat sebuah dataframe
    yang mengandung semua metrik evaluasi dari setiap dataset dan setiap model.

    :param train_metrics: Berupa list atau tupple, berisi metrik-metrik yang dievaluasi di data training
    :param dev_metrics: Berupa list atau tupple, berisi metrik-metrik yang dievaluasi di data development

    :return: Sebuah dataframe dengan semua metrik yang digunakan di penelitian ini
    """

    # Membuat sebuah dictionary, index model sudah diisi dengan model yang digunakan di penelitian
    metrics_train = {
        'model': ['logistic regression', 'random forest', 'neural network'],
        'tp': [],
        'tn': [],
        'fn': [],
        'fp': [],
        'accuracy': [],
        'recall': [],
        'precision': [],
        'f1': []
    }

    # Mengimpor fungsi deepcopy
    from copy import deepcopy

    # Membuat salinan dari metrics_train
    metrics_dev = deepcopy(metrics_train)

    # Melakukan looping untuk seluruh elemen di train_metrics dan mengatur elemen di metrics_train sesuai dengan
    # input di train_metrics
    for model_metric in train_metrics:
        for index in model_metric:
            metrics_train[index].append(model_metric[index])

    # Melakukan looping untuk seluruh elemen di train_metrics dan mengatur elemen di metrics_dev sesuai dengan
    # input di dev_metrics
    for model_metric in dev_metrics:
        for index in model_metric:
            metrics_dev[index].append(model_metric[index])

    # Membuat dataframe dari metrics_train dan metrics_dev
    train = pd.DataFrame(metrics_train)
    dev = pd.DataFrame(metrics_dev)

    # Membuat kolom dataset asal dari masing-masing dataframe
    train['dataset'] = 'train'
    dev['dataset'] = 'dev'

    # Menggambungkan kedua dataframe
    compare = pd.concat([train, dev], ignore_index=True)

    return compare


def select_all_nba(data, prob):
    """
    Fungsi ini digunakan untuk mencari pemain yang paling mungkin memenangkan penghargaan All-NBA.
    Aturan dari pemilihan adalah 6 Guards, 6 Forwards, dan 3 Centres.

    :param data: Data yang digunakan untuk membuat probabilitas
    :param prob: Probabilitas yang ingin digunakan

    :return: Dataframe dari 15 pemain dengan probabilitas tertinggi untuk memenangkan penghargaan All-NBA
    """

    # Membuat dataframe kosong
    all_nba = pd.DataFrame([])

    # Melakukan looping untuk ketiga posisi
    for i in data['pos'].unique().tolist():
        # Mengambil 3 pemain untuk posisi Centre
        if i == 'C':
            all_nba = pd.concat([all_nba, data.loc[data['pos'] == i].sort_values(by=prob,
                                                                                 ascending=False).head(3)])
        # Mengambil 6 pemain untuk posisi Guards dan Forwards
        else:
            all_nba = pd.concat([all_nba, data.loc[data['pos'] == i].sort_values(by=prob,
                                                                                 ascending=False).head(6)])

    return all_nba.sort_values(by=['pos', prob], ascending=False).reset_index(drop=True)[['player', 'pos', prob]]
