import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns


def classifier(input_shape):
    """
    Fungsi ini akan menginisiasikan model neural net yang digunakan untuk penelitian ini.
    Framework yang digunakan adalah Tensorflow dan menggunakan API fungsional.

    Secara umum, arsitektur dari model yang digunakan dapat dibagi menjadi beberapa "block", isi dari masing-masing
    block adalah lapisan "Dense", lapisan "Batch Normalization", lapisan aktivasi dan lapisan "Dropout".

    Seed diatur menjadi 0 supaya bisa menjamin hasil yang serupa (namun tentu tidak akan sama).

    :param input_shape: Dimensi input yang dimasukkan ke model
    :return: Model dengan arsitektur yang sudah ditentukan di fungsi di bawah ini
    """

    # Membuat lapisan input data dengan dimensi yang ditentukan di input_shape
    input_data = tf.keras.Input(shape=input_shape)

    # Membuat "block" pertama
    z1 = tf.keras.layers.Dense(units=256,
                               kernel_initializer=tf.keras.initializers.HeNormal(seed=0))(input_data)
    bn_1 = tf.keras.layers.BatchNormalization()(z1)
    a1 = tf.keras.layers.Activation('relu')(bn_1)
    dropout_1 = tf.keras.layers.Dropout(.6, seed=0)(a1)

    # Membuat "block" kedua
    z2 = tf.keras.layers.Dense(units=256)(dropout_1)
    bn_2 = tf.keras.layers.BatchNormalization()(z2)
    a2 = tf.keras.layers.Activation('relu')(bn_2)
    dropout_2 = tf.keras.layers.Dropout(.6, seed=0)(a2)

    # Membuat "block" ketiga
    z3 = tf.keras.layers.Dense(units=256)(dropout_2)
    bn_3 = tf.keras.layers.BatchNormalization()(z3)
    a3 = tf.keras.layers.Activation('relu')(bn_3)
    dropout_3 = tf.keras.layers.Dropout(.6, seed=0)(a3)

    # Membuat "block" keempat
    z4 = tf.keras.layers.Dense(units=256)(dropout_3)
    bn_4 = tf.keras.layers.BatchNormalization()(z4)
    a4 = tf.keras.layers.Activation('relu')(bn_4)
    dropout_4 = tf.keras.layers.Dropout(.6, seed=0)(a4)

    # Mengumpulkan hasil dari "block" keempat ke dalam sebuah lapisan "Dense" dengan aktivasi sigmoid
    outputs = tf.keras.layers.Dense(units=1, activation='sigmoid')(dropout_4)

    # Menginisasikan model dengan arsitektur yang tertera
    clf = tf.keras.Model(inputs=input_data, outputs=outputs)
    return clf


def plot_metrics(history, met, save=None, size=(15, 15)):
    """
    Membuat grafik dari history training model Neural Network

    :param history: Dataframe history dari hasil pelatihan model
    :param met: Metrik-metrik yang digunakan untuk mengevaluasi model
    :param save: Nama file apabila grafik ingin disimpan di lokasi eksternal
    :param size: Besar dari grafik yang diinginkan

    :return: Grafik progresi hasil training dengan sumbu x sebagai epoch
    """

    # Menginisiasikan subplots
    plt.subplots(3, 2, figsize=size)

    # Melakukan looping untuk metrik yang digunakan
    for n, metric in enumerate(met):
        # Mengisi subplot yang kosong
        plt.subplot(3, 2, n+1)
        # Membuat grafik dari progresi training di data train
        plt.plot(history.epoch, history.history[metric], label='Train')
        # Membuat grafik dari progresi hasil data validasi
        plt.plot(history.epoch, history.history[f'val_{metric}'],
                 color='orange', linestyle="--", label='Validation', alpha=0.5)
        # Pengaturan grafik
        plt.xlabel('Epoch')
        plt.ylabel(str.title(metric))
        plt.legend()
        sns.despine()

    # Mengecek apabila grafik akan disimpan di lokasi eksternal
    if save is not None:
        plt.savefig(f'{save}.png', dpi=500, transparent=True)

    # Memunculkan grafik
    plt.show()
