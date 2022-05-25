import tensorflow as tf
import tensorflow_addons as tfa
import matplotlib.pyplot as plt
from keras.regularizers import l2
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

    # Mengatur probabilitas dropout dan hyperparameter lambda untuk regularization
    do_prob = 0.6125  # Candidates 0.6 above
    lambda_l2 = 0.001  # Best 0.001

    # Membuat "block" pertama
    z1 = tf.keras.layers.Dense(units=256,
                               kernel_initializer=tf.keras.initializers.HeNormal(seed=0),
                               kernel_regularizer=l2(lambda_l2))(input_data)
    bn_1 = tf.keras.layers.BatchNormalization()(z1)
    a1 = tf.keras.layers.Activation('relu')(bn_1)
    dropout_1 = tf.keras.layers.Dropout(do_prob, seed=0)(a1)

    # Membuat "block" kedua
    z2 = tf.keras.layers.Dense(units=256, kernel_regularizer=l2(lambda_l2))(dropout_1)
    bn_2 = tf.keras.layers.BatchNormalization()(z2)
    a2 = tf.keras.layers.Activation('relu')(bn_2)
    dropout_2 = tf.keras.layers.Dropout(do_prob, seed=0)(a2)

    # Membuat "block" ketiga
    z3 = tf.keras.layers.Dense(units=256, kernel_regularizer=l2(lambda_l2))(dropout_2)
    bn_3 = tf.keras.layers.BatchNormalization()(z3)
    a3 = tf.keras.layers.Activation('relu')(bn_3)
    dropout_3 = tf.keras.layers.Dropout(do_prob, seed=0)(a3)

    # Membuat "block" keempat
    z4 = tf.keras.layers.Dense(units=256, kernel_regularizer=l2(lambda_l2))(dropout_3 + a1)
    bn_4 = tf.keras.layers.BatchNormalization()(z4)
    a4 = tf.keras.layers.Activation('relu')(bn_4)
    dropout_4 = tf.keras.layers.Dropout(do_prob, seed=0)(a4)

    # Membuat "block" kelima
    z5 = tf.keras.layers.Dense(units=256, kernel_regularizer=l2(lambda_l2))(dropout_4)
    bn_5 = tf.keras.layers.BatchNormalization()(z5)
    a5 = tf.keras.layers.Activation('relu')(bn_5)
    dropout_5 = tf.keras.layers.Dropout(do_prob, seed=0)(a5)

    # Membuat "block" keenam
    z6 = tf.keras.layers.Dense(units=256, kernel_regularizer=l2(lambda_l2))(dropout_5 + a3)
    bn_6 = tf.keras.layers.BatchNormalization()(z6)
    a6 = tf.keras.layers.Activation('relu')(bn_6)
    dropout_6 = tf.keras.layers.Dropout(do_prob, seed=0)(a6)

    # Membuat "block" ketujuh
    z7 = tf.keras.layers.Dense(units=256, kernel_regularizer=l2(lambda_l2))(dropout_6)
    bn_7 = tf.keras.layers.BatchNormalization()(z7)
    a7 = tf.keras.layers.Activation('relu')(bn_7)
    dropout_7 = tf.keras.layers.Dropout(do_prob, seed=0)(a7)

    # Mengumpulkan hasil dari "block" ketujuh ke dalam sebuah lapisan "Dense" dengan aktivasi sigmoid
    outputs = tf.keras.layers.Dense(units=1, activation='sigmoid',
                                    kernel_regularizer=l2(lambda_l2))(dropout_7 + a5)

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

    # Closing the plot
    plt.close()


def model_tf(x_train, y_train, x_dev, y_dev, epoch=200, threshold=0.9, savefile='Classifier.h5'):
    """
    Fungsi ini akan melakukan pemodelan neural network menggunakan Tensorflow sesuai dengan model yang dirancang
    untuk penelitian ini.
    :param x_train: Data variabel independen untuk training
    :param y_train: Data variabel dependen untuk training
    :param x_dev: Data variabel independen untuk validation
    :param y_dev: Data variabel dependen untuk validation
    :param epoch: Jumlah pelatihan neural network
    :param threshold: Batasan dari pembulatan probabilitas, digunakan saat evaluasi metrik
    :param savefile: Nama file untuk menyimpan model
    :return:
    """

    # Mendefinisikan metrik-metrik yang digunakan untuk evaluasi
    metrics = [tf.keras.metrics.BinaryAccuracy(name='accuracy'),
               tf.keras.metrics.Precision(name='precision', thresholds=threshold),
               tf.keras.metrics.Recall(name='recall', thresholds=threshold),
               tfa.metrics.F1Score(name='f1', num_classes=1, threshold=threshold),
               tf.keras.metrics.AUC(name='auc')]

    # Mendefinisikan model Neural Network menggunakan fungsi yang sudah ditentukan
    nn = classifier(x_train.shape[1], )

    # Membuat learning rate decay sesuai dengan n iterasi # Best so far decay 1 steps 0.5
    learning_rate_decay = tf.keras.optimizers.schedules.InverseTimeDecay(initial_learning_rate=0.1,
                                                                         decay_steps=2,
                                                                         decay_rate=0.35)

    # Meng-compile fungsi dan menentukan optimiser, metrik pengujian, dan juga loss function
    nn.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate_decay),
               loss='binary_crossentropy',
               metrics=metrics)

    # Memunculkan summary dari model
    nn.summary()

    # Membuat callback earlystopping apabila model tidak mengalami peningkatan metrik di iterasi yang tinggi
    earlystopping = tf.keras.callbacks.EarlyStopping(monitor='val_f1',
                                                     patience=50,
                                                     verbose=0,
                                                     mode='max',
                                                     restore_best_weights=True)  # Patience 50

    # Menyimpan model beban dari model sementara dengan metrik tertinggi
    # mcp_save = tf.keras.callbacks.ModelCheckpoint('Classifier.hdf5',
    #                                               save_best_only=True,
    #                                               monitor='val_f1',
    #                                               mode='max')

    # Melakukan fitting model dengan data training
    history = nn.fit(x=x_train,
                     y=y_train,
                     verbose=1,
                     epochs=epoch,
                     validation_data=(x_dev, y_dev),
                     callbacks=[earlystopping, ])

    # Menyimpan model yang sudah dilatih
    nn.save(savefile)

    # Menampilkan grafik progresi saat melatih model
    plot_metrics(history, ['loss', 'accuracy', 'precision', 'recall', 'f1', 'auc'], save='History')
