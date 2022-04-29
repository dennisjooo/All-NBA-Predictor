from statsmodels.stats.outliers_influence import variance_inflation_factor
import pandas as pd
import statsmodels.api as sm


def calc_vif(data):
    """
    Fungsi ini akan menghitung Variance Inflation Factor untuk setiap kolom yang ada pada input

    :parameter data: Input data yang akan dihitung

    :return: Nilai VIF dari setiap kolom di data dalam bentuk dataframe
    """

    # Membuat dataframe kosong
    vif = pd.DataFrame()

    # Mendefinisikan kolom 'var' sebagai kolom-kolom yang ada di data
    vif['Var'] = data.columns

    # Mengkalkulasikan VIF untuk setiap kolom
    vif['VIF'] = [variance_inflation_factor(data.values, i) for i in range(data.shape[1])]

    return vif


def clean_vif(data, vif_thres=5):
    """
    Fungsi ini akan menghilangkan kolom-kolom yang memiliki VIF lebih dari vif_tres yang ditentukan.
    Kolom-kolom tersebut tidak akan dihilangkan secara langsung, namun satu per satu secara rekursif.
    Fungsi ini menggunakan implementasi fungsi calc_vif

    :param data: Data yang akan dikalkulasikan
    :param vif_thres: Threshold dari VIF yang ditentukan

    :return: Kolom-kolom yang memiliki VIF kurang dari atau sama dengan Threshold
    """

    # Menghitung VIF menggunakan calc_vif
    temp_vif = calc_vif(data)

    # Menghilangkan kolom-kolom yang memiliki VIF > vif_thres
    while temp_vif['VIF'].max() > vif_thres:
        # Mencari kolom-kolom di temp_vif yang tidak memiliki nilai VIF terbesar
        index = temp_vif.loc[temp_vif['VIF'] != temp_vif['VIF'].max()]['Var'].to_list()
        # Menghitung kembali VIF dari kolom-kolom tersebut
        temp_vif = calc_vif(data[index])

    return temp_vif['Var']


def calc_pval(data, target):
    """
    Fungsi ini akan memodelkan data menggunakan Regresi Logistik (sm.Logit()) dan menghitung p-value dari setiap kolom.

    :param data: Data yang akan dimodelkan
    :param target: Variabel dependen yang ingin dimodelkan

    :return: Nilai p-value dari masing-masing kolom yang dimodelkan
    """

    # Menambahkan konstanta ke dataset data
    x = sm.add_constant(data)

    # Melakukan pemodelan menggunakan sm.Logit
    lr = sm.Logit(target, x).fit(disp=False)

    # Memasukan p-values ke dealam sebuah dataframe
    temp = pd.DataFrame(lr.pvalues, columns=['p_val']).drop('const')
    return temp


def p_eliminate(data, target, alpha=0.05):
    """
    Fungsi ini akan mengeliminasi kolom yang memiliki p-value lebih besar dari alpha secara rekursif.
    Fungsi ini akan menggunakan output dari fungsi calc_pval()

    :param data: Data yang ingin dimodelkan
    :param target: Variabel dependen yang ingin dimodelkan dengan Regresi Logistik
    :param alpha: Nilai alpha untuk dibandingkan dengan p-value

    :return: Kolom-kolom yang memiliki p-value <= alpha
    """
    # Mengkalkulasikan p-value menggunakan calc_pval()
    temp_p = calc_pval(data, target)

    # Secara rekursif membuang kolom dengan p-value yang lebih besar dari alpha
    while temp_p['p_val'].max() > alpha:
        # Mencari kolom-kolom di temp_p yang tidak memiliki nilai p-value terbesar
        temp_feat = temp_p.loc[temp_p['p_val'] != temp_p['p_val'].max()].index.to_list()

        # Menghitung kembali p-value dari kolom-kolom tersebut
        temp_p = calc_pval(data[temp_feat], target)

    # Menunjukan hasil dari rekursi di atas
    temp_p.reset_index(inplace=True)
    temp_p.rename(columns={'index': 'Var'}, inplace=True)
    display(temp_p)

    return temp_p['Var'].to_list()
