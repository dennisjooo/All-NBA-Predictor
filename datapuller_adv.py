# -*- coding: utf-8 -*-
"""
Created on Mon Apr 11 14:25:21 2022

@author: Dennis Jonathan

Credits to Angelica Dietzel
From her article called A Step-by-Step Guide to Web Scraping NBA Data With Python, Jupyter, BeautifulSoup and Pandas
"""


import requests
from bs4 import BeautifulSoup
import pandas as pd


import warnings
warnings.filterwarnings('ignore')


def br_parser(target_year, url):
    """
    Fungsi ini akan menarik data dari basketball-reference.com menggunakan algoritma yang dibuat oleh Dietzel
    :param target_year: Tahun dari data yang ingin ditarik
    :param url: Link dari halaman basketball-reference yang ingin ditarik

    :return: Dataframe yang berisi dengan data pemain di tahun tersebut
    """

    # Mengubah URL yang diinginkan dengan tahun
    link = url.format(target_year)

    # Membuat request ke URL tersebut dan menarik data html
    r = requests.get(link)
    r_html = r.text

    # Menggunakan Beautiful Soup untuk memilah data html
    soup = BeautifulSoup(r_html, 'html.parser')

    # Menemukan kelas full_table untuk menarik tabel dari data
    table = soup.find_all(class_="full_table")

    # Menarik nama-nama kolom dari data
    head = soup.find(class_="thead")
    column_names_raw = [head.text for _ in head][0]
    column_names_polished = column_names_raw.replace("\n", ",").split(",")[2:-1]

    # Membuat list kosong untuk menyimpan data seluruh pemain
    players = []

    # Melakukan looping untuk setiap pemain yang ada
    for i in range(len(table)):
        # Membuat list kosong untuk menyimpan data per pemain
        player_ = []

        # Menarik data per pemain dari html
        for td in table[i].find_all("td"):
            player_.append(td.text)

        # Menggabungkan data per pemain dengan data total
        players.append(player_)

    # Membuat dataframe dari data pemain dengan nama kolom yang sudah diambil
    df = pd.DataFrame(players, columns=column_names_polished)

    # Melakukan perbaikan nama pemain yang tidak benar
    df.Player = df.Player.str.replace(r"\*", '', regex=True)

    # Membuat kolom tahun di dataframe
    df['Year'] = year

    return df


# Membuat dataframe kosong
data = pd.DataFrame()

# Melakukan looping untuk tahun-tahun yang diinginkan
for year in range(1989, 2023):

    # Mengambil data advanced stats per tahun menggunakan fungsi yang sudah dibuat
    advanced = br_parser(year, 'https://www.basketball-reference.com/leagues/NBA_{}_advanced.html')

    # Mengubah nama kolom MP (Minutes Played) menjadi TMP (Total Minutes Played)
    advanced.rename(columns={'MP': 'TMP'}, inplace=True)

    # Menggabungkan dataframe untuk tahun 1989 hingga 2021 dan membuat file baru untuk tahun 2022.
    # Selain itu juga mengubah format nama kolom menjadi huruf kecil dan mengubah spasi menjadi "_"
    if year == 1989:
        data = advanced
        data.columns = data.columns.str.replace(r"\s", '_').str.lower()
        data.drop("_", axis=1, inplace=True)

    elif year == 2022:
        test = advanced
        test.columns = test.columns.str.replace(r"\s", '_').str.lower()
        test.drop("_", axis=1, inplace=True)
        test.to_csv(
            'Documents/Dennis/Prasetiya Mulya/Semester 6/Research and Methodology/Final Project/Project/test.csv',
            index=False)

    else:
        temp = advanced
        temp.columns = temp.columns.str.replace(r"\s", '_').str.lower()
        temp.drop("_", axis=1, inplace=True)

        data = pd.concat([data, temp], ignore_index=True)

    # Menunjukan status pengerjaan
    print('Year {} done'.format(year))

# Menyimpan data 1989-2021 menjadi sebuah file .csv
data.to_csv(
    'Documents/Dennis/Prasetiya Mulya/Semester 6/Research and Methodology/Final Project/Project/traindev.csv',
    index=False)
