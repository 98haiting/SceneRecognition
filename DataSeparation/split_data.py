import csv
import numpy as np
import pandas as pd

ODOR = pd.read_csv("ODOR.csv")
Rmuseum = pd.read_csv("rijksmuseum.csv")
WikiData = pd.read_csv("wikidata_artwork.csv")
ArtPlaces = pd.concat([Rmuseum, WikiData], ignore_index=True)

min_per_class = 1
unique_labels = np.unique(ArtPlaces['Label'])
# Separate the data by class
data_by_class = {label: ArtPlaces[ArtPlaces['Label'] == label] for label in unique_labels}
# Sample from each class
samples = [data.sample(max(int(len(data)*0.12), min_per_class), random_state=42) for data in data_by_class.values()]

Artplace_test = pd.concat(samples)
Artplace_train_val = ArtPlaces.drop(Artplace_test.axes[0])

Artplace_test.to_csv("Artplace_test_new.csv")
Artplace_train_val.to_csv("Artplace_train_new.csv")