import pandas as pd
from recordlinkage.preprocessing import phonetic

# Soundex method
def add_phonetic_features(df):
    df = df.copy()

    df["phonetic_given_name"] = phonetic(
        df["given_name"],
        method="soundex"
    )

    df["phonetic_surname"] = phonetic(
        df["surname"],
        method="soundex"
    )
    return df

