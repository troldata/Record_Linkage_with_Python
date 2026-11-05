import pandas as pd

def standardize_strings(df):
    # Delete exta spaces and make all letters lower. In our case there are no extra spaces.
    df = df.copy()
    string_columns = [
    "given_name",
    "surname",
    "address_1",
    "address_2",
    "suburb",
    "state"
    ]
    for column in string_columns:
        df[column] = (df[column].astype("string").str.lower())
    
    df["postcode"] = df["postcode"].astype("string")
    df["soc_sec_id"] = df["soc_sec_id"].astype("string")
    return df

def clean_dates(df, column):
    # Turn into data in date format "%Y%m%d"
    df = df.copy()

    df[column] = pd.to_datetime(
        df[column].astype("Int64").astype(str),
        format="%Y%m%d",
        errors="coerce"
    )
    return df

def clean_street_number(df):
    # Convert data street_number in integer format.
    df = df.copy()
    df["street_number"] = df["street_number"].astype("Int64")
    return df

def clean_postcode(df):
    # If the number of digits less 4, add 0 to the beginning.
    df = df.copy()
    df["postcode"] = (
        df["postcode"]
        .str.zfill(4)
    )
    return df



