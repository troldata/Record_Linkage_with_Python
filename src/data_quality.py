import pandas as pd

def dataset_info(dfA, dfB):
    print("Dataset A:")
    print(dfA.head())
    dfA.info()

    print("\nDataset B:")
    print(dfB.head())
    dfB.info()
#dataset_info(dfA, dfB)

def basic_quality_report(df, dataset_name):
    results = []

    for column in df.columns:
        series = df[column]
        non_null = series.dropna()

        result = {
            "Dataset": dataset_name,
            "Column": column,
            "Data type": str(series.dtype),
            "Total records": len(series),
            "Missing": series.isna().sum(),
            "Missing (%)": round(series.isna().mean() * 100, 2),
            "Unique values": series.nunique(),
            "Repeating values": series.duplicated().sum()
        }

        if series.dtype == "object":
            values = non_null.astype(str)
            result["Leading/trailing spaces"] = (values != values.str.strip()).sum()
            result["Multiple spaces"] = (values.str.contains(r"\s{2,}", regex=True)).sum()
            result["Mixed case candidates"] = (values != values.str.lower()).sum()

        results.append(result)

    return pd.DataFrame(results)

"""
Data Frame A: Column soc_sec_id contains no missing values. All values are unique, with no repeating values.
              The remaining columns contain repeating values.
              Column postcode contains no missing values.
              More than 3% of the values are missing in the columns street_number and address_2.
              No extra spaces were detected in columns with string data types. No inconsistent letter casing was identified.
              
Data Frame B: Column soc_sec_id contains no missing values. All values are unique, with no duplicates.
              The remaining columns contain duplicate values.
              Column postcode contains no missing values.
              More than 3% of the values are missing in the columns street_number, address_1, address_2, given_name, and date_of_birth.
              No extra spaces were detected in columns with string data types. No inconsistent letter casing was identified.
              
Conclusion:
Dataset B shows a broader distribution of missing values across attributes relevant to record linkage, particularly given_name and date_of_birth. 
These missing values may reduce the reliability of exact matching.
"""

def check_dates(df, column):
# Check of dates on missing data, correct format, future dates.
    raw_dates = df[column]

    date_strings = (
        raw_dates.dropna()
        .astype("int64")
        .astype(str)
    )

    dates = pd.to_datetime(
        date_strings,
        format="%Y%m%d",
        errors="coerce"
    )

    invalid_dates = dates.isna()
    future_dates = dates.notna() & (
        dates > pd.Timestamp.today()
    )

    return {
        "Invalid dates": invalid_dates.sum(),
        "Future dates": future_dates.sum(),
        "Invalid examples": date_strings[invalid_dates].head(10).tolist(),
        "Future examples": date_strings[future_dates].head(10).tolist()
    }    
"""
Dataset A and B:
    The values ​​are stored as float64, but they represent dates in the numerical format YYYYMMDD.
    The column contain the missing data. 
    There is no future dates.
Dataset B contains 64 invalid date-of-birth values. 
"""

def check_postcodes(df, column):
    # Check the data format. The postcode of Australia should contain 4 number. The first digit can be 0.
    raw_values = df[column].dropna()

    numeric_values = pd.to_numeric(raw_values, errors="coerce")

    integer_mask = numeric_values.notna() & (numeric_values % 1 == 0)

    values = numeric_values[integer_mask].astype("int64").astype(str)

    invalid_format = ~values.str.fullmatch(r"\d{4}")

    invalid_count = ((~integer_mask).sum() + invalid_format.sum())

    return {
        "Invalid format": invalid_count,
        "Invalid examples": raw_values.loc[
            ~integer_mask
        ].head(10).tolist() + raw_values.loc[
            raw_values.index[integer_mask][invalid_format]
        ].head(10).tolist()
    }
"""
Some postcode values do not conform to the expected four-digit format.
Three-digit values were identified in both datasets.
These may indicate leading-zero loss during data import or invalid source values.
"""

def check_street_numbers(df, column):
    # Check of the street_number on the missing, zero, negative and extemely big number.
    values = pd.to_numeric(df[column], errors="coerce").dropna()
     
    return {
        "Zero values": (values == 0).sum(),
        "Negative values": (values < 0).sum(),
        "Decimal values": (values % 1 != 0).sum(),
        "Unique values": values.nunique(),
        "Maximum value": values.max(),
        "Minimum value": values.min()
    }

"""
Dataset A contains an unusually high street number of 9999.0.
Dataset B contains an unusually high street number of 14430.0.
"""

