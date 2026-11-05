import pandas as pd
import recordlinkage
from IPython.display import display

def create_comparison():
    compare = recordlinkage.Compare()

    # String similarity: Jaro-Winkler
    compare.string("given_name", "given_name", method="jarowinkler", label="given_name_match")
    compare.string("surname", "surname", method="jarowinkler", label="surname_match")
    compare.string('suburb', 'suburb', method='jarowinkler', label="suburb_match")
    compare.string('address_1', 'address_1', method='jarowinkler', label="address_1_match")
    compare.string('address_2', 'address_2', method='jarowinkler', label="address_2_match")
    
    # Exact comparison
    compare.exact('phonetic_given_name', 'phonetic_given_name', label="phonetic_given_name_match")
    compare.exact('phonetic_surname', 'phonetic_surname', label='phonetic_surname_match')
    compare.exact('street_number', 'street_number', label='street_number_match')
    compare.exact("date_of_birth", "date_of_birth", label="dob_match")
    compare.exact("postcode","postcode", label="postcode_match")
    compare.exact("soc_sec_id", "soc_sec_id", label="soc_sec_id_match")
    compare.exact('state', 'state', label="state_match")
    
    return compare

def run_record_comparison(candidate_pairs, dfA, dfB):
    # Calculate comparison features for all candidate record pairs and display descriptive statistics.

    # Create candidate MultiIndex
    candidate_index = pd.MultiIndex.from_frame(candidate_pairs[["record_id_A", "record_id_B"]])

    # Define comparison features
    compare = create_comparison()

    # Calculate comparison features
    comparison_features = compare.compute(candidate_index, dfA, dfB)

    # Display comparison results
    record_comparison = comparison_features.reset_index()

    # Exact feature statistics
    exact_features = [
        "phonetic_given_name_match",
        "phonetic_surname_match",
        "street_number_match",
        "dob_match",
        "postcode_match",
        "soc_sec_id_match",
        "state_match"
    ]

    print("\nExact comparison statistics:")

    for column in exact_features:
        print(
            column,
            comparison_features[column].sum(),
            round(comparison_features[column].mean() * 100, 2),
            "%"
        )

    # String similarity statistics
    string_features = [
        "given_name_match",
        "surname_match",
        "suburb_match",
        "address_1_match",
        "address_2_match"
    ]

    print("\nString similarity statistics:")

    for column in string_features:
        print(f"\n{column}")
        print(comparison_features[column].describe())

    return comparison_features