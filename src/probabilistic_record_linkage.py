import pandas as pd
import recordlinkage

# Fellegi–Sunter method: Features are independent. Treasure is 0.8.
def probabilistic_matching(
    comparison_features,
    binarize=0.8,
    max_iter=100,
    atol=0.0001
):
    # Exclude features with many missing values and redundant phonetic features
    excluded_features = [
        "address_2_match",
        "phonetic_given_name_match",
        "phonetic_surname_match"
    ]
    model_features = comparison_features.drop(columns=excluded_features)

    # Initialize ECM classifier
    ecm = recordlinkage.ECMClassifier(
        init="jaro",
        binarize=binarize,
        max_iter=max_iter,
        atol=atol
    )

    # Fit model and predict matches
    ecm.fit(model_features)

    # Get predicted matches
    predictions = ecm.predict(model_features)

    # Calculate match probabilities
    probabilities = ecm.prob(model_features)

    # Create results table
    results = model_features.copy()

    results["match_probability"] = probabilities

    results["predicted_match"] = (
    results.index.isin(predictions).astype(int)
    )
    return results, ecm


def find_unmatched_records(dfA, dfB, record_comparison):
    # Find unmached records for the manual check
    
    # Get predicted matching pairs
    matched_pairs = record_comparison[
        record_comparison["predicted_match"] == 1
    ].index

    # Get IDs of matched records
    matched_ids_A = matched_pairs.get_level_values(0).unique()
    matched_ids_B = matched_pairs.get_level_values(1).unique()

    # Find records without any match
    unmatched_A = dfA.loc[
        ~dfA.index.isin(matched_ids_A)
    ].copy()

    unmatched_B = dfB.loc[
        ~dfB.index.isin(matched_ids_B)
    ].copy()

    print("Unmatched records in Dataset A:", len(unmatched_A))
    print("Unmatched records in Dataset B:", len(unmatched_B))

    return unmatched_A, unmatched_B


