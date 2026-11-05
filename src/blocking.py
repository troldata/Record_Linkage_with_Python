import pandas as pd

def blocking(dfA, dfB, blocking_columns):
    """
    Generate candidate record pairs based on identical blocking keys.
    Return only record identifiers.
    """

    candidate_pairs = dfA.merge(
        dfB,
        on=blocking_columns,
        how="inner",
        suffixes=("_A", "_B")
    )

    # Keep only record identifiers
    candidate_pairs = candidate_pairs[
        ["record_id_A", "record_id_B"]
    ].drop_duplicates()

    return candidate_pairs


def reduction_ratio(total_pairs, candidate_pairs):
    """Calculate the proportion of eliminated record pairs."""

    if total_pairs == 0:
        return 0

    return 1 - candidate_pairs / total_pairs


def run_blocking(dfA, dfB):
    """
    Apply multiple blocking rules, evaluate each rule,
    combine candidate pairs and calculate the overall reduction ratio.
    """

    total_possible_pairs = len(dfA) * len(dfB)

    # Define blocking rules
    blocking_rules = {
        "Social Security ID": ["soc_sec_id"],
        "Date of Birth": ["date_of_birth"],
        "Postcode": ["postcode"],
        "Phonetic Surname + DOB": [
            "phonetic_surname",
            "date_of_birth"
        ]
    }

    candidate_sets = []
    blocking_evaluation = []

    # Apply individual blocking rules
    for method_name, blocking_columns in blocking_rules.items():

        candidates = blocking(dfA, dfB, blocking_columns)
        candidate_sets.append(candidates)
        number_of_pairs = len(candidates)

        rr = reduction_ratio(
            total_possible_pairs,
            number_of_pairs
        )

        blocking_evaluation.append({
            "Blocking method": method_name,
            "Candidate pairs": number_of_pairs,
            "Reduction Ratio (%)": round(rr * 100, 2)
        })

    # Create evaluation table
    blocking_results = pd.DataFrame(blocking_evaluation)

    # Combine candidate pairs from all blocking rules
    combined_candidates = pd.concat(
        candidate_sets,
        ignore_index=True
    )

    # Remove duplicate candidate pairs
    combined_candidates = combined_candidates.drop_duplicates(
        subset=["record_id_A", "record_id_B"]
    ).reset_index(drop=True)

    # Calculate overall reduction ratio
    combined_rr = reduction_ratio(
        total_possible_pairs,
        len(combined_candidates)
    )

    #print("\nIndividual Blocking Results:")
    #print(blocking_results.to_string(index=False))

    #print("\nCombined Blocking Results:")
    #print("Total possible pairs:", total_possible_pairs)
    #print("Unique candidate pairs:", len(combined_candidates))
    #print("Combined Reduction Ratio:", f"{combined_rr:.2%}")

    return combined_candidates, blocking_results
"""
The result of 99.78% demonstrates the high efficiency of reducing the search space. 
99.78% of all possible pairs were excluded from further consideration, 
while 0.22% (54639 candidate pairs) were retained for the Matching stage.

When Blocking was after deterministic matching fulfilled, 
Combined Reduction Ratio equal 99.74 %  and the number of candidate pairs 7388.
"""