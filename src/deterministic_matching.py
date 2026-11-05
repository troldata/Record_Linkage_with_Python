import pandas as pd

def exact_matching(dfA, dfB, match_columns):
    # Identify record pairs with exact matches in the specified columns.
    matches = dfA.merge(
        dfB,
        on=match_columns,
        how="inner",
        suffixes=("_A", "_B")
    )
    return matches

def run_deterministic_matching(dfA_clean, dfB_clean):
    
    # Apply three deterministic matching rules, combine unique pairs and identify unmatched records.
    
    # 1. Matching by Social Security ID and surname
    matches_soc_sec_id_surname = exact_matching(
        dfA_clean,
        dfB_clean,
        ["soc_sec_id", "surname"]
    )

    # 2. Matching by first name, surname and date of birth
    matches_name_dob = exact_matching(
        dfA_clean,
        dfB_clean,
        ["given_name", "surname", "date_of_birth"]
    )

    # 3. Matching by surname, date of birth and postcode
    matches_surname_dob_postcode = exact_matching(
        dfA_clean,
        dfB_clean,
        ["surname", "date_of_birth", "postcode"]
    )

    # Display results of individual matching rules
    print("\nIndividual matching results:")
    print("SSN + surname:", len(matches_soc_sec_id_surname))
    print("First name + surname + DOB:", len(matches_name_dob))
    print("Surname + DOB + postcode:", len(matches_surname_dob_postcode))

    # 4. Combine all matching results
    all_matches = pd.concat(
        [
            matches_soc_sec_id_surname,
            matches_name_dob,
            matches_surname_dob_postcode
        ],
        ignore_index=True
    )

    # Remove duplicate record pairs
    all_matches = all_matches.drop_duplicates(subset=["record_id_A", "record_id_B"])

    # Set a MultiIndex using record identifiers
    all_matches = all_matches.set_index(["record_id_A", "record_id_B"], drop=False)

    print("\nCombined deterministic matches:")
    print("Number of unique pairs:", len(all_matches))

    # 5. Identify records already matched
    matched_ids_A = all_matches.index.get_level_values("record_id_A").unique()
    matched_ids_B = all_matches.index.get_level_values("record_id_B").unique()

    # 6. Identify remaining records
    dfA_remain = dfA_clean[
        ~dfA_clean.index.isin(matched_ids_A)
    ].copy()

    dfB_remain = dfB_clean[
        ~dfB_clean.index.isin(matched_ids_B)
    ].copy()

    #print("\nRemaining records:")
    #print("Dataset A:", len(dfA_remain))
    #print("Dataset B:", len(dfB_remain))

    # 7. Check for one-to-many / many-to-one matches
    duplicate_A = all_matches.index.get_level_values(
        "record_id_A"
    ).duplicated().sum()

    duplicate_B = all_matches.index.get_level_values(
        "record_id_B"
    ).duplicated().sum()

    return all_matches, dfA_remain, dfB_remain
# Result: 3330 rows 