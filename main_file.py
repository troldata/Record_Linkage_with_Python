from recordlinkage.datasets import load_febrl4

from src.data_quality import (
    dataset_info,
    basic_quality_report,
    check_dates,
    check_postcodes,
    check_street_numbers
)

from src.data_cleaning import (
    standardize_strings,
    clean_dates,
    clean_street_number,
    clean_postcode
)

from src.deterministic_matching import run_deterministic_matching
from src.phonetic_encoding import add_phonetic_features
from src.blocking import run_blocking
from src.record_comparison import run_record_comparison
from src.probabilistic_record_linkage import probabilistic_matching, find_unmatched_records
from results.result import evaluate_hybrid_matching


def main():
    # ==================================================
    # 1. Data loading
    # ==================================================
    print("\n1. Data Loading")
    dfA, dfB, true_links = load_febrl4(return_links=True)

    dfA = dfA.reset_index(names="record_id_A")
    dfB = dfB.reset_index(names="record_id_B")

    dfA = dfA.set_index("record_id_A", drop=False)
    dfB = dfB.set_index("record_id_B", drop=False)
    
    # ==================================================
    # 2. Data Quality Assesment
    # ==================================================

    print("\n2.Data Quality Assessment")
    dataset_info(dfA, dfB)
    report_A = basic_quality_report(dfA, "Dataset A")
    report_B = basic_quality_report(dfB, "Dataset B")

    print("Dataset A:", report_A.to_string(index=False))
    print("Dataset B:", report_B.to_string(index=False))

    print("\nDate validation:")
    print("Dataset A:", check_dates(dfA, "date_of_birth"))
    print("Dataset B:", check_dates(dfB, "date_of_birth"))

    print("\nPostcode validation:")
    print("Dataset A:", check_postcodes(dfA, "postcode"))
    print("Dataset B:", check_postcodes(dfB, "postcode"))

    print("\nStreet number validation:")
    print("Dataset A:", check_street_numbers(dfA, "street_number"))
    print("Dataset B:", check_street_numbers(dfB, "street_number"))
    
    # ==================================================
    # 3. Data Cleaning
    # ==================================================
    print("\n3. Data Cleaning")

    dfA_clean = dfA.copy()
    dfB_clean = dfB.copy()

    print("\nStandardizing strings.")
    dfA_clean = standardize_strings(dfA_clean)
    dfB_clean = standardize_strings(dfB_clean)

    print("\nCleaning dates.")
    dfA_clean = clean_dates(dfA_clean, "date_of_birth")
    dfB_clean = clean_dates(dfB_clean, "date_of_birth")

    print("\nCleaning street numbers.")
    dfA_clean = clean_street_number(dfA_clean)
    dfB_clean = clean_street_number(dfB_clean)

    print("\nCleaning postcodes.")
    dfA_clean = clean_postcode(dfA_clean)
    dfB_clean = clean_postcode(dfB_clean)

    print("\nData types after cleaning:")
    print("Dataset A:")
    dfA_clean.info()

    # ==================================================
    # 4. Deterministic Matching
    # ==================================================
    print("\n4. Deterministic Matching")
    
    all_matches, dfA_remain, dfB_remain = run_deterministic_matching(dfA_clean, dfB_clean)
    
    # ==================================================
    # 5. Phonetic Encoding
    # ==================================================
    print("\n5. Phonetic Encoding")
    
    dfA_remain = add_phonetic_features(dfA_remain)
    dfB_remain = add_phonetic_features(dfB_remain)
    
    # ==================================================
    # 6. Blocking
    # ==================================================
    print("\n6. Blocking")
    
    candidate_pairs, blocking_results = run_blocking(dfA_remain, dfB_remain)
    
    # ==================================================
    # 7. Record Comparison
    # ==================================================
    print("\n7. Record Comparison")
    
    record_comparison = run_record_comparison(candidate_pairs, dfA_remain, dfB_remain)
    
    # ==================================================
    # 8. Probalistic Record Linkage
    # ==================================================
    print("\n8. Probabilistic Record Linkage")
    
    record_comparison, ecm = probabilistic_matching(record_comparison)
    
    # Find unmached records for the manual check
    unmatched_A, unmatched_B = find_unmatched_records(dfA_remain, dfB_remain, record_comparison)
    print("Records for the manual check_Dataset_A:\n",unmatched_A)
    print("Records for the manual check_Dataset_B:\n", unmatched_B)  

    # ==================================================
    # 9. Results
    # ==================================================
    print("\n9. Hybrid Matching Evaluation")
    evaluation_results = evaluate_hybrid_matching(
        dfA,
        dfB,
        true_links=true_links,
        deterministic_matches=all_matches,
        probabilistic_comparison=record_comparison,
        remaining_A = dfA_remain,
        remaining_B = dfB_remain
    ) 
    print("\nPipeline completed successfully")
    
if __name__ == "__main__":
    main()