import pandas as pd
import recordlinkage

def evaluate_hybrid_matching(
    dfA,
    dfB,
    true_links,
    deterministic_matches,
    probabilistic_comparison,
    remaining_A,
    remaining_B
):

    # Get deterministic matches
    deterministic_pairs = deterministic_matches.index

    # Get candidate pairs from probabilistic stage
    probabilistic_candidates = probabilistic_comparison.index

    # Get ECM predicted matches
    ecm_matches = probabilistic_comparison[
        probabilistic_comparison["predicted_match"] == 1
    ].index

    # Combine final matches from both stages
    final_matches = deterministic_pairs.append(
        ecm_matches
    ).unique()

    # Evaluate Blocking on remaining records only

    remaining_true_links = true_links.difference(
        deterministic_pairs
    )

    candidate_true_links = remaining_true_links.intersection(
        probabilistic_candidates
    )

    pairs_completeness = (
        len(candidate_true_links) / len(remaining_true_links)
        if len(remaining_true_links) > 0
        else 0
    )

    total_remaining_pairs = len(remaining_A) * len(remaining_B)

    blocking_reduction_ratio = (
        1 - len(probabilistic_candidates) / total_remaining_pairs
        if total_remaining_pairs > 0
        else 0
    )

    # Evaluate final hybrid matching
    precision = recordlinkage.precision(
        true_links, final_matches
    )

    recall = recordlinkage.recall(
        true_links, final_matches
    )

    f_score = (
        2 * precision * recall / (precision + recall)
        if precision + recall > 0
        else 0
    )

    # Print results
    print("\nInitial data")
    
    print(f"Dataset A: {len(dfA):,} records")
    print(f"Dataset B: {len(dfB):,} records")
    print(f"Initial possible pairs: {len(dfA) * len(dfB):,}")

    print("\nStage 1 — Deterministic Matching:")
    print(f"Matched pairs: {len(deterministic_pairs):,}")
    print(f"Remaining records in A: {len(remaining_A):,}")
    print(f"Remaining records in B: {len(remaining_B):,}")
    
    print("\nStage 2 — Probabilistic Matching:")
    print(f"Total possible remaining pairs: {total_remaining_pairs:,}")
    print(f"Candidate pairs after Blocking: {len(probabilistic_candidates):,}")
    print(f"ECM predicted matches: {len(ecm_matches):,}")
    print("True links captured by Blocking:", len(candidate_true_links))
    
    print(f"\nBlocking Pairs Completeness: {pairs_completeness:.2%}")
    print(f"Blocking Reduction Ratio: {blocking_reduction_ratio:.2%}")
    
    print("\nFinal Hybrid Matching Evaluation")
    print("Final matches:", len(final_matches))
    print(f"Precision: {precision:.2%}")
    print(f"Recall: {recall:.2%}")
    print(f"F-score: {f_score:.2%}")

    return {
        "deterministic_matches": len(deterministic_pairs),
        "remaining_records_A": len(remaining_A),
        "remaining_records_B": len(remaining_B),
        "probabilistic_candidates": len(probabilistic_candidates),
        "ecm_matches": len(ecm_matches),
        "final_matches": len(final_matches),
        "pairs_completeness": pairs_completeness,
        "blocking_reduction_ratio": blocking_reduction_ratio,
        "precision": precision,
        "recall": recall,
        "f_score": f_score,
        "final_pairs": final_matches
    }

"""
Initial data
Dataset A: 5,000 records
Dataset B: 5,000 records
Initial possible pairs: 25,000,000

Stage 1 — Deterministic Matching:
Matched pairs: 3,330
Remaining records in A: 1,670
Remaining records in B: 1,670

Stage 2 — Probabilistic Matching:
Total possible remaining pairs: 2,788,900
Candidate pairs after Blocking: 7,388
ECM predicted matches: 1,663
True links captured by Blocking: 1663

Blocking Pairs Completeness: 99.58%
Blocking Reduction Ratio: 99.74%

Final Hybrid Matching Evaluation
Final matches: 4993
Precision: 100.00%
Recall: 99.86%
F-score: 99.93%
"""

