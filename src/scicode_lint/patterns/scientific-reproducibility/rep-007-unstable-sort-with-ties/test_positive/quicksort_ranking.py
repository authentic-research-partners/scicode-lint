import numpy as np


def rank_candidates(exam_scores, candidate_ids):
    order = np.argsort(-exam_scores)
    return [candidate_ids[i] for i in order]


def assign_priority(severity_levels, patient_ids):
    idx = np.argsort(severity_levels)
    return patient_ids[idx]
