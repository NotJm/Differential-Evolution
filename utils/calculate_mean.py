import numpy as np

def calculate_mean_boxes_constraints(*results):
    try:
        max_lengths = [max(len(seq) for seq in result) for result in results]
        max_length = max(max_lengths)

        normalized_results = [np.array([seq + [np.nan] * (max_length - len(seq)) for seq in result]) for result in results]

        mean_results = [np.nanmean(normalized_result, axis=0) for normalized_result in normalized_results]

        return tuple(mean_results)
    except RuntimeWarning:
        pass