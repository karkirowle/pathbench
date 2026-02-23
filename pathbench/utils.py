import numpy as np
def normalise_signal(x, method):
    """
    Normalise signal to RMS or zero mean

    :param x: signal
    :param method: Method to use for normalisation
    :return:

    """

    assert method in ['RMS', 'zero_mean'], 'normalisation method should be RMS or zero_mean'
    if method == 'RMS':
        max_abs = np.max(np.abs(x))
        #if max_abs == 0:
        #    return x
        x = 0.8 * (x / max_abs)
        return x
    elif method == 'zero_mean':
        std = np.std(x)
        #if std == 0:
        #    return x
        x = (x - np.mean(x)) / std
        return x
    else:
        raise Exception('normalisation method should be RMS or zero_mean')

def moving_average_filtering(x, N=5):
    return np.apply_along_axis(lambda m: np.convolve(m, np.ones((N,))/N, mode='valid'), axis = 1, arr = x)


def write_correlation_table(output_file, results: dict, metrics: list):
    """Writes a markdown-style summary table of PCC results.

    Args:
        output_file: file-like object
        results: dataset_dir -> {'pcc_<metric>': float}
        metrics: ordered list of metric names (without 'pcc_' prefix)
    """
    datasets = list(results.keys())
    if not datasets:
        output_file.write("No results to summarize.\n")
        return

    output_file.write("\n--- Evaluation Summary ---\n")
    header = "| Metric |" + "".join(f" {d} |" for d in datasets)
    output_file.write(header + "\n")
    output_file.write("|" + "---|" * (len(datasets) + 1) + "\n")

    for metric in metrics:
        row = f"| {metric} |"
        for dataset in datasets:
            pcc = results[dataset].get(f"pcc_{metric}")
            row += f" {pcc:.4f} |" if pcc is not None else " N/A |"
        output_file.write(row + "\n")
