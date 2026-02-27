import pandas as pd
import numpy as np
import glob
import os
import re
from scipy.stats import wilcoxon

# --- CONFIGURATION ---
FILE_PATTERN = "results_11/*.txt"
OUTPUT_TEX_FILE = "complex_evaluation_summary_3.tex"
DATASETS_ROOT = "datasets"

DATASET_DIR_MAP = {
    'UASpeech': 'uaspeech',
    'NeuroVoz': 'neurovoz',
    'EasyCall': 'easycall',
    'COPAS': 'copas',
    'TORGO': 'torgo',
    'YT': 'youtube',
}
TYPE_DIR_MAP = {'Word': 'word', 'Utterance': 'utterances'}
COND_DIR_MAP = {'PB': 'balanced', 'PU': 'unbalanced', 'ALL': 'all'}

# --- METRIC DEFINITIONS ---
METRIC_ROW_MAP = {
    'speech_rate': 'Speech Rate',
    'speech_rate_fa': 'Speech Rate (FA)',
    'praat_speech_rate': 'Praat Speech Rate',
    'praat_speech_rate_fa': 'Speech Rate',
    'cpp': 'CPP',
    'cpp_fa': 'CPP',
    'std_pitch': 'Std Pitch',
    'std_pitch_fa': 'Std Pitch',
    'wada_snr': 'WADA SNR',
    'artp_old': 'Confidence',
    'age': 'Age',
    'spk2age': 'Age',
    'vsa': 'VSA',
    'vsa_fa': 'VSA (FA)',
    'double_asr': 'Double ASR',
    'artp_double_asr': 'DArtP',
    'per': 'PER (SEM)',
    'dper': 'PER (Phone)',
    'artp': 'ArtP',
    'p_estoi_control': 'P-ESTOI Control',
    'p_estoi_all': 'P-ESTOI All',
    'p_estoi_fa_control': 'P-ESTOI FA Control',
    'p_estoi_fa_all': 'P-ESTOI',
    'nad_control': 'NAD Control',
    'nad_all': 'NAD All',
    'nad_fa_control': 'NAD FA Control',
    'nad_fa_all': 'NAD'
}

# Multi-lingual support, Explainability columns per metric key
# \cmark* = limited/requires adaptation
METRIC_MULTI_EXPL = {
    'praat_speech_rate_fa': (r'\cmark',  r'\cmark'),
    'cpp_fa':               (r'\cmark',  r'\cmark'),
    'std_pitch_fa':         (r'\cmark',  r'\cmark'),
    'vsa':                  (r'\cmark*', r'\cmark'),
    'double_asr':           (r'\xmark',  r'\cmark'),
    'artp_double_asr':      (r'\xmark',  r'\cmark'),
    'artp_old':             (r'\xmark',  r'\cmark'),
    'per':                  (r'\xmark',  r'\cmark'),
    'dper':                 (r'\cmark*', r'\cmark'),
    'artp':                 (r'\cmark*', r'\cmark'),
    'p_estoi_fa_all':       (r'\cmark',  r'\xmark'),
    'nad_fa_all':           (r'\cmark',  r'\xmark'),
}

# Added Praat metrics here so they are considered for "Best Reference-Free" (underlining)
REF_FREE_KEYS = [
    'praat_speech_rate_fa', 'cpp_fa', 'std_pitch_fa', 'vsa', 'double_asr', 
    'artp_double_asr', 'artp_old'
]

# --- STATISTICAL GROUPS ---
GROUPS = {
    'Reference-Free (Signal)': [
        'speech_rate', 'speech_rate_fa', 
        'praat_speech_rate', 'praat_speech_rate_fa',
        'cpp', 'cpp_fa', 
        'std_pitch', 'std_pitch_fa', 
        'wada_snr', 'vsa', 'vsa_fa'
    ], 
    'Reference-Free (Model)': ['double_asr', 'artp_double_asr', 'artp_old'], 
    'Reference-Text': ['per', 'dper', 'artp'],
    'Reference-Audio': ['p_estoi_fa_all', 'nad_fa_all']
}

def count_lines(filepath):
    try:
        with open(filepath) as f:
            return sum(1 for line in f if line.strip())
    except FileNotFoundError:
        return None


def get_dataset_stats(datasets_root):
    """Return dict (dataset, type, cond, split) -> {'spk': int|None, 'utt': int|None}."""
    stats = {}
    for col_def in get_table_column_order():
        dataset, dtype, cond = col_def
        if dataset == 'YT':
            base = os.path.join(datasets_root, 'youtube')
            stats[(dataset, dtype, cond, 'pathological')] = {
                'spk': count_lines(os.path.join(base, 'spk2score')),
                'utt': count_lines(os.path.join(base, 'text')),
            }
            stats[(dataset, dtype, cond, 'control')] = {'spk': None, 'utt': None}
            continue
        d_dir = DATASET_DIR_MAP[dataset]
        t_dir = TYPE_DIR_MAP[dtype]
        c_dir = COND_DIR_MAP[cond]
        for split in ('pathological', 'control'):
            base = os.path.join(datasets_root, d_dir, split, t_dir, c_dir)
            stats[(dataset, dtype, cond, split)] = {
                'spk': count_lines(os.path.join(base, 'spk2score')),
                'utt': count_lines(os.path.join(base, 'text')),
            }
    return stats


def parse_txt_file(filepath):
    filename = os.path.basename(filepath).lower()
    
    # --- 1. DETECT METADATA ---
    dataset = 'Unknown'
    if 'uaspeech' in filename: dataset = 'UASpeech'
    elif 'neurovoz' in filename: dataset = 'NeuroVoz'
    elif 'easycall' in filename: dataset = 'EasyCall'
    elif 'copas' in filename: dataset = 'COPAS'
    elif 'torgo' in filename: dataset = 'TORGO'
    elif 'youtube' in filename or 'yt' in filename: dataset = 'YT'

    dtype = 'Word'
    if 'utterance' in filename or 'sentence' in filename:
        dtype = 'Utterance'
    
    # Condition Detection
    if 'unbalanced' in filename:
        cond = 'PU'
    elif 'balanced' in filename:
        cond = 'PB'
    elif 'all' in filename and dataset != 'EasyCall': 
        cond = 'ALL'
    elif 'all' in filename and dataset == 'EasyCall':
        if '_all' in filename or 'all_' in filename:
            cond = 'ALL'
        else:
            cond = 'Unknown' 
    else:
        cond = 'PU'

    if dataset == 'YT':
        dtype = 'Utterance'
        cond = 'ALL'

    # --- 2. PARSE CONTENT ---
    with open(filepath, 'r') as f:
        lines = f.read().splitlines()

    data = []
    in_table = False
    
    for line in lines:
        if "--- Evaluation Summary ---" in line:
            in_table = True
            continue
        if not in_table or "---" in line or "Metric" in line: continue
        if not line.strip().startswith("|"): continue

        parts = [p.strip() for p in line.split("|")]
        if len(parts) < 3: continue

        metric_raw = parts[1].strip()
        val_str = parts[2].strip()

        # Regex Extraction
        match = re.search(r'vs\s+([^\)]+)', metric_raw, re.IGNORECASE)
        if match:
            metric_key = match.group(1).strip().lower()
        else:
            metric_key = metric_raw.replace("PCC", "").replace("(", "").replace(")", "").strip().lower()

        if metric_key == 'spk2age':
            metric_key = 'age'

        try:
            val = float(val_str)
        except ValueError:
            val = np.nan

        data.append({
            "MetricKey": metric_key,
            "Value": val,
            "Dataset": dataset,
            "Type": dtype,
            "Condition": cond
        })
    return data

def align_signs(df):
    if df.empty: return df
    metrics = df['MetricKey'].unique()
    for m in metrics:
        if m in ['per', 'dper', 'nad_control', 'nad_all', 'nad_fa_control', 'nad_fa_all']: 
            mask = df['MetricKey'] == m
            df.loc[mask, 'Value'] = df.loc[mask, 'Value'].abs()
        elif m == 'double_asr':
            mask = df['MetricKey'] == m
            df.loc[mask, 'Value'] = df.loc[mask, 'Value'] * -1
        else:
            m_data = df[df['MetricKey'] == m]['Value'].dropna()
            if len(m_data) > 0 and (m_data < 0).all():
                 mask = df['MetricKey'] == m
                 df.loc[mask, 'Value'] = df.loc[mask, 'Value'] * -1
    return df

def run_bidirectional_wilcoxon(x, y, name, label_x="X", label_y="Y"):
    """Helper to run and print Wilcoxon tests in both directions."""
    n = len(x)
    if n < 2:
        print(f"--- {name} (N={n}) ---")
        print("  Insufficient data.")
        return

    try:
        # Test 1: X > Y?
        stat_x, p_x = wilcoxon(x, y, alternative='greater')
        sig_x = '*' if p_x < 0.05 else ''
        
        # Test 2: Y > X?
        stat_y, p_y = wilcoxon(x, y, alternative='less')
        sig_y = '*' if p_y < 0.05 else ''
        
        print(f"--- {name} (N={n}) ---")
        print(f"  {label_x} > {label_y}: p={p_x:.4f} {sig_x}")
        print(f"  {label_y} > {label_x}: p={p_y:.4f} {sig_y}")
        
    except ValueError:
        print(f"--- {name} (N={n}) ---")
        print("  Failed (likely identical values or insufficient variance)")

def perform_rq2_test(df):
    """PB vs PU"""
    print("\n" + "="*50)
    print("   RQ2: PB vs PU (Balance Effect)")
    print("="*50)
    
    displayed_metrics = [
        'praat_speech_rate_fa', 'cpp_fa', 'std_pitch_fa', 'vsa', 
        'double_asr', 'artp_double_asr', 'artp_old', 'per', 'dper', 'artp', 
        'p_estoi_fa_all', 'nad_fa_all'
    ]

    # Pivot
    df_stat = df[df['Condition'].isin(['PB', 'PU'])].copy()
    df_stat = df_stat[df_stat['MetricKey'].isin(displayed_metrics)]
    pivot = df_stat.pivot_table(
        index=['MetricKey', 'Dataset', 'Type'],
        columns='Condition',
        values='Value'
    ).dropna() 

    if pivot.empty:
        print("No paired PB/PU data found.")
        return

    run_bidirectional_wilcoxon(pivot['PU'], pivot['PB'], "GLOBAL", "PU", "PB")

    for group_name, metric_list in GROUPS.items():
        g_pivot = pivot[pivot.index.get_level_values('MetricKey').isin(metric_list)]
        if not g_pivot.empty:
            run_bidirectional_wilcoxon(g_pivot['PU'], g_pivot['PB'], group_name, "PU", "PB")

def perform_rq3_test(df):
    """Word vs Utterance"""
    print("\n" + "="*50)
    print("   RQ3: Word vs Utterance (Task Effect)")
    print("="*50)
    
    displayed_metrics = [
        'praat_speech_rate_fa', 'cpp_fa', 'std_pitch_fa', 'vsa', 
        'double_asr', 'artp_double_asr', 'artp_old', 'per', 'dper', 'artp', 
        'p_estoi_fa_all', 'nad_fa_all'
    ]

    # --- 1. Identify Intersection Datasets ---
    datasets_word = set(df[df['Type'] == 'Word']['Dataset'].unique())
    datasets_utt = set(df[df['Type'] == 'Utterance']['Dataset'].unique())
    common_datasets = list(datasets_word.intersection(datasets_utt))
    
    print(f"Filtering to datasets with BOTH Word and Utterance tasks: {common_datasets}")
    
    # Filter DataFrame
    df_stat = df[df['Dataset'].isin(common_datasets)].copy()
    df_stat = df_stat[df_stat['MetricKey'].isin(displayed_metrics)]
    
    # --- 2. Pivot ---
    pivot = df_stat.pivot_table(
        index=['MetricKey', 'Dataset', 'Condition'],
        columns='Type',
        values='Value'
    ).dropna() 

    if pivot.empty:
        print("No paired Word/Utterance data found after intersection filter.")
        return

    # --- 3. Run Tests ---
    run_bidirectional_wilcoxon(pivot['Utterance'], pivot['Word'], "GLOBAL", "Utt", "Word")

    for group_name, metric_list in GROUPS.items():
        g_pivot = pivot[pivot.index.get_level_values('MetricKey').isin(metric_list)]
        if not g_pivot.empty:
            run_bidirectional_wilcoxon(g_pivot['Utterance'], g_pivot['Word'], group_name, "Utt", "Word")

def get_table_column_order():
    return [
        ('UASpeech', 'Word', 'PB'), ('UASpeech', 'Word', 'PU'),
        ('NeuroVoz', 'Utterance', 'PB'), ('NeuroVoz', 'Utterance', 'PU'),
        ('EasyCall', 'Word', 'PB'), ('EasyCall', 'Word', 'PU'),
        ('EasyCall', 'Utterance', 'PB'), ('EasyCall', 'Utterance', 'PU'),
        ('COPAS', 'Word', 'PB'), ('COPAS', 'Word', 'PU'), ('COPAS', 'Word', 'ALL'),
        ('COPAS', 'Utterance', 'PB'), ('COPAS', 'Utterance', 'PU'), ('COPAS', 'Utterance', 'ALL'),
        ('TORGO', 'Word', 'PB'), ('TORGO', 'Word', 'PU'),
        ('TORGO', 'Utterance', 'PB'), ('TORGO', 'Utterance', 'PU'),
        ('YT', 'Utterance', 'ALL')
    ]

def generate_latex(df, datasets_root=DATASETS_ROOT):
    target_tuples = get_table_column_order()
    stats = get_dataset_stats(datasets_root)

    pivot = df.pivot_table(
        index='MetricKey',
        columns=['Dataset', 'Type', 'Condition'],
        values='Value',
        aggfunc='first'
    )
    target_index = pd.MultiIndex.from_tuples(target_tuples, names=['Dataset', 'Type', 'Condition'])
    pivot = pivot.reindex(columns=target_index)

    col_max_global = {}
    col_max_rf = {}
    for col_def in target_tuples:
        col_series = pivot[col_def].dropna().round(2)
        col_max_global[col_def] = col_series.max() if not col_series.empty else -999
        rf_series = col_series[col_series.index.isin(REF_FREE_KEYS)]
        col_max_rf[col_def] = rf_series.max() if not rf_series.empty else -999

    # Metric + Multi + Expl + data cols + Avg
    N_COLS = 1 + 2 + len(target_tuples) + 1

    def fmt_stat(val):
        return '--' if val is None else str(val)

    def stat_row(label, split, key):
        cells = ' & '.join(fmt_stat(stats.get((d, t, c, split), {}).get(key)) for d, t, c in target_tuples)
        return f"        {label} & & & {cells} & \\\\"

    latex_rows = []

    # --- Dataset Information ---
    latex_rows.append(f"        \\multicolumn{{{N_COLS}}}{{l}}{{\\textit{{Dataset Information}}}} \\\\")
    latex_rows.append(
        r"        Language & & & \multicolumn{2}{c|}{English} & \multicolumn{2}{c|}{Spanish}"
        r" & \multicolumn{4}{c|}{Italian} & \multicolumn{6}{c|}{Dutch}"
        r" & \multicolumn{4}{c|}{English} & English & \\"
    )
    latex_rows.append(
        r"        Disorder & & & \multicolumn{2}{c|}{Dysarthria} & \multicolumn{2}{c|}{Parkinson's}"
        r" & \multicolumn{4}{c|}{Dysarthria} & \multicolumn{6}{c|}{Variety of pathologies}"
        r" & \multicolumn{4}{c|}{Dysarthria} & OC & \\"
    )
    latex_rows.append(r"        \midrule")

    # --- Pathological Statistics ---
    latex_rows.append(f"        \\multicolumn{{{N_COLS}}}{{l}}{{\\textit{{Statistics: Pathological}}}} \\\\")
    latex_rows.append(stat_row(r"\# Spk", 'pathological', 'spk'))
    latex_rows.append(stat_row(r"\# Utt", 'pathological', 'utt'))
    latex_rows.append(r"        \midrule")

    # --- Control Statistics ---
    latex_rows.append(f"        \\multicolumn{{{N_COLS}}}{{l}}{{\\textit{{Statistics: Control}}}} \\\\")
    latex_rows.append(stat_row(r"\# Spk", 'control', 'spk'))
    latex_rows.append(stat_row(r"\# Utt", 'control', 'utt'))
    latex_rows.append(r"        \midrule")
    latex_rows.append(r"        \midrule")

    # --- Metric Groups ---
    groups = [
        ("Reference-Free (Signal)", ['praat_speech_rate_fa', 'cpp_fa', 'std_pitch_fa']),
        ("Reference-Free (Speaker)", ['vsa']),
        ("Reference-Free (Model)", ['double_asr', 'artp_double_asr', 'artp_old']),
        ("Reference-Text", ['per', 'dper', 'artp']),
        ("Reference-Audio (Parallel)", ['p_estoi_fa_all', 'nad_fa_all']),
    ]

    all_metrics = [m for _, metrics in groups for m in metrics]

    # Pass 1: compute per-metric averages (rounded)
    metric_avg = {}
    for m_key in all_metrics:
        if m_key not in pivot.index:
            metric_avg[m_key] = np.nan
            continue
        vals = [round(pivot.loc[m_key, c], 2) for c in target_tuples if not pd.isna(pivot.loc[m_key, c])]
        metric_avg[m_key] = round(np.mean(vals), 2) if vals else np.nan

    avg_max_global = max((v for v in metric_avg.values() if not np.isnan(v)), default=-999)
    avg_max_rf = max((v for k, v in metric_avg.items() if k in REF_FREE_KEYS and not np.isnan(v)), default=-999)

    def fmt_cell(cell_tex, is_global_best, is_rf_best):
        if is_global_best and is_rf_best:
            return f"\\textbf{{\\underline{{{cell_tex}}}}}"
        if is_global_best:
            return f"\\textbf{{{cell_tex}}}"
        if is_rf_best:
            return f"\\underline{{{cell_tex}}}"
        return cell_tex

    for group_name, metrics in groups:
        latex_rows.append(f"        \\multicolumn{{{N_COLS}}}{{l}}{{\\textit{{{group_name}}}}} \\\\")
        for m_key in metrics:
            display_name = METRIC_ROW_MAP.get(m_key, m_key)
            multi, expl = METRIC_MULTI_EXPL.get(m_key, ('', ''))
            row_str = f"        {display_name} & {multi} & {expl}"
            for col_def in target_tuples:
                val = pivot.loc[m_key, col_def] if m_key in pivot.index else np.nan
                cell_tex = "--"
                if not pd.isna(val):
                    rounded = round(val, 2)
                    cell_tex = fmt_cell(
                        f"{val:.2f}",
                        rounded == col_max_global[col_def] and col_max_global[col_def] != -999,
                        m_key in REF_FREE_KEYS and rounded == col_max_rf[col_def] and col_max_rf[col_def] != -999,
                    )
                row_str += f" & {cell_tex}"
            avg = metric_avg[m_key]
            if np.isnan(avg):
                avg_tex = "--"
            else:
                avg_tex = fmt_cell(
                    f"{avg:.2f}",
                    avg == avg_max_global and avg_max_global != -999,
                    m_key in REF_FREE_KEYS and avg == avg_max_rf and avg_max_rf != -999,
                )
            row_str += f" & {avg_tex} \\\\"
            latex_rows.append(row_str)
        latex_rows.append(r"        \midrule")

    header = r"""\begin{table*}[t]
    \centering
    \caption{Speaker-level Pearson Correlation Coefficient (PCC). \textbf{Multi}: Multilingual Support (\cmark*: Limited/Requires adaptation), \textbf{Expl}: Explainable/Interpretable. \textbf{MC}: Matched Content, \textbf{EX}: Extended, \textbf{Full}: Combined (COPAS only). \textbf{Bold}: Best overall. \underline{Underline}: Best Reference-Free. \textbf{OC}: Oral Cancer. Spk: number of speakers. Utt: number of utterances.}
    \label{tab:main_results}
    \resizebox{\textwidth}{!}{%
    \begin{tabular}{l|cc|cc|cc|cc|cc|ccc|ccc|cc|cc|c|c}
        \toprule
        & & & \multicolumn{2}{c|}{\textbf{UASpeech \cite{kim08c_interspeech}}} & \multicolumn{2}{c|}{\textbf{NeuroVoz \cite{mendes2024neurovoz}}} & \multicolumn{2}{c|}{\textbf{EasyCall \cite{turrisi21_interspeech}}} & \multicolumn{2}{c|}{\textbf{EasyCall \cite{turrisi21_interspeech}}} & \multicolumn{3}{c|}{\textbf{COPAS \cite{van2009dutch}}} & \multicolumn{3}{c|}{\textbf{COPAS \cite{van2009dutch}}} & \multicolumn{2}{c|}{\textbf{TORGO \cite{rudzicz2012torgo}}} & \multicolumn{2}{c|}{\textbf{TORGO \cite{rudzicz2012torgo}}} & \textbf{YT \cite{halpern25_interspeech}} & \\
        & & & \multicolumn{2}{c|}{\textit{Word}} & \multicolumn{2}{c|}{\textit{Sentence}} & \multicolumn{2}{c|}{\textit{Word}} & \multicolumn{2}{c|}{\textit{Sentence}} & \multicolumn{3}{c|}{\textit{Word}} & \multicolumn{3}{c|}{\textit{Sentence}} & \multicolumn{2}{c|}{\textit{Word}} & \multicolumn{2}{c|}{\textit{Sentence}} & \textit{Sent} & \\
        \cmidrule(lr){4-5} \cmidrule(lr){6-7} \cmidrule(lr){8-9} \cmidrule(lr){10-11} \cmidrule(lr){12-14} \cmidrule(lr){15-17} \cmidrule(lr){18-19} \cmidrule(lr){20-21} \cmidrule(lr){22-22}
        \textbf{Metric} & \textbf{Multi} & \textbf{Expl} & \textbf{MC} & \textbf{EX} & \textbf{MC} & \textbf{EX} & \textbf{MC} & \textbf{EX} & \textbf{MC} & \textbf{EX} & \textbf{MC} & \textbf{EX} & \textbf{Full} & \textbf{MC} & \textbf{EX} & \textbf{Full} & \textbf{MC} & \textbf{EX} & \textbf{MC} & \textbf{EX} & \textbf{Full} & \textbf{Avg} \\
        \midrule"""
    footer = r"""        \bottomrule
    \end{tabular}%
    }
\end{table*}"""

    with open(OUTPUT_TEX_FILE, 'w') as f:
        f.write(header + "\n" + "\n".join(latex_rows) + "\n" + footer)
    print(f"LaTeX table saved to {OUTPUT_TEX_FILE}")

def print_wada_snr_results(df):
    """Print WADA-SNR PCC results as a simple protocol/value table."""
    target_tuples = get_table_column_order()

    snr_df = df[df['MetricKey'] == 'wada_snr'].copy()
    if snr_df.empty:
        print("\nNo WADA-SNR data found.")
        return

    pivot = snr_df.pivot_table(
        index='MetricKey',
        columns=['Dataset', 'Type', 'Condition'],
        values='Value',
        aggfunc='first'
    )
    target_index = pd.MultiIndex.from_tuples(target_tuples, names=['Dataset', 'Type', 'Condition'])
    pivot = pivot.reindex(columns=target_index)

    print("\n" + "="*40)
    print("   WADA-SNR (PCC)")
    print("="*40)
    print(f"  {'Protocol':<28}  {'PCC':>6}")
    print("-" * 40)

    vals = []
    for d, t, c in target_tuples:
        protocol = f"{d} {t} {c}"
        v = pivot.loc['wada_snr', (d, t, c)] if 'wada_snr' in pivot.index else np.nan
        if pd.isna(v):
            print(f"  {protocol:<28}  {'--':>6}")
        else:
            vals.append(round(v, 2))
            print(f"  {protocol:<28}  {v:>6.2f}")

    print("-" * 40)
    avg = round(np.mean(vals), 2) if vals else float('nan')
    print(f"  {'Average':<28}  {avg:>6.2f}")


def main():
    files = glob.glob(FILE_PATTERN)
    if not files:
        print(f"No files found matching: {FILE_PATTERN}")
        return

    print(f"Found {len(files)} files. Parsing...")
    all_data = []
    for f in files:
        all_data.extend(parse_txt_file(f))

    if not all_data:
        print("No valid data extracted.")
        return

    df = pd.DataFrame(all_data)
    df = align_signs(df)

    # 1. Generate Latex
    generate_latex(df)

    # 2. Print WADA-SNR separately
    print_wada_snr_results(df)

    # 3. Run RQ2
    perform_rq2_test(df)

    # 4. Run RQ3 (Filtered to common datasets)
    perform_rq3_test(df)

if __name__ == "__main__":
    main()