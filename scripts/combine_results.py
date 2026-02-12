import pandas as pd
import numpy as np
import glob
import os
import re
from scipy.stats import wilcoxon

# --- CONFIGURATION ---
FILE_PATTERN = "results_6/*.txt"
OUTPUT_TEX_FILE = "complex_evaluation_summary.tex"

# --- METRIC DEFINITIONS ---
METRIC_ROW_MAP = {
    'speech_rate': 'Speech Rate',
    'speech_rate_fa': 'Speech Rate (FA)',
    'praat_speech_rate': 'Praat Speech Rate',
    'praat_speech_rate_fa': 'Praat Speech Rate (FA)',
    'cpp': 'CPP',
    'cpp_fa': 'CPP (FA)',
    'std_pitch': 'Std Pitch',
    'std_pitch_fa': 'Std Pitch (FA)',
    'wada_snr': 'WADA SNR',
    'artp_old': 'Entropy',
    'age': 'Age',
    'spk2age': 'Age',
    'vsa': 'VSA',
    'vsa_fa': 'VSA (FA)',
    'double_asr': 'Double ASR',
    'artp_double_asr': 'ArtP (DASR)',
    'per': 'PER',
    'dper': 'DPER',
    'artp': 'ArtP',
    'p_estoi_control': 'P-ESTOI Control',
    'p_estoi_all': 'P-ESTOI All',
    'p_estoi_fa_control': 'P-ESTOI FA Control',
    'p_estoi_fa_all': 'P-ESTOI FA All',
    'nad_control': 'NAD Control',
    'nad_all': 'NAD All',
    'nad_fa_control': 'NAD FA Control',
    'nad_fa_all': 'NAD FA All'
}

# Added Praat metrics here so they are considered for "Best Reference-Free" (underlining)
REF_FREE_KEYS = [
    'speech_rate', 'speech_rate_fa',
    'praat_speech_rate', 'praat_speech_rate_fa',
    'cpp', 'cpp_fa',
    'f0_range', 
    'std_pitch', 'std_pitch_fa',
    'wada_snr', 
    'vsa', 'vsa_fa',
    'double_asr', 'artp_double_asr', 'artp_old', 'age', 'spk2age'
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
    'Reference-Audio': ['p_estoi_control', 'p_estoi_all', 'nad_control', 'nad_all']
}

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
    
    # Pivot
    df_stat = df[df['Condition'].isin(['PB', 'PU'])].copy()
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
    
    # --- 1. Identify Intersection Datasets ---
    datasets_word = set(df[df['Type'] == 'Word']['Dataset'].unique())
    datasets_utt = set(df[df['Type'] == 'Utterance']['Dataset'].unique())
    common_datasets = list(datasets_word.intersection(datasets_utt))
    
    print(f"Filtering to datasets with BOTH Word and Utterance tasks: {common_datasets}")
    
    # Filter DataFrame
    df_stat = df[df['Dataset'].isin(common_datasets)].copy()
    
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

def generate_latex(df):
    target_tuples = get_table_column_order()
    
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
        col_series = pivot[col_def]
        valid_vals = col_series.dropna()
        col_max_global[col_def] = valid_vals.max() if not valid_vals.empty else -999
        rf_series = col_series[col_series.index.isin(REF_FREE_KEYS)].dropna()
        col_max_rf[col_def] = rf_series.max() if not rf_series.empty else -999

    latex_rows = []
    
    # --- UPDATED TABLE GROUPS TO INCLUDE PRAAT METRICS ---
    groups = [
        ("Reference-Free (Signal)", [
            'speech_rate', 'speech_rate_fa', 
            'praat_speech_rate', 'praat_speech_rate_fa',
            'cpp', 'cpp_fa',
            'f0_range', 
            'std_pitch', 'std_pitch_fa', 
            'wada_snr', 
            'age'
        ]), 
        ("Reference-Free (Speaker)", ['vsa', 'vsa_fa']),
        ("Reference-Free (Model)", ['double_asr', 'artp_double_asr', 'artp_old']),
        ("Reference-Text", ['per', 'dper', 'artp']),
        ("Reference-Audio (Parallel)", [
            'p_estoi_control', 'p_estoi_all', 
            'p_estoi_fa_control', 'p_estoi_fa_all', 
            'nad_control', 'nad_all',
            'nad_fa_control', 'nad_fa_all'
        ])
    ]
    
    for group_name, metrics in groups:
        latex_rows.append(f"        \\multicolumn{{20}}{{l}}{{\\textit{{{group_name}}}}} \\\\")
        
        for m_key in metrics:
            display_name = METRIC_ROW_MAP.get(m_key, m_key)
            
            # --- Logic to handle indented Control/All rows ---
            if m_key in ['p_estoi_control', 'nad_control', 'p_estoi_fa_control', 'nad_fa_control']:
                # Clean name: "P-ESTOI Control" -> "P-ESTOI", "P-ESTOI FA Control" -> "P-ESTOI FA"
                base_name = display_name.replace(" Control", "").replace(" FA", " FA")
                latex_rows.append(f"        {base_name} &&&&&&&&&&&&&&&&&&& \\\\")
                display_name = "\\hspace{3mm} \\textit{Control}"
            elif m_key in ['p_estoi_all', 'nad_all', 'p_estoi_fa_all', 'nad_fa_all']:
                display_name = "\\hspace{3mm} \\textit{All}"
            
            row_str = f"        {display_name}"
            for col_def in target_tuples:
                val = pivot.loc[m_key, col_def] if m_key in pivot.index else np.nan
                cell_tex = "--"
                if not pd.isna(val):
                    cell_tex = f"{val:.2f}"
                    # Bold global max
                    if val == col_max_global[col_def] and val != -999:
                        cell_tex = f"\\textbf{{{cell_tex}}}"
                    # Underline ref-free max
                    elif (m_key in REF_FREE_KEYS) and (val == col_max_rf[col_def]) and (val != -999):
                        cell_tex = f"\\underline{{{cell_tex}}}"
                row_str += f" & {cell_tex}"
            row_str += " \\\\"
            latex_rows.append(row_str)

    header = r"""\begin{table*}[t]
    \centering
    \caption{Speaker-level Pearson Correlation Coefficient (PCC). \textbf{PB}: Phonetically Balanced, \textbf{PU}: Phonetically Unbalanced, \textbf{ALL}: Combined (COPAS only). \textbf{Bold}: Best overall. \underline{Underline}: Best Reference-Free.}
    \label{tab:main_results}
    \resizebox{\textwidth}{!}{%
    \begin{tabular}{l|cc|cc|cc|cc|ccc|ccc|cc|cc|c}
        \toprule
        & \multicolumn{2}{c|}{\textbf{UASpeech}} & \multicolumn{2}{c|}{\textbf{NeuroVoz}} & \multicolumn{2}{c|}{\textbf{EasyCall}} & \multicolumn{2}{c|}{\textbf{EasyCall}} & \multicolumn{3}{c|}{\textbf{COPAS}} & \multicolumn{3}{c|}{\textbf{COPAS}} & \multicolumn{2}{c|}{\textbf{TORGO}} & \multicolumn{2}{c|}{\textbf{TORGO}} & \textbf{YT} \\
        & \multicolumn{2}{c|}{\textit{Word}} & \multicolumn{2}{c|}{\textit{Utterance}} & \multicolumn{2}{c|}{\textit{Word}} & \multicolumn{2}{c|}{\textit{Utterance}} & \multicolumn{3}{c|}{\textit{Word}} & \multicolumn{3}{c|}{\textit{Utterance}} & \multicolumn{2}{c|}{\textit{Word}} & \multicolumn{2}{c|}{\textit{Utterance}} & \textit{Utt} \\
        \cmidrule(lr){2-3} \cmidrule(lr){4-5} \cmidrule(lr){6-7} \cmidrule(lr){8-9} \cmidrule(lr){10-12} \cmidrule(lr){13-15} \cmidrule(lr){16-17} \cmidrule(lr){18-19} \cmidrule(lr){20-20}
        \textbf{Metric} & \textbf{PB} & \textbf{PU} & \textbf{PB} & \textbf{PU} & \textbf{PB} & \textbf{PU} & \textbf{PB} & \textbf{PU} & \textbf{PB} & \textbf{PU} & \textbf{ALL} & \textbf{PB} & \textbf{PU} & \textbf{ALL} & \textbf{PB} & \textbf{PU} & \textbf{PB} & \textbf{PU} & \textbf{ALL} \\
        \midrule"""
    footer = r"""        \bottomrule
    \end{tabular}%
    }
\end{table*}"""

    with open(OUTPUT_TEX_FILE, 'w') as f:
        f.write(header + "\n" + "\n".join(latex_rows) + footer)
    print(f"LaTeX table saved to {OUTPUT_TEX_FILE}")

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
    
    # 2. Run RQ2
    perform_rq2_test(df)

    # 3. Run RQ3 (Filtered to common datasets)
    perform_rq3_test(df)

if __name__ == "__main__":
    main()