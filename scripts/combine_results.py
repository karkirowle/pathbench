import pandas as pd
import numpy as np
import glob
import os
import re

# --- CONFIGURATION ---
FILE_PATTERN = "results_5/*.txt"
OUTPUT_TEX_FILE = "complex_evaluation_summary.tex"

# --- 2. METRIC DEFINITIONS ---
METRIC_ROW_MAP = {
    'speech_rate': 'Speech Rate',
    'cpp': 'CPP',
    'f0_range': 'F0 Range',
    'std_pitch': 'Std Pitch',
    'wada_snr': 'WADA SNR',
    'artp_old': 'Entropy',
    'age': 'Age',       # Mapped key
    'spk2age': 'Age',   # Handle raw key if it appears
    'vsa': 'VSA',
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
    'nad_all': 'NAD All'
}

REF_FREE_KEYS = [
    'speech_rate', 'cpp', 'f0_range', 'std_pitch', 'wada_snr', 
    'vsa', 'double_asr', 'artp_double_asr', 'artp_old', 'age', 'spk2age'
]

def parse_txt_file(filepath):
    filename = os.path.basename(filepath).lower()
    
    # --- 1. DETECT METADATA FROM FILENAME ---
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

    # --- 2. PARSE FILE CONTENT ---
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

        # --- FIX: Map spk2age to age explicitly ---
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
        if m in ['per', 'dper', 'nad_control', 'nad_all']: 
            mask = df['MetricKey'] == m
            df.loc[mask, 'Value'] = df.loc[mask, 'Value'].abs()
            
        m_data = df[df['MetricKey'] == m]['Value'].dropna()
        if len(m_data) > 0 and (m_data < 0).all():
             mask = df['MetricKey'] == m
             df.loc[mask, 'Value'] = df.loc[mask, 'Value'] * -1
    return df

def get_table_column_order():
    """
    Defines the Python list order.
    MUST match the LaTeX Header indices exactly.
    """
    return [
        ('UASpeech', 'Word', 'PB'), ('UASpeech', 'Word', 'PU'),
        ('NeuroVoz', 'Utterance', 'PB'), ('NeuroVoz', 'Utterance', 'PU'),
        ('EasyCall', 'Word', 'PB'), ('EasyCall', 'Word', 'PU'),
        ('EasyCall', 'Utterance', 'PB'), ('EasyCall', 'Utterance', 'PU'),
        
        # COPAS Word: PB, PU, ALL
        ('COPAS', 'Word', 'PB'), ('COPAS', 'Word', 'PU'), ('COPAS', 'Word', 'ALL'),
        
        # COPAS Utterance: PB, PU, ALL
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
    # Added 'age' to the signal group
    groups = [
        ("Reference-Free (Signal)", ['speech_rate', 'cpp', 'f0_range', 'std_pitch', 'wada_snr', 'artp_old', 'age']),
        ("Reference-Free (Speaker)", ['vsa']),
        ("Reference-Free (Model)", ['double_asr', 'artp_double_asr']),
        ("Reference-Text", ['per', 'dper', 'artp']),
        ("Reference-Audio (Parallel)", ['p_estoi_control', 'p_estoi_all', 'p_estoi_fa_control', 'p_estoi_fa_all', 'nad_control', 'nad_all'])
    ]
    
    for group_name, metrics in groups:
        latex_rows.append(f"        \\multicolumn{{20}}{{l}}{{\\textit{{{group_name}}}}} \\\\")
        
        for m_key in metrics:
            display_name = METRIC_ROW_MAP.get(m_key, m_key)
            if m_key in ['p_estoi_control', 'nad_control', 'p_estoi_fa_control']:
                base_name = display_name.replace(" Control", "").replace(" FA", " FA")
                latex_rows.append(f"        {base_name} &&&&&&&&&&&&&&&&&&& \\\\")
                display_name = "\\hspace{3mm} \\textit{Control}"
            elif m_key in ['p_estoi_all', 'nad_all', 'p_estoi_fa_all']:
                display_name = "\\hspace{3mm} \\textit{All}"
            
            row_str = f"        {display_name}"
            for col_def in target_tuples:
                val = pivot.loc[m_key, col_def] if m_key in pivot.index else np.nan
                cell_tex = "--"
                if not pd.isna(val):
                    cell_tex = f"{val:.2f}"
                    if val == col_max_global[col_def] and val != -999:
                        cell_tex = f"\\textbf{{{cell_tex}}}"
                    elif (m_key in REF_FREE_KEYS) and (val == col_max_rf[col_def]) and (val != -999):
                        cell_tex = f"\\underline{{{cell_tex}}}"
                row_str += f" & {cell_tex}"
            row_str += " \\\\"
            latex_rows.append(row_str)

    # --- LATEX HEADER ---
    # Matches: UA(2) | NV(2) | EC-W(2) | EC-U(2) | CO-W(3: PB,PU,ALL) | CO-U(3: PB,PU,ALL) | TO-W(2) | TO-U(2) | YT(1)
    header = r"""\begin{table*}[t]
    \centering
    \caption{Speaker-level Pearson Correlation Coefficient (PCC). \textbf{PB}: Phonetically Balanced, \textbf{PU}: Phonetically Unbalanced, \textbf{ALL}: Combined (COPAS only). \textbf{Bold}: Best overall. \underline{Underline}: Best Reference-Free. (\textit{Indented rows indicate reference set: Control vs. All}).}
    \label{tab:main_results}
    \resizebox{\textwidth}{!}{%
    \begin{tabular}{l|cc|cc|cc|cc|ccc|ccc|cc|cc|c}
        \toprule
        & \multicolumn{2}{c|}{\textbf{UASpeech}} & \multicolumn{2}{c|}{\textbf{NeuroVoz}} & \multicolumn{2}{c|}{\textbf{EasyCall}} & \multicolumn{2}{c|}{\textbf{EasyCall}} & \multicolumn{3}{c|}{\textbf{COPAS}} & \multicolumn{3}{c|}{\textbf{COPAS}} & \multicolumn{2}{c|}{\textbf{TORGO}} & \multicolumn{2}{c|}{\textbf{TORGO}} & \textbf{YT} \\
        & \multicolumn{2}{c|}{\textit{Word}} & \multicolumn{2}{c|}{\textit{Utterance}} & \multicolumn{2}{c|}{\textit{Word}} & \multicolumn{2}{c|}{\textit{Utterance}} & \multicolumn{3}{c|}{\textit{Word}} & \multicolumn{3}{c|}{\textit{Utterance}} & \multicolumn{2}{c|}{\textit{Word}} & \multicolumn{2}{c|}{\textit{Utterance}} & \textit{Utt} \\
        \cmidrule(lr){2-3} \cmidrule(lr){4-5} \cmidrule(lr){6-7} \cmidrule(lr){8-9} \cmidrule(lr){10-12} \cmidrule(lr){13-15} \cmidrule(lr){16-17} \cmidrule(lr){18-19} \cmidrule(lr){20-20}
        \textbf{Metric} & \textbf{PB} & \textbf{PU} & \textbf{PB} & \textbf{PU} & \textbf{PB} & \textbf{PU} & \textbf{PB} & \textbf{PU} & \textbf{PB} & \textbf{PU} & \textbf{ALL} & \textbf{PB} & \textbf{PU} & \textbf{ALL} & \textbf{PB} & \textbf{PU} & \textbf{PB} & \textbf{PU} & \textbf{ALL} \\
        \midrule"""
    footer = r"""
        \bottomrule
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
    generate_latex(df)

if __name__ == "__main__":
    main()