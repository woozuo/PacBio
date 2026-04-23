import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from scipy.sparse import csr_matrix

def parse_bed12_to_density(file_path, label, bin_size, min_len, max_len, seq_len):
    """
    convert format
    """
    num_bins = seq_len // bin_size
    print(f"--- reading [{label}]: {file_path} ---")
    
    try:
        df = pd.read_csv(file_path, sep='\t', header=None, compression='infer')
        # 1. reads length filter
        df['read_len'] = df[2] - df[1]
        df_filtered = df[(df['read_len'] >= min_len) & (df['read_len'] <= max_len)].copy()
    except Exception as e:
        print(f"err in reading: {e}")
        return pd.DataFrame()

    all_rows = []
    
    # 2. scan Read (Fiber)
    for _, row in df_filtered.iterrows():
        st = row[1]
        fiber_signal = np.zeros(num_bins)
        try:
            starts = [int(x) for x in str(row[11]).strip(',').split(',')]
            
            for off in starts:
                b_idx = int((st + off) // bin_size)
                if b_idx < num_bins:
                    fiber_signal[b_idx] += 1 
            
            # Seaborn 
            for b in range(num_bins):
                all_rows.append({
                    'Position': b * bin_size,
                    'Signal': fiber_signal[b],
                    'Group': label
                })
        except:
            continue
            
    print(f"Group [{label}] done, valid Read number: {len(df_filtered)}")
    return pd.DataFrame(all_rows)

def main():
    parser = argparse.ArgumentParser(description="plot m6a coverage")
    parser.add_argument("-i", "--inputs", nargs='+', required=True, help="BED12 input number")
    parser.add_argument("-l", "--labels", nargs='+', required=True, help="label names")
    parser.add_argument("-o", "--output", default="HBV_signal_lineplot.pdf")
    parser.add_argument("-b", "--bin", type=int, default=20, help="Bin size (default 20bp)")
    parser.add_argument("-min", "--min_len", type=int, default=3000, help="min Read length")
    parser.add_argument("-max", "--max_len", type=int, default=3400, help="max Read length")
    parser.add_argument("-s", "--seq_len", type=int, default=3200, help="reference genome size")
    
    args = parser.parse_args()

    if len(args.inputs) != len(args.labels):
        print("err: check number of files and labels")
        return

    # 3. summarize
    final_df_list = []
    for f, label in zip(args.inputs, args.labels):
        group_df = parse_bed12_to_density(f, label, args.bin, args.min_len, args.max_len, args.seq_len)
        final_df_list.append(group_df)
    
    combined_df = pd.concat(final_df_list, ignore_index=True)

    # 4. plot
    print("plotting...")
    plt.figure(figsize=(14, 7), dpi=300)
    sns.set_style("ticks")
    
    ax = sns.lineplot(
        data=combined_df, 
        x='Position', 
        y='Signal', 
        hue='Group', 
        errorbar='sd', 
        palette='Set1',
        linewidth=2,
        alpha=0.3
    )

    plt.title(f"HBV Single-molecule Signal Profile (Bin={args.bin}bp)", fontsize=15)
    plt.xlabel("Genomic Position (bp)", fontsize=12)
    plt.ylabel("Mean Signal Intensity ± SD", fontsize=12)
    plt.xlim(0, args.seq_len)
    plt.legend(title="Groups", loc='upper right')
    
    sns.despine()
    plt.tight_layout()
    plt.savefig(args.output)
    print(f"Plot saved to: {args.output}")

if __name__ == "__main__":
    main()
