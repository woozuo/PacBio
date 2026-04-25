import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from statsmodels.nonparametric.smoothers_lowess import lowess
import sys

def parse_m6a_to_matrix(file_path, label, bin_size, min_len, max_len, seq_len):
    """decoding BED12, reture Fibers x Bins matrix"""
    num_bins = seq_len // bin_size
    print(f"--- parsing [{label}]: {file_path} ---")
    
    try:
        df = pd.read_csv(file_path, sep='\t', header=None, compression='infer')
        df['read_len'] = df[2] - df[1]
        df_filtered = df[(df['read_len'] >= min_len) & (df['read_len'] <= max_len)]
    except Exception as e:
        print(f"err: parsing: {e}")
        return None
    
    matrix = np.zeros((len(df_filtered), num_bins))
    
    for i, (_, row) in enumerate(df_filtered.iterrows()):
        st = row[1]
        try:
            starts = [int(x) for x in str(row[11]).strip(',').split(',')]
            for off in starts:
                b_idx = int((st + off) // bin_size)
                if b_idx < num_bins:
                    matrix[i, b_idx] += 1
        except:
            continue
            
    print(f"Group [{label}] processed {matrix.shape[0]} Reads.")
    return matrix

def smooth_curve(x, y, frac=0.05):
    """LOESS smooth"""
    filtered = lowess(y, x, frac=frac)
    return filtered[:, 1]

def main():
    parser = argparse.ArgumentParser(description="Plot m6A profile with LOESS smoothing and SD shading")
    parser.add_argument("-i", "--inputs", nargs='+', required=True, help="Input BED12 file list")
    parser.add_argument("-l", "--labels", nargs='+', required=True, help="Corresponding group labels")
    parser.add_argument("-o", "--output", default="HBV_m6A_Loess_SD.png")
    parser.add_argument("-b", "--bin", type=int, default=10, help="Bin resolution (default 10bp)")
    parser.add_argument("-f", "--frac", type=float, default=0.05, help="LOESS smoothing coefficient (0-1, larger values result in smoother curves)")
    parser.add_argument("-min", "--min_len", type=int, default=3000)
    parser.add_argument("-max", "--max_len", type=int, default=3400)
    
    args = parser.parse_args()
    seq_len = 3200
    x_axis = np.arange(0, seq_len, args.bin)[:seq_len // args.bin]

    plt.figure(figsize=(12, 6), dpi=300)
    sns.set_style("white")
    colors = sns.color_palette("Set1", len(args.inputs))

    for f, label, color in zip(args.inputs, args.labels, colors):
        matrix = parse_m6a_to_matrix(f, label, args.bin, args.min_len, args.max_len, seq_len)
        if matrix is None or matrix.shape[0] == 0: continue

        # 1. Calculate mean and standard deviation for each bin (across molecules)
        mean_signal = np.mean(matrix, axis=0)
        sd_signal = np.std(matrix, axis=0)

        # 2. LOESS smooth average signal
        y_smooth = smooth_curve(x_axis, mean_signal, frac=args.frac)
        
        # 3. LOESS smooth SD
        sd_smooth = smooth_curve(x_axis, sd_signal, frac=args.frac)

        # 4. Plotting
        plt.plot(x_axis, y_smooth, label=f"{label} (Loess)", color=color, lw=2)
        plt.fill_between(x_axis, 
                         np.maximum(0, y_smooth - sd_smooth), 
                         y_smooth + sd_smooth, 
                         color=color, alpha=0.2)

    plt.title(f"HBV cccDNA m6A Profile (Loess Smooth, frac={args.frac})", fontsize=14)
    plt.xlabel("Genomic Position (bp)", fontsize=12)
    plt.ylabel("Normalized m6A Signal Density", fontsize=12)
    plt.xlim(0, seq_len)
    plt.legend()
    sns.despine()
    
    plt.tight_layout()
    plt.savefig(args.output)
    print(f"Generated smooth plot: {args.output}")

if __name__ == "__main__":
    main()
