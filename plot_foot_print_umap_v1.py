import pandas as pd
import numpy as np
import umap
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.sparse import csr_matrix, vstack
import argparse
import sys

def parse_bed12(file_path, label, bin_size, min_len, max_len, seq_len):
    num_bins = seq_len // bin_size
    rows, cols_idx = [], []
    fiber_names = []
    
    print(f"--- loading [{label}]: {file_path} ---")
    try:
        df = pd.read_csv(file_path, sep='\t', header=None, compression='infer')
    except Exception as e:
        print(f"fail to load: {e}")
        return None, None, None

    df['read_len'] = df[2] - df[1]
    df_filtered = df[(df['read_len'] >= min_len) & (df['read_len'] <= max_len)].copy()
    
    valid_fiber_count = 0
    for _, row in df_filtered.iterrows():
        st = row[1]
        try:
            sizes = [int(x) for x in str(row[10]).strip(',').split(',')]
            starts = [int(x) for x in str(row[11]).strip(',').split(',')]
            
            if len(sizes) > 2:
                for sz, off in zip(sizes[1:-1], starts[1:-1]):
                    abs_s, abs_e = st + off, st + off + sz
                    b_s = int(max(0, abs_s // bin_size))
                    b_e = int(min(num_bins, abs_e // bin_size))
                    for b in range(b_s, b_e):
                        rows.append(valid_fiber_count)
                        cols_idx.append(b)
                
                fiber_names.append(row[3])
                valid_fiber_count += 1
        except:
            continue
            
    print(f"group [{label}] done: {valid_fiber_count} Reads kept (original: {len(df)})")
    
    data = np.ones(len(rows), dtype=np.int8)
    mat = csr_matrix((data, (rows, cols_idx)), shape=(valid_fiber_count, num_bins))
    
    return mat, fiber_names, [label] * valid_fiber_count

def main():
    parser = argparse.ArgumentParser(description="nucleosome positioning UMAP analysis for multiple BED12 files")
    parser.add_argument("-i", "--inputs", nargs='+', required=True, help="input BED12 files")
    parser.add_argument("-l", "--labels", nargs='+', required=True, help="corresponding group names")
    parser.add_argument("-o", "--output", required=True, help="output file prefix")
    parser.add_argument("-min", "--min_len", type=int, default=3000, help="minimum read length filter (default 3000)")
    parser.add_argument("-max", "--max_len", type=int, default=3400, help="maximum read length filter (default 3400)")
    parser.add_argument("-b", "--bin", type=int, default=10, help="resolution (default 10bp)")
    parser.add_argument("-s", "--seq_len", type=int, default=3200, help="genome reference length (default 3200)")
    
    args = parser.parse_args()
    if len(args.inputs) != len(args.labels):
        print("Error: The number of input files does not match the number of labels!")
        sys.exit(1)

    all_mats, all_names, all_labs = [], [], []

    # 1. input
    for f, label in zip(args.inputs, args.labels):
        mat, names, labs = parse_bed12(f, label, args.bin, args.min_len, args.max_len, args.seq_len)
        if mat is not None:
            all_mats.append(mat)
            all_names.extend(names)
            all_labs.extend(labs)

    if not all_mats:
        print("No data available for processing. Please check the files or filtering conditions.")
        return

    # 2. generate matrix and UMAP
    combined_mat = vstack(all_mats)
    print(f"All data has been merged. Total matrix size: {combined_mat.shape}. Computing UMAP...")
    
    reducer = umap.UMAP(n_neighbors=30, min_dist=0.1, metric='jaccard', n_jobs=-1, random_state=42)
    embedding = reducer.fit_transform(combined_mat)

    # 3. summary results
    res_df = pd.DataFrame({
        'UMAP1': embedding[:, 0],
        'UMAP2': embedding[:, 1],
        'Group': all_labs,
        'Fiber': all_names
    })
    res_df.to_csv(f"{args.output}_coords.csv", index=False)

    # 4. visualization
    print("Generating comparison plots...")
    sns.set_style("white")
    
    # figure 1: overlay plot (all groups together)
    plt.figure(figsize=(10, 8))
    sns.scatterplot(data=res_df, x='UMAP1', y='UMAP2', hue='Group', s=1, alpha=0.4, palette='tab10', edgecolors='none')
    plt.title(f"Joint UMAP: {', '.join(args.labels)}")
    plt.savefig(f"{args.output}_overlay.pdf", dpi=300, bbox_inches='tight')

    # figure 2: facet plot (individual clusters)
    g = sns.FacetGrid(res_df, col="Group", hue="Group", palette='tab10', height=5, col_wrap=3)
    g.map(sns.scatterplot, "UMAP1", "UMAP2", s=1, alpha=0.3, edgecolors='none')
    g.add_legend()
    g.set_titles("{col_name}")
    g.savefig(f"{args.output}_facets.pdf", dpi=300, bbox_inches='tight')

    print(f"Analysis complete! Results saved to {args.output}_*.pdf")

if __name__ == "__main__":
    main()
