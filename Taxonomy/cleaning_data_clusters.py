import os
import re
import argparse
from typing import Dict, List, Set

import numpy as np
import pandas as pd
from tqdm import tqdm
from huggingface_hub import login
from datasets import Dataset, DatasetDict

from docker_cdhit import cd_hit


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Build cluster-based splits (no representatives) for taxonomy ranks using CD-HIT clusters."
    )
    parser.add_argument("--hf_username", type=str, default="GleghornLab", help="Hugging Face username.")
    parser.add_argument("--hf_token", type=str, default=None, help="Hugging Face token.")
    parser.add_argument("--similarity_threshold", type=float, default=0.4, help="Similarity threshold for CD-HIT.")
    parser.add_argument("--n", type=int, default=2, help="CD-HIT word size (5 faster, 3 more sensitive).")
    parser.add_argument("--memory_percentage", type=float, default=0.5, help="Memory percentage for CD-HIT.")
    parser.add_argument("--raw_data_file", type=str, default="raw.tsv", help="Raw data file.")
    parser.add_argument("--eval_size", type=int, default=5000, help="Target number of eval rows for each of valid/test.")
    parser.add_argument("--min_class_size", type=int, default=100, help="Minimum samples per class to keep.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for cluster selection.")
    parser.add_argument("--force_recluster", action="store_true", help="Force rebuild of FASTA and CD-HIT clusters.")
    return parser.parse_args()


def extract_taxonomic_ids(lineage_str: str, ranks: List[str]) -> List[str]:
    rank_id_map: Dict[str, str] = {}
    if pd.isna(lineage_str):
        return [None] * len(ranks)
    for item in str(lineage_str).split(","):
        match = re.match(r"\s*(\d+)\s*\(([^)]+)\)", item.strip())
        if match:
            tax_id, rank = match.groups()
            rank = rank.strip().lower()
            if rank in ranks:
                rank_id_map[rank] = tax_id
    return [rank_id_map.get(rank) for rank in ranks]


def parse_cdhit_clusters(cluster_file: str) -> Dict[str, str]:
    """Parse a .clstr file from CD-HIT and return mapping from Entry (header) to cluster_id.

    CD-HIT .clstr lines look like:
    >Cluster 0
    0 123aa, >EntryA... *
    1 124aa, >EntryB... at 99.20%
    """
    entry_to_cluster: Dict[str, str] = {}
    current_cluster = None
    with open(cluster_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                # e.g. ">Cluster 0"
                # keep the raw suffix as the id to avoid collisions
                current_cluster = line.split("Cluster")[1].strip()
            else:
                # e.g. "0 123aa, >EntryA... *"
                try:
                    entry = line.split(">")[1].split("...")[0].strip()
                except Exception:
                    continue
                if current_cluster is not None:
                    entry_to_cluster[entry] = current_cluster
    return entry_to_cluster


def select_clusters_to_reach_size(
    df: pd.DataFrame, cluster_column: str, target_size: int, rng: np.random.Generator
) -> Set[str]:
    """Randomly select clusters until at least target_size rows are covered."""
    clusters = df[cluster_column].unique().tolist()
    rng.shuffle(clusters)
    selected: Set[str] = set()
    total = 0
    counts = df.groupby(cluster_column).size().to_dict()
    for cluster in clusters:
        selected.add(cluster)
        total += int(counts.get(cluster, 0))
        if total >= target_size:
            break
    return selected


def build_and_push_splits(
    df: pd.DataFrame,
    entry_to_cluster: Dict[str, str],
    ranks: List[str],
    hf_path_base: str,
    eval_size: int,
    min_class_size: int,
    similarity_threshold: float,
) -> None:
    rng = np.random.default_rng(42)

    for rank in tqdm(ranks, desc="Processing ranks with clusters"):
        # Attach current rank and cluster ids
        rank_df = df[["Entry", "Sequence", rank]].copy()
        # Normalize sequences to ensure consistent comparison and mapping
        rank_df["Sequence"] = (
            rank_df["Sequence"].astype(str).str.replace(r"\s+", "", regex=True).str.upper()
        )
        rank_df = rank_df.dropna(subset=[rank])
        rank_df["cluster"] = rank_df["Entry"].map(entry_to_cluster)
        rank_df = rank_df.dropna(subset=["cluster"]).copy()
        rank_df["current_rank"] = rank_df[rank]
        rank_df = rank_df.drop(columns=[rank])

        # Assert: any identical sequences must belong to exactly one cluster
        # This ensures cluster-based splitting guarantees sequence disjointness
        seq_cluster_uniques = rank_df.groupby("Sequence")["cluster"].nunique()
        multi_cluster_sequences = seq_cluster_uniques[seq_cluster_uniques > 1]
        assert multi_cluster_sequences.empty, (
            f"Found {len(multi_cluster_sequences)} sequences assigned to multiple clusters for rank '{rank}'. "
            f"Duplicates must be in the same cluster."
        )

        # Remove rare classes
        class_counts = rank_df["current_rank"].value_counts()
        keep_classes = set(class_counts[class_counts >= min_class_size].index)
        rank_df = rank_df[rank_df["current_rank"].isin(keep_classes)].copy()

        # Factorize labels over the whole filtered df to keep mapping consistent
        rank_df["labels"] = pd.factorize(rank_df["current_rank"])[0]

        # Select clusters for test and valid
        test_clusters = select_clusters_to_reach_size(rank_df, "cluster", eval_size, rng)
        remaining_df = rank_df[~rank_df["cluster"].isin(test_clusters)]
        valid_clusters = select_clusters_to_reach_size(remaining_df, "cluster", eval_size, rng)

        test_df = rank_df[rank_df["cluster"].isin(test_clusters)].copy()
        valid_df = rank_df[rank_df["cluster"].isin(valid_clusters)].copy()
        train_df = rank_df[
            ~rank_df["cluster"].isin(test_clusters.union(valid_clusters))
        ].copy()

        # Basic size checks
        assert len(test_df) >= eval_size, f"Test set too small for rank {rank}: {len(test_df)} < {eval_size}"
        assert len(valid_df) >= eval_size, f"Valid set too small for rank {rank}: {len(valid_df)} < {eval_size}"
        assert len(train_df) > 0, f"Train set empty for rank {rank}"

        # Assert no overlaps by Entry
        test_entries = set(test_df["Entry"])  # type: ignore[arg-type]
        valid_entries = set(valid_df["Entry"])  # type: ignore[arg-type]
        train_entries = set(train_df["Entry"])  # type: ignore[arg-type]
        assert test_entries.isdisjoint(valid_entries)
        assert test_entries.isdisjoint(train_entries)
        assert valid_entries.isdisjoint(train_entries)

        # Assert no overlaps by cluster
        test_clusters_set = set(test_df["cluster"])  # type: ignore[arg-type]
        valid_clusters_set = set(valid_df["cluster"])  # type: ignore[arg-type]
        train_clusters_set = set(train_df["cluster"])  # type: ignore[arg-type]
        assert test_clusters_set.isdisjoint(valid_clusters_set)
        assert test_clusters_set.isdisjoint(train_clusters_set)
        assert valid_clusters_set.isdisjoint(train_clusters_set)

        # Assert no overlaps by exact sequence strings
        test_seqs = set(test_df["Sequence"])  # type: ignore[arg-type]
        valid_seqs = set(valid_df["Sequence"])  # type: ignore[arg-type]
        train_seqs = set(train_df["Sequence"])  # type: ignore[arg-type]
        assert test_seqs.isdisjoint(valid_seqs)
        assert test_seqs.isdisjoint(train_seqs)
        assert valid_seqs.isdisjoint(train_seqs)

        # Shuffle rows
        test_df = test_df.sample(frac=1, random_state=42).reset_index(drop=True)
        valid_df = valid_df.sample(frac=1, random_state=42).reset_index(drop=True)
        train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)

        dataset_dict = DatasetDict({
            "train": Dataset.from_pandas(train_df),
            "valid": Dataset.from_pandas(valid_df),
            "test": Dataset.from_pandas(test_df),
        })

        # Suffix with _clusters to distinguish from representative-based datasets
        dataset_name = f"{hf_path_base}_{rank}_{similarity_threshold}_clusters"
        dataset_dict.push_to_hub(dataset_name)


def main(args):
    if args.hf_token:
        login(token=args.hf_token)
    else:
        print("No Hugging Face token provided, skipping login.")

    # Load raw data
    df = pd.read_csv(args.raw_data_file, sep='\t', encoding='utf-8')
    df.columns = df.columns.str.strip()
    print("Columns found:", df.columns.tolist())

    # Filter by sequence length
    filtered = df[(df['Length'] >= 20) & (df['Length'] <= 2046)].reset_index(drop=True)
    ranks = ["domain", "kingdom", "phylum", "class", "order", "family", "genus", "species"]

    # Extract taxonomy ids
    tax_ids = filtered["Taxonomic lineage (Ids)"].apply(lambda s: extract_taxonomic_ids(s, ranks))
    tax_id_df = pd.DataFrame(tax_ids.tolist(), columns=ranks)
    rank_df = pd.concat([filtered, tax_id_df], axis=1)

    hf_path_base = f"{args.hf_username}/taxonomy"

    # Normalize sequences consistently before writing FASTA and processing
    rank_df["Sequence"] = (
        rank_df["Sequence"].astype(str).str.replace(r"\s+", "", regex=True).str.upper()
    )

    # Build combined FASTA (unique Entry)
    all_fasta_path = "all.fasta"
    if args.force_recluster or not os.path.exists(all_fasta_path):
        unique_entries = rank_df[["Entry", "Sequence"]].drop_duplicates(subset=["Sequence"]).reset_index(drop=True)
        with open(all_fasta_path, 'w') as fasta:
            for _, row in tqdm(unique_entries.iterrows(), total=len(unique_entries), desc="Writing combined FASTA"):
                fasta.write(f">{row['Entry']}\n{row['Sequence']}\n")
    else:
        print(f"Found existing FASTA at {all_fasta_path}; skipping write.")

    # Run CD-HIT on combined FASTA (once)
    output_base = os.path.splitext(os.path.basename(all_fasta_path))[0]  # "all"
    output_prefix = f"output_{output_base}_{args.similarity_threshold}"
    cluster_file = f"{output_prefix}.clstr"
    if not args.force_recluster and os.path.exists(cluster_file):
        print(f"Found existing cluster file at {cluster_file}; skipping clustering.")
    else:
        cd_hit(
            all_fasta_path,
            similarity_threshold=args.similarity_threshold,
            n=args.n,
            memory_percentage=args.memory_percentage,
            output_path=output_prefix,
        )
        print("CD-HIT completed for combined dataset")

    # Parse clusters
    entry_to_cluster = parse_cdhit_clusters(cluster_file)
    print(f"Parsed {len(entry_to_cluster)} entries with cluster assignments")

    # Build and push splits per rank
    build_and_push_splits(
        df=rank_df,
        entry_to_cluster=entry_to_cluster,
        ranks=ranks,
        hf_path_base=hf_path_base,
        eval_size=args.eval_size,
        min_class_size=args.min_class_size,
        similarity_threshold=args.similarity_threshold,
    )


if __name__ == "__main__":
    args = parse_arguments()
    main(args)


