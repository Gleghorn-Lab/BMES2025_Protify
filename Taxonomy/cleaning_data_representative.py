import pandas as pd
import re
import subprocess
import os
import csv
import psutil
import sklearn.model_selection
import argparse
from tqdm import tqdm
from huggingface_hub import login
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split

from docker_cdhit import cd_hit


def parse_arguments():
    parser = argparse.ArgumentParser(description="Script with arguments mirroring the provided YAML settings.")
    parser.add_argument("--hf_username", type=str, default="GleghornLab", help="Hugging Face username.")
    parser.add_argument("--hf_token", type=str, default=None, help="Hugging Face token.")
    parser.add_argument("--similarity_threshold", type=float, default=0.4, help="Similarity threshold for CD-HIT.")
    parser.add_argument("--n", type=int, default=2, help="Number of threads for CD-HIT.")
    parser.add_argument("--memory_percentage", type=float, default=0.5, help="Memory percentage for CD-HIT.")
    parser.add_argument("--raw_data_file", type=str, default="raw.tsv", help="Raw data file.")
    parser.add_argument("--eval_size", type=int, default=5000, help="Evaluation size.")
    return parser.parse_args()


def main(args):
    if args.hf_token:
        login(token=args.hf_token)
    else:
        print("No Hugging Face token provided, skipping login.")

    # Load the raw data
    df = pd.read_csv(args.raw_data_file, sep='\t', encoding='utf-8')
    df.columns = df.columns.str.strip()
    print("Columns found:", df.columns.tolist())

    hf_path_base = f'{args.hf_username}/taxonomy'


    # Filter by sequence length
    filtered = df[(df['Length'] >= 20) & (df['Length'] <= 2048)]
    ranks = ["domain", "kingdom", "phylum", "class", "order", "family", "genus", "species"]


    def extract_taxonomic_ids(lineage_str):
        rank_id_map = {}
        if pd.isna(lineage_str):
            return [None] * len(ranks)
        for item in lineage_str.split(","):
            match = re.match(r"\s*(\d+)\s*\(([^)]+)\)", item.strip())
            if match:
                tax_id, rank = match.groups()
                rank = rank.strip().lower()
                if rank in ranks:
                    rank_id_map[rank] = tax_id
        return [rank_id_map.get(rank) for rank in ranks]


    # Apply to the filtered DataFrame
    filtered = filtered.reset_index(drop=True)
    tax_ids = filtered["Taxonomic lineage (Ids)"].apply(extract_taxonomic_ids)
    tax_id_df = pd.DataFrame(tax_ids.tolist(), columns=ranks)
    rank_df = pd.concat([filtered, tax_id_df], axis=1)

    # Build a single FASTA for the entire dataset (one header per Entry)
    all_fasta_path = "all.fasta"
    if not os.path.exists(all_fasta_path):
        unique_entries = rank_df[["Entry", "Sequence"]].drop_duplicates(subset=["Entry"]).reset_index(drop=True)
        with open(all_fasta_path, 'w') as fasta:
            for _, row in tqdm(unique_entries.iterrows(), total=len(unique_entries), desc="Writing combined FASTA"):
                header = f">{row['Entry']}"
                seq = row['Sequence']
                fasta.write(f"{header}\n{seq}\n")
    else:
        print(f"Found existing FASTA at {all_fasta_path}; skipping write.")

    # Read the representative sequences from CD-HIT output
    output_base = os.path.splitext(os.path.basename(all_fasta_path))[0]  # "all"
    output_fasta_path = f"output_{output_base}_{args.similarity_threshold}"

    # Run CD-HIT once on the combined FASTA
    if os.path.exists(output_fasta_path):
        print(f"Found existing CD-HIT output at {output_fasta_path}; skipping clustering.")
    else:
        cd_hit(
            all_fasta_path,
            similarity_threshold=args.similarity_threshold,
            n=args.n,
            memory_percentage=args.memory_percentage,
            output_path=output_fasta_path
        )
        print("CD-HIT completed for combined dataset")


    with open(output_fasta_path, "r") as f:
        lines = f.read().splitlines()

    rep_records = []
    entry, sequence = "", ""
    for line in tqdm(lines, desc="Reading combined representative FASTA"):
        if line.startswith(">"):
            if entry:
                rep_records.append({
                    "Entry": entry,
                    "Sequence": sequence
                })
            sequence = ""
            header = line[1:]  # remove ">"
            entry = header.strip()
        else:
            sequence += line.strip()
    if entry:
        rep_records.append({
            "Entry": entry,
            "Sequence": sequence
        })

    reps_df = pd.DataFrame(rep_records)
    print(reps_df.head())


    for rank in tqdm(ranks, desc="Processing ranks"):
        current_rank_df = rank_df[['Entry', rank]].copy()
        current_rank_df = current_rank_df.dropna()
        print(f"Number of unique {rank}: {current_rank_df[rank].nunique()}")
        print(f"Total number of rows: {len(current_rank_df)}")
        print(current_rank_df.head())

        # Merge representative sequences with current rank metadata
        df = reps_df.merge(rank_df[["Entry", rank]], on="Entry", how="left")
        df = df.dropna(subset=[rank]).copy()
        df["current_rank"] = df[rank]
        df = df.drop(columns=[rank])
        print(df.head())

        print(f"Before dropping duplicates: {len(df)}")
        df = df.drop_duplicates(subset=['current_rank','Sequence'], keep='first') #no duplicates found
        print(f"After dropping duplicates: {len(df)}")
        ##RESULT - no duplicated found or removed

        print(f"Number of unique classes: {df['current_rank'].nunique()}")
        print(f"Total number of rows: {len(df)}")

        # Remove species/classes with fewer than 100 samples
        rank_counts = df['current_rank'].value_counts()
        df = df[df['current_rank'].isin(rank_counts[rank_counts >= 100].index)]
        print(f"Number of unique classes after removing rare classes: {df['current_rank'].nunique()}")
        print(f"Total number of rows after removing rare classes: {len(df)}")

        # Create a new 'label' column based on species
        df["labels"] = pd.factorize(df["current_rank"])[0]
        print(df.head())

        # Stratified split: first get test set (5,000), then eval (5,000), rest is train
        train_valid, test = train_test_split(df, test_size=args.eval_size, stratify=df['labels'], random_state=42)
        train, valid = train_test_split(train_valid, test_size=args.eval_size, stratify=train_valid['labels'], random_state=42)

        train = train.reset_index(drop=True)
        valid = valid.reset_index(drop=True)
        test = test.reset_index(drop=True)

        # Combine into a DatasetDict
        dataset_dict = DatasetDict({
            "train": Dataset.from_pandas(train),
            "valid": Dataset.from_pandas(valid),
            "test": Dataset.from_pandas(test)
        })

        dataset_dict.push_to_hub(f"{hf_path_base}_{rank}_{args.similarity_threshold}")


if __name__ == "__main__":
    args = parse_arguments()
    main(args)