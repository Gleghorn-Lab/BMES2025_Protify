import pandas as pd
import re
import subprocess
import os
import csv
import psutil
from datasets import Dataset, DatasetDict
import sklearn.model_selection
from sklearn.model_selection import train_test_split

# Load the raw data
df = pd.read_csv('raw.tsv', sep='\t')

# Strip any extra spaces from column names
df.columns = df.columns.str.strip()
print("Columns found:", df.columns.tolist()) 

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
phylum_df = pd.concat([filtered, tax_id_df], axis=1)

# Create phylum dataset (phylum name and sequence)
phylum_df = phylum_df[['Entry','phylum', 'Sequence']].copy()
phylum_df = phylum_df.dropna()
print(f"Number of unique phylum: {phylum_df['phylum'].nunique()}") #prints 116
print(f"Total number of rows: {len(phylum_df)}") #prints 565,789
print(phylum_df.head())
phylum_df.to_csv('phylum.tsv', sep='\t', index=False)
print(f"Filtered entries saved to {"phylum.tsv"}")

#convert the tsv file to a fasta format
with open("phylum.tsv", newline='') as tsv, open("phylum.fasta", 'w') as fasta:
    reader = csv.DictReader(tsv, delimiter='\t')
    for row in reader:
        # build a FASTA header:
        header = f">{row['Entry']}|{row['phylum']}"
        seq    = row['Sequence']
        fasta.write(f"{header}\n{seq}\n")


#CD-HIT using Docker
def cd_hit(
        fasta_file: str,
        similarity_threshold: float = 0.5,
        n: int = 2, # word size, 5 is faster but 3 is more sensitive
        memory_percentage: float = 0.5,
    ):
    output_path = f"output_{fasta_file.split('.')[0]}_{similarity_threshold}"

    # Run cd-hit in Docker
    # Build the cd-hit Docker image if not already built
    num_cpu = os.cpu_count() - 4 if os.cpu_count() > 4 else 1
    memory_max = int(memory_percentage * psutil.virtual_memory().total / 1024 / 1024)  # in MB
    print(f'Using {num_cpu} CPUs and {memory_max} MB memory')

    print("Building cd-hit Docker image...")
    docker_image = "cd-hit"
    dockerfile_url = "https://raw.githubusercontent.com/weizhongli/cdhit/master/Docker/Dockerfile"
    # Build the Docker image
    try:
        subprocess.run([
            "docker", "build", "--tag", docker_image, dockerfile_url
        ], check=True)

        subprocess.run([
            "docker", "run",
            "-v", f"{os.getcwd()}:/data",
            "-w", "/data",
            docker_image,
            "cd-hit",
            "-i", fasta_file,
            "-o", output_path,
            "-d", "0",
            "-c", str(similarity_threshold),
            "-n", str(n),
            "-T", str(num_cpu),
            "-M", str(memory_max)
        ], check=True)
    except:
        subprocess.run([
            "sudo", "docker", "build", "--tag", docker_image, dockerfile_url
        ], check=True)

        subprocess.run([
            "sudo", "docker", "run",
            "-v", f"{os.getcwd()}:/data",
            "-w", "/data",
            docker_image,
            "cd-hit",
            "-i", fasta_file,
            "-o", output_path,
            "-d", "0",
            "-c", str(similarity_threshold),
            "-n", str(n),
            "-T", str(num_cpu),
            "-M", str(memory_max)
        ], check=True)

### make sure docker desktop is running
fasta_file = 'phylum.fasta'
cd_hit(fasta_file, similarity_threshold=0.5, n=2, memory_percentage=0.5)

#RESULT - output_phylum_0.5.fasta file showing the representative seqeunces and output_phylum_0.5.clstr showing the cluster results from CD-HIT
#Read the FASTA into a list of dicts
with open("output_phylum_0.5", "r") as f:
    lines = f.read().splitlines()

records = []
entry, phylum, sequence = "", "", ""

for line in lines:
    if line.startswith(">"):
        # Save the previous entry if any
        if entry:
            records.append({
                "Entry": entry,
                "phylum": phylum,
                "Sequence": sequence
            })

        # Start a new record
        sequence = ""
        header = line[1:]  # remove ">"
        parts = header.split("|", 1)
        entry = parts[0] if len(parts) > 0 else None
        phylum = parts[1] if len(parts) > 1 else None
    else:
        sequence += line.strip()

if entry:
    records.append({
        "Entry": entry,
        "phylum": phylum,
        "Sequence": sequence
    })

df = pd.DataFrame(records)
print(df.head())

print(f"Before dropping duplicates: {len(df)}")
df = df.drop_duplicates(subset=['phylum','Sequence'], keep='first') #no duplicates found
print(f"After dropping duplicates: {len(df)}")
##RESULT - no duplicated found or removed

print(f"Number of unique phylum: {df['phylum'].nunique()}") #prints 116
print(f"Total number of rows: {len(df)}") #prints 269,838

# Remove fewer than 100 samples
phylum_counts = df['phylum'].value_counts()
df = df[df['phylum'].isin(phylum_counts[phylum_counts >= 100].index)]
print(f"Number of unique phylum after removing rare phylum: {df['phylum'].nunique()}") #prints 65
print(f"Total number of rows after removing rare phylum: {len(df)}") #prints 268,989

# Create a new 'label' column based on species
df["labels"] = pd.factorize(df["phylum"])[0]
print(df.head())

output_filename = 'phylum_labeled.csv'
df.to_csv(output_filename, index=False)
print(df.head())

# Stratified split: first get test set (5,000), then eval (5,000), rest is train
train_valid, test = train_test_split(df, test_size=5000, stratify=df['labels'], random_state=42)
train, valid = train_test_split(train_valid, test_size=5000, stratify=train_valid['labels'], random_state=42)

train = train.reset_index(drop=True)
valid = valid.reset_index(drop=True)
test = test.reset_index(drop=True)

train.to_csv('train.csv', index=False)
valid.to_csv('valid.csv', index=False)
test.to_csv('test.csv', index=False)

# Convert each to Hugging Face Datasets
dataset1 = Dataset.from_pandas(train)
dataset2 = Dataset.from_pandas(valid)
dataset3 = Dataset.from_pandas(test)

# Combine into a DatasetDict
dataset_dict = DatasetDict({
    "train": dataset1,
    "valid": dataset2,
    "test": dataset3
})

dataset_dict.push_to_hub("")