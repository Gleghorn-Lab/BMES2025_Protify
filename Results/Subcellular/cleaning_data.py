import pandas as pd
import re
import subprocess
import os
import csv
import psutil
from datasets import Dataset, DatasetDict

tsv_path = "input_condensed.tsv"
output_path = "processed.tsv"
df = pd.read_csv(tsv_path, sep='\t', dtype=str)

# Strip any extra spaces from column names
df.columns = df.columns.str.strip()
print("Columns found:", df.columns.tolist()) 

#drop entries with missing subcellular location
print(f"Initial number of entries: {len(df)}")
df = df.dropna(subset=['Subcellular location [CC]'])
print(f"Entries after dropping invalid values: {len(df)}")
##RESULT- no entries were dropped


#filter based on sequence length (remove too short or too long)
df['Length'] = pd.to_numeric(df['Length'], errors='coerce')
df = df[(df['Length'] >= 20) & (df['Length'] <= 2048)]
print(f"Entries after dropping length: {len(df)}")
##RESULT - 2767 entries were dropped due to length

#Save the filtered DataFrame to a new TSV file
df.to_csv(output_path, sep='\t', index=False)
print(f"Filtered entries saved to {output_path}")


#convert the tsv file to a fasta format
with open("processed.tsv", newline='') as tsv, open("sl.fasta", 'w') as fasta:
    reader = csv.DictReader(tsv, delimiter='\t')
    for row in reader:
        # build a FASTA header:
        header = f">{row['Entry']}|{row['Entry Name']}|SL={row['Subcellular location [CC]']}"
        seq    = row['Sequence']
        fasta.write(f"{header}\n{seq}\n")


#CD-HIT using Docker
def cd_hit(
        fasta_file: str,
        similarity_threshold: float = 0.5,
        n: int = 5, # word size, 5 is faster but 3 is more sensitive
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
fasta_file = 'sl.fasta'
cd_hit(fasta_file, similarity_threshold=0.8, n=5, memory_percentage=0.5)

#RESULT - output_sl_0.8.fasta file showing the representative seqeunces and output_sl_0.8.clstr showing the cluster results from CD-HIT

#Read the FASTA into a list of dicts
with open("output_sl_0.8", "r") as f:
    lines = f.read().splitlines()

records = []
entry, subcellular_location, interpro, sequence = None, "", "", ""

for line in lines:
    if line.startswith(">"):
        # Save the previous entry if any
        if entry:
            records.append({
                "Entry": entry,
                "subcellular_location": subcellular_location,
                "Sequence": sequence
            })

        # Start a new record
        sequence = ""  # reset
        subcellular_location = ""
        interpro = "None"
        header = line[1:]  # remove ">"
        
        # Entry
        entry_match = re.match(r"(\w+)", header)
        entry = entry_match.group(1) if entry_match else None

        # Subcellular Location (SL)
        sl_match = re.search(r"SL=([^|]*)", header)
        subcellular_location = sl_match.group(1).strip() if sl_match else ""
    
    else:
        sequence += line.strip()

if entry:
    records.append({
        "Entry": entry,
        "subcellular_location": subcellular_location,
        "Sequence": sequence
    })

df = pd.DataFrame(records)
print(df['subcellular_location'])


#Parse subcellular_location into a clean list of labels
def parse_locations(raw: str) -> list[str]:
    if not isinstance(raw, str):
        return []
    s = raw

    #remove [] and {}
    s = re.sub(r"\{[^}]*\}", "", s)
    s = re.sub(r"\[[^\]]*\]", "", s)

    #If there's a stray '{' with no closing '}', drop from '{' to end
    s = re.sub(r"\{.*", "", s)

    #Remove any "note=..." fragments (up to next comma/semicolon/period)
    s = re.split(r'notes?=', s, flags=re.IGNORECASE)[0]

    #Split off the prefix "SUBCELLULAR LOCATION:"
    s = re.sub(r"^.*subcellular location\s*:?[\s]*", "", s, flags=re.IGNORECASE)

    #Split on any of , ; or . and clean
    raw_labels = re.split(r"[,;.]", s)

    #Strip/normalize
    labels = [re.sub(r"^[;:\s]+", "", lbl.strip().lower()) for lbl in raw_labels if lbl.strip()]
    #return labels

    mapped = [location_map[label] for label in labels if label in location_map]
    return list(set(mapped))


#mapping dictionary
location_map = {
    "cytoplasm": "cytoplasm",
    "nucleus": "nucleus",
    "secreted": "secreted",
    "membrane": "membrane",
    "mitochondrion": "mitochondrion",
    "plastid": "plastid",
    "endoplasmic reticulum": "endoplasmic reticulum",
    "vacuole": "vacuole",
    "lysosome": "lysosome",
    "golgi apparatus":"golgi apparatus",
    "peroxisome":"peroxisome",
    "chloroplast": "chloroplast",
    "cell wall": "cell wall",
}

# Apply parsing
df["locations"] = df["subcellular_location"].apply(parse_locations)
print(df['locations'])

#Build the unique label list
all_labels = sorted({lbl for row in df["locations"] for lbl in row})
n_labels = len(all_labels)
print(n_labels)


def make_bitstring(locs: list[str]) -> str:
    return "".join("1" if lbl in locs else "0" for lbl in all_labels)

df["bitstring"] = df["locations"].apply(make_bitstring)
df["labels"] = df["bitstring"].apply(lambda s: str([int(ch) for ch in s]))

print("Unique labels (in order):")
for idx, lbl in enumerate(all_labels):
    print(f"  [{idx}] {lbl}")

label_data = []
for idx, lbl in enumerate(all_labels):
    label_data.append({'Index': idx, 'Label': lbl})
labels_df = pd.DataFrame(label_data)
labels_df = labels_df.sort_values(by='Index').reset_index(drop=True)
output_filename = 'condensed_subcellular_labels.csv'
labels_df.to_csv(output_filename, index=False)

print(df.head())

print(f"Before dropping duplicates: {len(df)}")
df = df.drop_duplicates(subset=['subcellular_location','Sequence'], keep='first')
print(f"After dropping duplicates: {len(df)}")
##RESULT - no duplicated found or removed
print(df.head())
df = df[df["bitstring"] != "0" * len(df["bitstring"].iloc[0])]

df['seqs'] = df['Sequence']
df = df[[ x for x in df.columns if x in ['labels', 'seqs']]]
print(df.head())
print(len(df))

#shuffle
df = df.sample(frac=1).reset_index(drop=True)
print(df.head())

#split 
df1 = df.iloc[2000:].reset_index(drop=True)
df2 = df.iloc[1000:2000].reset_index(drop=True)
df3 = df.iloc[:1000].reset_index(drop=True)

# Convert each to Hugging Face Datasets
dataset1 = Dataset.from_pandas(df1)
dataset2 = Dataset.from_pandas(df2)
dataset3 = Dataset.from_pandas(df3)

# Combine into a DatasetDict
dataset_dict = DatasetDict({
    "train": dataset1,
    "valid": dataset2,
    "test": dataset3
})

dataset_dict.push_to_hub("")
