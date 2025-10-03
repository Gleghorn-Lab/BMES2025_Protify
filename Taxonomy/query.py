import requests
from Bio import Entrez
from Bio import SeqIO
import io
import time

def _map_uniprot_id(uniprot_id: str, target_db: str) -> str | None:
    """Helper function to run a mapping job on the UniProt REST API."""
    mapping_url = "https://rest.uniprot.org/idmapping/run"
    mapping_params = {
        "ids": uniprot_id,
        "from": "UniProtKB_AC-ID",
        "to": target_db,
    }
    
    try:
        job_submission = requests.post(mapping_url, data=mapping_params)
        job_submission.raise_for_status()
        job_id = job_submission.json()["jobId"]

        # Poll for results
        results_url = f"https://rest.uniprot.org/idmapping/status/{job_id}"
        while True:
            response = requests.get(results_url)
            response.raise_for_status()
            status = response.json()
            if "results" in status or status.get("jobStatus") != "RUNNING":
                break
            time.sleep(1)

        results_data = requests.get(f"https://rest.uniprot.org/idmapping/results/{job_id}").json()
        if not results_data.get("results"):
            return None
            
        return results_data["results"][0]["to"]

    except requests.exceptions.RequestException as e:
        print(f"Error during UniProt mapping to {target_db}: {e}")
        return None


def get_dna_sequence_from_uniprot(uniprot_id: str, email: str) -> str | None:
    """
    Fetches the reference DNA coding sequence (CDS) for a given UniProt ID.

    This function uses a direct, two-query approach:
    1. Maps the UniProt ID to its source Nucleotide ID (e.g., CU633730).
    2. Maps the UniProt ID to its primary GenBank Protein ID (e.g., CAR68305.1).
    3. Fetches the full nucleotide record and finds the specific CDS feature that
       matches the protein ID, extracting its DNA sequence.
       
    This method avoids the fragile Entrez elink step and is more robust.

    Args:
        uniprot_id: The UniProt accession ID (e.g., "A9VM99").
        email: Your email address (required by NCBI for Entrez access).

    Returns:
        The DNA sequence as a string if found, otherwise None.
    """
    Entrez.email = email

    # Step 1: Get the source Nucleotide ID and the specific Protein ID from UniProt
    print(f"-> Starting process for {uniprot_id}...")
    nucleotide_id = _map_uniprot_id(uniprot_id, "EMBL-GenBank-DDBJ")
    protein_id = _map_uniprot_id(uniprot_id, "EMBL-GenBank-DDBJ_CDS")
    
    if not nucleotide_id:
        print(f"Error: Could not map {uniprot_id} to a Nucleotide record.")
        return None
    if not protein_id:
        # Sometimes the CDS mapping can be to RefSeq, try that as a fallback.
        protein_id = _map_uniprot_id(uniprot_id, "RefSeq_Protein")
        if not protein_id:
            print(f"Error: Could not map {uniprot_id} to a Protein record.")
            return None

    print(f"Step 1: Mapped {uniprot_id} to Nucleotide ID: {nucleotide_id}")
    print(f"Step 2: Mapped {uniprot_id} to Protein ID: {protein_id}")

    # Step 2: Fetch Nucleotide Record and Extract the specific CDS
    try:
        gb_handle = Entrez.efetch(db="nuccore", id=nucleotide_id, rettype="gb", retmode="text")
        gb_record = SeqIO.read(io.StringIO(gb_handle.read()), "genbank")
        gb_handle.close()
        print(f"Step 3: Fetched GenBank record '{gb_record.id}'")

        # Find the correct CDS feature that corresponds to our protein of interest
        for feature in gb_record.features:
            if feature.type == "CDS" and "protein_id" in feature.qualifiers:
                # Check against all protein IDs associated with this feature
                if protein_id in feature.qualifiers["protein_id"]:
                    print(f"Step 4: Found matching CDS feature for {protein_id}. Extracting sequence...")
                    return str(feature.extract(gb_record.seq))

        print(f"Error: Could not find a matching CDS feature for {protein_id} in record {nucleotide_id}.")
        return None

    except Exception as e:
        print(f"Error fetching or parsing GenBank record {nucleotide_id}: {e}")
        return None


# --- Example Usage ---
if __name__ == "__main__":
    YOUR_EMAIL = "lhallee@udel.edu"
    
    # Your failing example from Swiss-Prot
    uniprot_accession = "A9VM99"
    
    dna_sequence = get_dna_sequence_from_uniprot(uniprot_accession, YOUR_EMAIL)

    if dna_sequence:
        print(f"\n✅ Successfully retrieved DNA sequence for {uniprot_accession}.")
        print(f"Sequence (first 75 bp): {dna_sequence[:75]}...")
        print(f"Sequence (last 75 bp):  ...{dna_sequence[-75:]}")
        print(f"Total length: {len(dna_sequence)} bp")
    else:
        print(f"\n❌ Could not retrieve DNA sequence for {uniprot_accession}.")