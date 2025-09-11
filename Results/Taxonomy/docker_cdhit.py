import subprocess
import os
import psutil


def cd_hit(
        fasta_file: str,
        similarity_threshold: float = 0.5,
        n: int = 2, # word size, 5 is faster but 3 is more sensitive
        memory_percentage: float = 0.5,
        output_path: str = None,
    ):
    if output_path is None:
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