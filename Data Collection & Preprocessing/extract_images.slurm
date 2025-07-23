#!/bin/bash
#SBATCH --job-name=extract_remaining
#SBATCH --output=extract_remaining.out
#SBATCH --error=extract_remaining.err
#SBATCH --time=02:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --partition=main

cd /scratch/sg1806/forensic_faces/supreme_dataset || { echo "Directory not found"; exit 1; }

echo "Extracting VGGFace2 TEST set..."
mkdir -p vggface2_test
tar -xvzf vggface2_test.tar.gz -C vggface2_test

echo "Extracting CelebA aligned images..."
mkdir -p img_align_celeba
tar -xvzf img_align_celebA.tar.gz -C img_align_celeba

echo "Extraction complete."
