#!/bin/bash
#SBATCH --job-name=uniformize_imgs
#SBATCH --output=uniformize_images.out
#SBATCH --error=uniformize_images.err
#SBATCH --time=04:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --partition=main

cd /scratch/sg1806/forensic_faces/supreme_dataset || { echo " dir not found"; exit 1; }

echo "Installing Python deps into user site..."
# this will install pillow and tqdm into ~/.local/lib, which Python picks up automatically
python3 -m pip install --upgrade --user pillow tqdm

echo "Starting image uniformization…"

cat > uniformize_images.py << 'EOF'
import os
from PIL import Image
folders = ['vggface2_train','vggface2_test','img_align_celeba','faces']
size = (512,512)
for folder in folders:
    dst = folder + '_uniform'
    for root, _, files in os.walk(folder):
        rel = os.path.relpath(root, folder)
        outdir = os.path.join(dst, rel)
        os.makedirs(outdir, exist_ok=True)
        for fn in files:
            if fn.lower().endswith(('.jpg','.jpeg','.png')):
                inp = os.path.join(root, fn)
                try:
                    im = Image.open(inp).convert('RGB')
                    im = im.resize(size, Image.LANCZOS)
                    base = os.path.splitext(fn)[0] + '.jpg'
                    im.save(os.path.join(outdir, base), 'JPEG', quality=95)
                except Exception as e:
                    print(f"ERR {inp}: {e}")
print("✅ Done uniformizing all folders")
EOF

python3 uniformize_images.py