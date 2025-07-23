# Uniforming Process: 

- Convert to RGB (no alpha)
- Saved as JPG at quality 95
- Filled transparent areas with white background


# Images

vggface2_train_uniform: 1918121
vggface2_test_uniform: 169396
img_align_celeba_uniform: 1526


# Next Steps: CLIP Labelling

Define & store attributes (100–200 face features). Generate a prompt template or numeric vector schema per image.

Create train/val/test splits (don’t touch test during training).

Extract conditioning

Text route: auto-build a descriptive prompt from attrs.

Numeric route: MLP → embedding.

Fine‑tune Stable Diffusion (LoRA) on train_uniform using those conditionings.

Evaluate on the held‑out set: attribute match rate, CLIP similarity, FID/KID.

Build a minimal app: form → attrs → conditioning → generate N images → pick best (optionally with an automatic scorer).
