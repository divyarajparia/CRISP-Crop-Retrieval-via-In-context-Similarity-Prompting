"""
run_crop.py – Command-line interface for Text-Guided Cropping (CRISP Task 2)

Usage examples
--------------
# Minimal (heuristic mode, no GPU needed):
python run_crop.py \\
    --image photo.jpg \\
    --prompt "Focus on the person on the left" \\
    --output cropped.jpg

# With CLIP retrieval from a GAICD database:
python run_crop.py \\
    --image photo.jpg \\
    --prompt "Wide cinematic crop" \\
    --gaicd_root ./data/GAICD \\
    --output cropped.jpg

# Full pipeline (CLIP + Mantis-8B VLM):
python run_crop.py \\
    --image photo.jpg \\
    --prompt "Crop for Instagram square post" \\
    --gaicd_root ./data/GAICD \\
    --vlm \\
    --vlm_model TIGER-Lab/Mantis-8B-Idefics2 \\
    --quant 4 \\
    --S 10 --R 5 --L 2 \\
    --output cropped.jpg

# Resize to a fixed output size (backwards-compatible):
python run_crop.py \\
    --image photo.jpg \\
    --prompt "Emphasize the mountain, not the lake" \\
    --crop_size 640 360 \\
    --output cropped.jpg
"""

import argparse
import logging
import sys
from pathlib import Path

from PIL import Image

from text_guided_crop import (
    CropBox,
    TextGuidedCropper,
    build_database_from_gaicd,
    load_clip_model,
    load_vlm_model,
    VRAM_GUIDE,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Text-Guided Cropping (CRISP Task 2)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # --- Required ---------------------------------------------------------
    p.add_argument("--image",  required=True, help="Path to input image.")
    p.add_argument("--prompt", required=True,
                   help='Natural language crop intent, e.g. "Focus on the left person".')
    p.add_argument("--output", required=True, help="Path to save cropped image.")

    # --- Optional output size (backwards-compatible) ---------------------
    p.add_argument("--crop_size", type=int, nargs=2, metavar=("W", "H"),
                   default=None,
                   help="Resize output to this size (pixels). "
                        "If omitted the crop is saved at its natural size.")

    # --- CLIP settings ---------------------------------------------------
    p.add_argument("--clip_model",     default="ViT-B-32",
                   help="OpenCLIP model architecture.")
    p.add_argument("--clip_pretrained", default="openai",
                   help="OpenCLIP pretrained weights.")
    p.add_argument("--no_clip", action="store_true",
                   help="Disable CLIP (heuristic fallback only).")

    # --- VLM settings ----------------------------------------------------
    p.add_argument("--vlm", action="store_true",
                   help="Enable the VLM for crop generation.")
    p.add_argument("--vlm_model",
                   default="TIGER-Lab/Mantis-8B-Idefics2",
                   help="HuggingFace VLM model ID. "
                        "For 8 GB VRAM: use --quant 4 (any 8B model) "
                        "or 'Qwen/Qwen2-VL-2B-Instruct' (no quant needed).")
    p.add_argument("--quant", type=int, choices=[4, 8], default=None,
                   metavar="{4,8}",
                   help="Quantize the VLM to reduce VRAM usage. "
                        "4 = NF4 ~5 GB for 8B models (recommended for 8 GB VRAM). "
                        "8 = int8 ~10 GB for 8B models. "
                        "Requires: pip install bitsandbytes accelerate")
    p.add_argument("--no_fp16", action="store_true",
                   help="Load VLM in float32 (only relevant without --quant).")
    p.add_argument("--vram_guide", action="store_true",
                   help="Print VRAM requirements table and exit.")

    # --- Database (GAICD) ------------------------------------------------
    p.add_argument("--gaicd_root", default=None,
                   help="Path to GAICD dataset root for ICL retrieval. "
                        "If omitted no database is loaded.")
    p.add_argument("--db_max_images", type=int, default=500,
                   help="Maximum number of GAICD images to load.")

    # --- Pipeline hyperparameters ----------------------------------------
    p.add_argument("--S", type=int, default=10,
                   help="Number of ICL examples to retrieve.")
    p.add_argument("--R", type=int, default=5,
                   help="Number of candidate crops per VLM call.")
    p.add_argument("--L", type=int, default=2,
                   help="Number of iterative refinement steps.")
    p.add_argument("--text_rerank_weight", type=float, default=0.5,
                   help="Weight for text-crop re-ranking in retrieval [0,1].")
    p.add_argument("--coord_range", type=float, nargs=2,
                   metavar=("LO", "HI"), default=[1.0, 1000.0],
                   help="Coordinate normalisation range.")

    # --- Scorer weights --------------------------------------------------
    p.add_argument("--vila_weight", type=float, default=0.3,
                   help="Aesthetic score weight in composite scorer.")
    p.add_argument("--text_weight", type=float, default=0.5,
                   help="CLIP text-alignment weight in composite scorer.")
    p.add_argument("--area_weight", type=float, default=0.2,
                   help="Area preservation weight in composite scorer.")

    # --- Misc ------------------------------------------------------------
    p.add_argument("--device", default=None,
                   help="Torch device ('cuda', 'cpu'). Auto-detected if omitted.")
    p.add_argument("--verbose", action="store_true",
                   help="Enable DEBUG logging.")

    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    if args.vram_guide:
        print(VRAM_GUIDE)
        return

    # ---- Load image -----------------------------------------------------
    img_path = Path(args.image)
    if not img_path.exists():
        logger.error("Image not found: %s", img_path)
        sys.exit(1)
    image = Image.open(img_path).convert("RGB")
    logger.info("Loaded image %s  (%dx%d)", img_path.name, *image.size)

    # ---- Load CLIP -------------------------------------------------------
    clip_model, clip_preprocess = None, None
    if not args.no_clip:
        logger.info("Loading CLIP (%s / %s) ...",
                    args.clip_model, args.clip_pretrained)
        clip_model, clip_preprocess = load_clip_model(
            args.clip_model, args.clip_pretrained
        )
        if clip_model is None:
            logger.warning("CLIP unavailable; running without CLIP features.")

    # ---- Load VLM --------------------------------------------------------
    vlm_model, vlm_processor = None, None
    if args.vlm:
        logger.info("Loading VLM (%s) ...", args.vlm_model)
        vlm_model, vlm_processor = load_vlm_model(
            model_name=args.vlm_model,
            device=args.device or "cuda",
            use_fp16=not args.no_fp16,
            quantize=args.quant,
            print_vram_guide=True,
        )

    # ---- Build cropper ---------------------------------------------------
    cropper = TextGuidedCropper(
        vlm_model=vlm_model,
        vlm_processor=vlm_processor,
        clip_model=clip_model,
        clip_preprocess=clip_preprocess,
        device=args.device,
        S=args.S,
        R=args.R,
        L=args.L,
        text_rerank_weight=args.text_rerank_weight,
        coord_range=tuple(args.coord_range),
        vila_weight=args.vila_weight,
        text_weight=args.text_weight,
        area_weight=args.area_weight,
    )

    # ---- Load retrieval database (optional) ------------------------------
    if args.gaicd_root:
        logger.info("Loading GAICD database from '%s' ...", args.gaicd_root)
        db_images, db_crops = build_database_from_gaicd(
            gaicd_root=args.gaicd_root,
            split="train",
            max_images=args.db_max_images,
            coord_range=tuple(args.coord_range),
        )
        if db_images:
            cropper.add_examples(db_images, db_crops)
            logger.info("Database loaded: %d images.", len(db_images))
        else:
            logger.warning("Database empty – proceeding without ICL examples.")

    # ---- Run cropping ----------------------------------------------------
    logger.info('Cropping with prompt: "%s"', args.prompt)
    result = cropper.crop(image, args.prompt, return_all_candidates=False)

    # ---- Save output -----------------------------------------------------
    output_img = result.cropped_image
    if args.crop_size is not None:
        output_img = output_img.resize(tuple(args.crop_size), Image.LANCZOS)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    output_img.save(out_path)

    b = result.crop_box
    logger.info(
        "Saved %s  (%dx%d)  box=(%d,%d,%d,%d)  score=%.4f",
        out_path, *output_img.size,
        b.x1, b.y1, b.x2, b.y2,
        result.composite_score,
    )


if __name__ == "__main__":
    main()
