"""
Generate AI images using Stable Diffusion for training dataset.

This script generates AI-generated images that can be used as the
"AI-generated" class for training the detection system.
"""
import os
import sys
import argparse
from tqdm import tqdm
import random


# Sample prompts for diverse image generation
SAMPLE_PROMPTS = [
    # Landscapes
    "beautiful mountain landscape at sunset, realistic photo",
    "ocean beach with palm trees, professional photography",
    "misty forest with rays of sunlight, nature photo",
    "desert canyon with dramatic clouds, landscape photography",
    "snowy winter scene with frozen lake, realistic",

    # Urban
    "modern city skyline at night, photorealistic",
    "old european street with cobblestones, realistic photo",
    "futuristic architecture building, professional photo",
    "busy marketplace with people, documentary style",
    "quiet suburban neighborhood, realistic photography",

    # People
    "portrait of a person smiling, professional headshot",
    "group of friends laughing together, candid photo",
    "athlete in action, sports photography",
    "chef cooking in kitchen, realistic photo",
    "artist working in studio, documentary photography",

    # Animals
    "cute dog playing in park, wildlife photography",
    "majestic lion in savanna, nature photo",
    "colorful bird on branch, wildlife photo",
    "dolphins swimming in ocean, underwater photography",
    "cat sleeping in sunlight, pet photography",

    # Objects
    "vintage camera on wooden table, product photography",
    "fresh vegetables at market, food photography",
    "colorful flowers in vase, still life",
    "old books on shelf, artistic photography",
    "classic car in garage, automotive photography",

    # Abstract/Artistic
    "abstract colorful pattern, digital art",
    "geometric shapes and colors, modern art",
    "watercolor painting of landscape",
    "oil painting of city scene",
    "minimalist composition, fine art photography",
]


def check_dependencies():
    """Check if required packages are installed."""
    try:
        import torch
        return True
    except ImportError:
        print("❌ PyTorch not found!")
        print("\nPlease install PyTorch:")
        print("pip install torch torchvision")
        return False


def generate_with_stable_diffusion(prompts, output_dir, model_id="runwayml/stable-diffusion-v1-5"):
    """
    Generate images using Stable Diffusion.

    Args:
        prompts: List of text prompts
        output_dir: Directory to save images
        model_id: Hugging Face model ID
    """
    try:
        from diffusers import StableDiffusionPipeline
        import torch

        print("\n" + "="*60)
        print("Loading Stable Diffusion model...")
        print("="*60)
        print("(This may take a few minutes on first run)\n")

        # Determine device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")

        if device == "cpu":
            print("\n⚠️  WARNING: Running on CPU will be very slow!")
            print("   For faster generation, use a GPU.")
            print("   Each image may take 30-60 seconds on CPU.\n")

        # Load model
        pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32
        )
        pipe = pipe.to(device)

        # Enable memory optimizations
        if device == "cuda":
            pipe.enable_attention_slicing()

        print("✅ Model loaded successfully!\n")

        # Generate images
        print(f"Generating {len(prompts)} images...")
        os.makedirs(output_dir, exist_ok=True)

        for i, prompt in enumerate(tqdm(prompts)):
            try:
                # Generate image
                image = pipe(
                    prompt,
                    num_inference_steps=50,
                    guidance_scale=7.5
                ).images[0]

                # Save image
                filename = f"ai_generated_{i:04d}.png"
                filepath = os.path.join(output_dir, filename)
                image.save(filepath)

            except Exception as e:
                print(f"\n⚠️  Error generating image {i}: {e}")
                continue

        print(f"\n✅ Generated {len(prompts)} images in {output_dir}")
        return True

    except ImportError as e:
        print("\n❌ Missing required package!")
        print(f"Error: {e}")
        print("\nPlease install:")
        print("pip install diffusers transformers accelerate")
        return False
    except Exception as e:
        print(f"\n❌ Error during generation: {e}")
        return False


def generate_placeholder_images(count, output_dir):
    """
    Generate placeholder images as fallback.

    This creates simple colored noise images when Stable Diffusion isn't available.
    """
    print("\n" + "="*60)
    print("Generating placeholder images...")
    print("="*60)
    print("(For real AI images, install: pip install diffusers transformers)")
    print()

    try:
        from PIL import Image
        import numpy as np

        os.makedirs(output_dir, exist_ok=True)

        for i in tqdm(range(count)):
            # Create random noise image
            img_array = np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8)
            img = Image.fromarray(img_array)

            # Save
            filename = f"placeholder_{i:04d}.png"
            filepath = os.path.join(output_dir, filename)
            img.save(filepath)

        print(f"\n✅ Generated {count} placeholder images")
        print("\n⚠️  NOTE: These are placeholder images, not real AI-generated images!")
        print("For actual AI images:")
        print("1. Install: pip install diffusers transformers accelerate")
        print("2. Re-run this script")
        print("3. Or download AI images from online sources")
        return True

    except Exception as e:
        print(f"\n❌ Error generating placeholders: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Generate AI images using Stable Diffusion',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate 100 images
  python scripts/generate_ai_images.py --count 100

  # Generate 1000 images to specific directory
  python scripts/generate_ai_images.py --count 1000 --output data/ai_generated/

  # Use custom prompts file
  python scripts/generate_ai_images.py --count 500 --prompts my_prompts.txt

Requirements:
  pip install torch torchvision diffusers transformers accelerate

Note: First run will download ~4GB model. Subsequent runs use cached model.
        """
    )

    parser.add_argument('--count', type=int, default=100,
                        help='Number of images to generate (default: 100)')
    parser.add_argument('--output', type=str, default='data/ai_generated',
                        help='Output directory (default: data/ai_generated)')
    parser.add_argument('--prompts', type=str, default=None,
                        help='Text file with prompts (one per line)')
    parser.add_argument('--model', type=str, default='runwayml/stable-diffusion-v1-5',
                        help='Stable Diffusion model ID')
    parser.add_argument('--placeholder', action='store_true',
                        help='Generate placeholder images instead of using Stable Diffusion')

    args = parser.parse_args()

    print("\n" + "="*60)
    print("AI IMAGE GENERATION")
    print("="*60)

    # Load prompts
    if args.prompts and os.path.exists(args.prompts):
        print(f"\nLoading prompts from {args.prompts}...")
        with open(args.prompts, 'r') as f:
            prompts = [line.strip() for line in f if line.strip()]
    else:
        # Use sample prompts, repeating as needed
        prompts = []
        while len(prompts) < args.count:
            prompts.extend(SAMPLE_PROMPTS)
        prompts = prompts[:args.count]

    # Shuffle for variety
    random.shuffle(prompts)

    print(f"Generating {args.count} images...")
    print(f"Output directory: {args.output}")

    # Generate images
    if args.placeholder:
        success = generate_placeholder_images(args.count, args.output)
    else:
        if not check_dependencies():
            print("\nFalling back to placeholder images...")
            print("Install diffusers for real AI-generated images:")
            print("pip install torch diffusers transformers accelerate\n")
            success = generate_placeholder_images(args.count, args.output)
        else:
            success = generate_with_stable_diffusion(prompts, args.output, args.model)

    if success:
        print("\n" + "="*60)
        print("✅ Generation complete!")
        print("="*60)
        print(f"\nImages saved to: {args.output}")
        print("\nNext steps:")
        print("1. Verify dataset: python scripts/verify_dataset.py")
        print("2. Train models: python src/train.py --data_dir data --model_type both")
        print("="*60 + "\n")
    else:
        print("\n" + "="*60)
        print("❌ Generation failed")
        print("="*60)
        print("\nAlternative options:")
        print("1. Fix the errors above and try again")
        print("2. Download AI images manually from online sources")
        print("3. See: python scripts/download_dataset.py --manual")
        print("="*60 + "\n")

    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
