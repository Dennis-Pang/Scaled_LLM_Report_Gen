#!/usr/bin/env python3
"""
PDF Image Extractor - Extract images from PDF files using PyMuPDF.

Usage:
    python scripts/pdf_extractor.py input.pdf --output-dir ./images
    python scripts/pdf_extractor.py input.pdf --sample-id sample1
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

try:
    import fitz  # PyMuPDF
except ImportError:
    print("Error: PyMuPDF not installed. Run: pip install pymupdf")
    exit(1)

# Project paths
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent

# Sample data location
SAMPLES_DIR = PROJECT_ROOT / "sample"


def extract_images_from_pdf(
    pdf_path: str,
    output_dir: str = None,
    min_width: int = 100,
    min_height: int = 100,
) -> List[Dict]:
    """
    Extract images from a PDF file.

    Args:
        pdf_path: Path to the PDF file
        output_dir: Directory to save extracted images
        min_width: Minimum image width to extract (filters out logos)
        min_height: Minimum image height to extract

    Returns:
        List of extracted image info dictionaries
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    doc = fitz.open(pdf_path)
    extracted_images = []

    for page_num in range(doc.page_count):
        page = doc[page_num]
        images = page.get_images(full=True)

        # Get page text for context
        text = page.get_text()
        first_lines = [l.strip() for l in text.split('\n')[:10] if l.strip()]

        for img_idx, img in enumerate(images):
            xref = img[0]
            base_image = doc.extract_image(xref)
            width = base_image['width']
            height = base_image['height']
            ext = base_image['ext']

            # Skip small images (logos, decorations)
            if width < min_width or height < min_height:
                continue

            img_info = {
                'page': page_num + 1,
                'index': img_idx,
                'width': width,
                'height': height,
                'format': ext,
                'context': first_lines[:3],
            }

            if output_dir:
                filename = f"page{page_num + 1}_img{img_idx}.{ext}"
                filepath = output_dir / filename
                with open(filepath, 'wb') as f:
                    f.write(base_image['image'])
                img_info['path'] = str(filepath)
                img_info['filename'] = filename

            extracted_images.append(img_info)

    doc.close()
    return extracted_images


def analyze_pdf_structure(pdf_path: str) -> Dict:
    """
    Analyze PDF structure and return page-by-page summary.

    Args:
        pdf_path: Path to the PDF file

    Returns:
        Dictionary with PDF analysis
    """
    pdf_path = Path(pdf_path)
    doc = fitz.open(pdf_path)

    analysis = {
        'total_pages': doc.page_count,
        'pages': []
    }

    for page_num in range(doc.page_count):
        page = doc[page_num]
        images = page.get_images(full=True)
        text = page.get_text()

        # Count content images (exclude small logos)
        content_images = []
        for img in images:
            xref = img[0]
            base_image = doc.extract_image(xref)
            if base_image['width'] >= 100 and base_image['height'] >= 100:
                content_images.append({
                    'width': base_image['width'],
                    'height': base_image['height'],
                    'format': base_image['ext']
                })

        analysis['pages'].append({
            'page_number': page_num + 1,
            'text_preview': text[:500].strip(),
            'total_images': len(images),
            'content_images': content_images,
        })

    doc.close()
    return analysis


def setup_sample_from_pdf(
    pdf_path: str,
    sample_id: str,
    image_mapping: Dict[int, str] = None,
) -> str:
    """
    Extract images from PDF and set up a new sample directory.

    Args:
        pdf_path: Path to the PDF file
        sample_id: New sample ID (e.g., "sample1")
        image_mapping: Optional mapping of page numbers to subdirectories

    Returns:
        Path to the created sample directory
    """
    sample_dir = SAMPLES_DIR / sample_id / "extracted_data"
    sample_dir.mkdir(parents=True, exist_ok=True)

    # Default mapping if not provided
    if image_mapping is None:
        image_mapping = {
            4: "lokomat_assessment",
            5: "lokomat_assessment",
            6: "lokomat_assessment",
            9: "scientific_background",
        }

    doc = fitz.open(pdf_path)

    for page_num in range(doc.page_count):
        page_1based = page_num + 1

        if page_1based not in image_mapping:
            continue

        subdir = image_mapping[page_1based]
        output_dir = sample_dir / subdir
        output_dir.mkdir(parents=True, exist_ok=True)

        page = doc[page_num]
        images = page.get_images(full=True)

        for img_idx, img in enumerate(images):
            xref = img[0]
            base_image = doc.extract_image(xref)

            # Skip small images
            if base_image['width'] < 100 or base_image['height'] < 100:
                continue

            filename = f"page{page_1based}_img{img_idx}.{base_image['ext']}"
            filepath = output_dir / filename

            with open(filepath, 'wb') as f:
                f.write(base_image['image'])

            print(f"Saved: {filepath}")

    doc.close()
    return str(sample_dir)


def main():
    parser = argparse.ArgumentParser(
        description="Extract images from PDF files"
    )
    parser.add_argument(
        "pdf",
        help="Path to the PDF file"
    )
    parser.add_argument(
        "-o", "--output-dir",
        help="Output directory for extracted images"
    )
    parser.add_argument(
        "--sample-id",
        help="Create a new sample directory with extracted images"
    )
    parser.add_argument(
        "--patient-id",
        dest="sample_id",
        help="Deprecated: use --sample-id",
    )
    parser.add_argument(
        "--analyze",
        action="store_true",
        help="Analyze PDF structure without extracting"
    )
    parser.add_argument(
        "--min-size",
        type=int,
        default=100,
        help="Minimum image size to extract (default: 100px)"
    )

    args = parser.parse_args()

    if args.analyze:
        print(f"Analyzing: {args.pdf}")
        analysis = analyze_pdf_structure(args.pdf)

        print(f"\nTotal pages: {analysis['total_pages']}")
        print("=" * 60)

        for page in analysis['pages']:
            print(f"\n--- Page {page['page_number']} ---")
            print(f"Content images: {len(page['content_images'])}")
            if page['content_images']:
                for img in page['content_images']:
                    print(f"  {img['width']}x{img['height']} ({img['format']})")
            print(f"Text preview: {page['text_preview'][:200]}...")

    elif args.sample_id:
        print(f"Setting up sample: {args.sample_id}")
        sample_dir = setup_sample_from_pdf(args.pdf, args.sample_id)
        print(f"\nSample directory created: {sample_dir}")

    else:
        output_dir = args.output_dir or "./extracted_images"
        print(f"Extracting images to: {output_dir}")

        images = extract_images_from_pdf(
            args.pdf,
            output_dir=output_dir,
            min_width=args.min_size,
            min_height=args.min_size,
        )

        print(f"\nExtracted {len(images)} images:")
        for img in images:
            print(f"  Page {img['page']}: {img['width']}x{img['height']} -> {img.get('filename', 'N/A')}")


if __name__ == "__main__":
    main()
