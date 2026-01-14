#!/usr/bin/env python3
"""
VLM Data Loader - Combine sample data (text + images) into VLM-ready format

Usage:
    from scripts.vlm_data_loader import PatientDataLoader

    loader = PatientDataLoader("sample1")  # or full path
    messages = loader.build_vlm_messages()
"""

import json
import base64
import argparse
from pathlib import Path
from typing import Optional, Union

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent

# Default sample data location
SAMPLE_DATA_DIR = PROJECT_ROOT / "sample"


class PatientDataLoader:
    """Load sample data and combine text + images for VLM use"""

    def __init__(self, sample_id_or_path: Union[str, Path]):
        """
        Initialize loader with sample ID or full path.

        Args:
            sample_id_or_path: Either a sample ID (e.g., "sample1")
                              or full path to sample/extracted_data directory
        """
        sample_path = Path(sample_id_or_path)

        # If it's just an ID, look in the sample data directory
        if not sample_path.is_absolute() and not sample_path.exists():
            sample_path = SAMPLE_DATA_DIR / sample_id_or_path

        # If still relative, resolve from project root
        if not sample_path.is_absolute():
            sample_path = PROJECT_ROOT / sample_path

        data_dir = sample_path
        if sample_path.is_dir() and (sample_path / "extracted_data").is_dir():
            data_dir = sample_path / "extracted_data"

        self.sample_dir = data_dir.parent if data_dir.name == "extracted_data" else data_dir
        self.sample_id = self.sample_dir.name
        self.data_dir = data_dir
        self.data = {}
        self._load_all_json()

    def _load_all_json(self):
        """Load all JSON files"""
        json_files = [
            "patient_info.json",
            "anamnesis.json",
            "therapy_goals.json",
            "assessments.json",
            "therapy_program.json",
            "scientific_background.json"
        ]
        for filename in json_files:
            filepath = self.data_dir / filename
            if filepath.exists():
                with open(filepath, 'r', encoding='utf-8') as f:
                    key = filename.replace('.json', '')
                    self.data[key] = json.load(f)

    def _load_image_base64(self, relative_path: str) -> Optional[str]:
        """Load image and convert to base64"""
        img_path = self.data_dir / relative_path
        if not img_path.exists():
            return None
        with open(img_path, 'rb') as f:
            return base64.b64encode(f.read()).decode('utf-8')

    def _get_mime_type(self, path: str) -> str:
        """Get MIME type based on file extension"""
        ext = Path(path).suffix.lower()
        mime_map = {
            '.png': 'image/png',
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.gif': 'image/gif',
            '.webp': 'image/webp'
        }
        return mime_map.get(ext, 'image/png')

    def get_assessment_with_images(self, assessment_type: str = "c_mill_assessment") -> dict:
        """
        Get assessment data with associated images.

        Returns format suitable for VLM API:
        {
            "text": "Description text...",
            "images": [
                {"base64": "...", "mime_type": "image/png", "label": "..."},
                ...
            ]
        }
        """
        if "assessments" not in self.data:
            return {"text": "", "images": []}

        assessment = self.data["assessments"].get(assessment_type, {})
        if not assessment:
            return {"text": "", "images": []}

        # Build text description
        text_parts = [f"## {assessment_type.replace('_', ' ').title()}"]
        text_parts.append(f"\n**Description:** {assessment.get('description', '')}")

        if "interpretation" in assessment:
            text_parts.append(f"\n**Result interpretation:** {assessment['interpretation']}")

        images = []

        # Iterate through all sub-items that may contain images
        for key, value in assessment.items():
            if isinstance(value, dict):
                if "description" in value:
                    text_parts.append(f"\n### {key.replace('_', ' ').title()}")
                    text_parts.append(f"{value['description']}")

                if "interpretation" in value:
                    text_parts.append(f"\n**Interpretation:** {value['interpretation']}")

                # Handle images
                if "images" in value:
                    for img_info in value["images"]:
                        img_base64 = self._load_image_base64(img_info["path"])
                        if img_base64:
                            images.append({
                                "base64": img_base64,
                                "mime_type": self._get_mime_type(img_info["path"]),
                                "label": img_info.get("label", ""),
                                "context": img_info.get("context", "")
                            })
                            text_parts.append(f"\n[Image: {img_info.get('label', '')} - {img_info.get('context', '')}]")

        return {
            "text": "\n".join(text_parts),
            "images": images
        }

    def get_scientific_background_with_images(self) -> dict:
        """Get scientific background data with associated images"""
        if "scientific_background" not in self.data:
            return {"text": "", "images": []}

        sb = self.data["scientific_background"]

        text_parts = [f"## {sb.get('title', 'Scientific Background')}"]
        text_parts.append(f"\n{sb.get('summary', '')}")

        images = []

        for finding in sb.get("key_findings", []):
            text_parts.append(f"\n### {finding.get('topic', '')}")
            text_parts.append(finding.get("description", ""))

            refs = finding.get("references", [])
            if refs:
                text_parts.append(f"\n*References: {', '.join(refs)}*")

            # Handle single image
            if "image" in finding:
                img_info = finding["image"]
                img_base64 = self._load_image_base64(img_info["path"])
                if img_base64:
                    images.append({
                        "base64": img_base64,
                        "mime_type": self._get_mime_type(img_info["path"]),
                        "label": img_info.get("label", ""),
                        "context": img_info.get("context", "")
                    })
                    text_parts.append(f"\n[Image: {img_info.get('label', '')}]")

            # Handle multiple images
            if "images" in finding:
                for img_info in finding["images"]:
                    img_base64 = self._load_image_base64(img_info["path"])
                    if img_base64:
                        images.append({
                            "base64": img_base64,
                            "mime_type": self._get_mime_type(img_info["path"]),
                            "label": img_info.get("label", ""),
                            "context": img_info.get("context", "")
                        })
                        text_parts.append(f"\n[Image: {img_info.get('label', '')}]")

        return {
            "text": "\n".join(text_parts),
            "images": images
        }

    def get_patient_summary(self) -> str:
        """Get patient basic information summary (text only)"""
        parts = []

        if "patient_info" in self.data:
            info = self.data["patient_info"]
            parts.append("## Patient Information")
            parts.append(f"- Diagnosis: {info.get('diagnosis', 'N/A')}")
            parts.append(f"- Report date: {info.get('report_date', 'N/A')}")

        if "anamnesis" in self.data:
            an = self.data["anamnesis"]
            parts.append("\n## Medical History")
            parts.append(an.get("anamnesis_text", ""))
            parts.append(f"\nTherapy history: {an.get('therapy_history', '')}")

        if "therapy_goals" in self.data:
            goals = self.data["therapy_goals"].get("goals", [])
            if goals:
                parts.append("\n## Therapy Goals")
                for g in goals:
                    parts.append(f"- {g.get('description', '')} (Importance: {g.get('importance', 'N/A')})")

        return "\n".join(parts)

    def build_vlm_messages(
        self,
        include_images: bool = True,
        system_prompt: str = None
    ) -> list:
        """
        Build messages in OpenAI VLM API format with interleaved text and images.

        The content is organized like the original PDF:
        - Patient summary (text)
        - For each assessment section: description (text) -> images -> interpretation (text)
        - Scientific background with inline images

        Args:
            include_images: Whether to include images
            system_prompt: Optional system prompt to prepend

        Returns:
            List of messages ready for SGLang/OpenAI API
        """
        messages = []

        # Add system prompt if provided
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        content = []

        # 1. Patient summary
        content.append({
            "type": "text",
            "text": self.get_patient_summary()
        })

        # 2. Assessments with interleaved images
        if "assessments" in self.data:
            content.extend(self._build_assessment_content(include_images))

        # 3. Scientific background with interleaved images
        if "scientific_background" in self.data:
            content.extend(self._build_scientific_background_content(include_images))

        messages.append({"role": "user", "content": content})
        return messages

    def _build_assessment_content(self, include_images: bool) -> list:
        """Build assessment content with interleaved text and images."""
        content = []
        assessments = self.data.get("assessments", {})

        # 6-minute walk test (no images)
        if "six_minute_walk_test" in assessments:
            smwt = assessments["six_minute_walk_test"]
            text = f"\n## Six Minute Walk Test\n"
            text += f"**Description:** {smwt.get('description', '')}\n"
            if "improvement" in smwt:
                text += f"**Improvement:** {smwt['improvement'].get('interpretation', '')}"
            content.append({"type": "text", "text": text})

        # C-Mill assessment with images
        if "c_mill_assessment" in assessments:
            cmill = assessments["c_mill_assessment"]

            # Header
            content.append({
                "type": "text",
                "text": f"\n## C-Mill Assessment\n**Description:** {cmill.get('description', '')}"
            })

            # Process each sub-section with its images
            sections = [
                ("spatial_temporal", "Spatial Temporal Analysis"),
                ("butterfly_diagram", "Butterfly Diagram (CoP Gaitogram)"),
                ("stability_limit", "Stability Limits"),
                ("postural_stability", "Postural Stability"),
            ]

            for key, title in sections:
                if key not in cmill:
                    continue

                section = cmill[key]

                # Section description
                text = f"\n### {title}\n{section.get('description', '')}"
                if "interpretation" in section:
                    text += f"\n\n**Interpretation:** {section['interpretation']}"
                content.append({"type": "text", "text": text})

                # Section images (immediately after description)
                if include_images and "images" in section:
                    for img_info in section["images"]:
                        img_base64 = self._load_image_base64(img_info["path"])
                        if img_base64:
                            # Add image label as text
                            content.append({
                                "type": "text",
                                "text": f"\n[{img_info.get('label', 'Image')}: {img_info.get('context', '')}]"
                            })
                            content.append({
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{self._get_mime_type(img_info['path'])};base64,{img_base64}",
                                    "detail": "high"
                                }
                            })

        return content

    def _build_scientific_background_content(self, include_images: bool) -> list:
        """Build scientific background content with interleaved images."""
        content = []
        sb = self.data.get("scientific_background", {})

        # Header
        content.append({
            "type": "text",
            "text": f"\n## {sb.get('title', 'Scientific Background')}\n{sb.get('summary', '')}"
        })

        # Key findings with images
        for finding in sb.get("key_findings", []):
            # Finding text
            text = f"\n### {finding.get('topic', '')}\n{finding.get('description', '')}"
            refs = finding.get("references", [])
            if refs:
                text += f"\n*References: {', '.join(refs)}*"
            content.append({"type": "text", "text": text})

            # Finding images
            if include_images:
                # Single image
                if "image" in finding:
                    img_info = finding["image"]
                    img_base64 = self._load_image_base64(img_info["path"])
                    if img_base64:
                        content.append({
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{self._get_mime_type(img_info['path'])};base64,{img_base64}",
                                "detail": "high"
                            }
                        })

                # Multiple images
                if "images" in finding:
                    for img_info in finding["images"]:
                        img_base64 = self._load_image_base64(img_info["path"])
                        if img_base64:
                            content.append({
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{self._get_mime_type(img_info['path'])};base64,{img_base64}",
                                    "detail": "high"
                                }
                            })

        return content

    @staticmethod
    def list_patients() -> list:
        """List all available sample IDs from sample data"""
        if not SAMPLE_DATA_DIR.exists():
            return []
        return [d.name for d in SAMPLE_DATA_DIR.iterdir() if d.is_dir()]


def main():
    """CLI entry point"""
    parser = argparse.ArgumentParser(
        description="Load patient data for VLM processing"
    )
    parser.add_argument(
        "sample_id",
        nargs="?",
        default="sample1",
        help="Sample ID (default: sample1)"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all available samples"
    )
    parser.add_argument(
        "--no-images",
        action="store_true",
        help="Exclude images from output"
    )

    args = parser.parse_args()

    if args.list:
        patients = PatientDataLoader.list_patients()
        print(f"Available patients: {patients}")
        return

    loader = PatientDataLoader(args.sample_id)

    print("=" * 60)
    print(f"Sample: {loader.sample_id}")
    print("=" * 60)
    print(loader.get_patient_summary())

    print("\n" + "=" * 60)
    print("C-Mill Assessment Data (with image references)")
    print("=" * 60)
    assessment = loader.get_assessment_with_images("c_mill_assessment")
    print(assessment["text"])
    print(f"\n[Loaded {len(assessment['images'])} images]")
    for img in assessment["images"]:
        print(f"  - {img['label']}: {img['context']}")

    print("\n" + "=" * 60)
    print("Scientific Background (with image references)")
    print("=" * 60)
    sb = loader.get_scientific_background_with_images()
    print(sb["text"])
    print(f"\n[Loaded {len(sb['images'])} images]")

    print("\n" + "=" * 60)
    print("VLM Messages Structure Preview")
    print("=" * 60)
    messages = loader.build_vlm_messages(include_images=not args.no_images)
    text_count = sum(1 for c in messages[0]["content"] if c["type"] == "text")
    img_count = sum(1 for c in messages[0]["content"] if c["type"] == "image_url")
    print(f"Messages contain: {text_count} text blocks, {img_count} images")


if __name__ == "__main__":
    main()
