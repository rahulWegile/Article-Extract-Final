from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

_REGISTRY_PATH = Path(__file__).parent / "registry_data.json"

_REQUIRED_FIELDS = (
    "template_id",
    "newspaper_name",
    "edition",
    "language",
    "name_ocr_variants",
    "masthead_region",
    "date_region",
    "edition_region",
    "edition_patterns",
    "edition_primary_token",
)


@dataclass
class MastheadTemplate:
    template_id: str
    newspaper_name: str
    edition: str
    language: str

    name_ocr_variants: list[str]

    masthead_region: tuple[float, float, float, float]
    date_region: tuple[float, float, float, float]
    edition_region: tuple[float, float, float, float]

    edition_patterns: list[str]
    edition_primary_token: str

    date_format_hint: str = ""
    sample_source: str = ""
    verified: bool = False
    notes: str = ""
    confirmation_count: int = 0


def _validate(entry: dict, index: int) -> None:
    missing = [key for key in _REQUIRED_FIELDS if key not in entry]

    if missing:
        raise ValueError(
            f"Masthead registry entry #{index} "
            f"({entry.get('template_id', '<unknown>')}) is missing "
            f"required field(s): {', '.join(missing)}"
        )


def load_registry(path: Path | None = None) -> list[MastheadTemplate]:
    registry_path = path or _REGISTRY_PATH

    with open(registry_path, "r", encoding="utf-8") as f:
        raw_entries = json.load(f)

    if not isinstance(raw_entries, list):
        raise ValueError(
            f"Masthead registry at {registry_path} must be a JSON array."
        )

    templates = []

    for index, entry in enumerate(raw_entries):
        _validate(entry, index)

        templates.append(
            MastheadTemplate(
                template_id=entry["template_id"],
                newspaper_name=entry["newspaper_name"],
                edition=entry["edition"],
                language=entry["language"],
                name_ocr_variants=list(entry["name_ocr_variants"]),
                masthead_region=tuple(entry["masthead_region"]),
                date_region=tuple(entry["date_region"]),
                edition_region=tuple(entry["edition_region"]),
                edition_patterns=list(entry["edition_patterns"]),
                edition_primary_token=entry["edition_primary_token"],
                date_format_hint=entry.get("date_format_hint", ""),
                sample_source=entry.get("sample_source", ""),
                verified=bool(entry.get("verified", False)),
                notes=entry.get("notes", ""),
                confirmation_count=int(entry.get("confirmation_count", 0)),
            )
        )

    return templates


def _template_to_dict(template: MastheadTemplate) -> dict:
    return {
        "template_id": template.template_id,
        "newspaper_name": template.newspaper_name,
        "edition": template.edition,
        "language": template.language,
        "name_ocr_variants": template.name_ocr_variants,
        "masthead_region": list(template.masthead_region),
        "date_region": list(template.date_region),
        "edition_region": list(template.edition_region),
        "edition_patterns": template.edition_patterns,
        "edition_primary_token": template.edition_primary_token,
        "date_format_hint": template.date_format_hint,
        "sample_source": template.sample_source,
        "verified": template.verified,
        "notes": template.notes,
        "confirmation_count": template.confirmation_count,
    }


def save_registry(
    templates: list[MastheadTemplate],
    path: Path | None = None,
) -> None:
    registry_path = path or _REGISTRY_PATH

    with open(registry_path, "w", encoding="utf-8") as f:
        json.dump(
            [_template_to_dict(template) for template in templates],
            f,
            indent=4,
            ensure_ascii=False,
        )
