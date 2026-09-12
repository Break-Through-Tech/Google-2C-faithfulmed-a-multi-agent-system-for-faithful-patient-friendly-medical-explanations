from __future__ import annotations

from pathlib import Path
import xml.etree.ElementTree as ET

from retrieval.ids import glossary_parent_id
from retrieval.schema import DocumentSection, NormalizedDocument
from retrieval.text_utils import child_text, local_name, clean_glossary_text 


def parse_medlineplus_definitions(
    xml_path: Path,
    *,
    category: str,
    canonical_url: str,
) -> list[NormalizedDocument]:
    """
    Parse one MedlinePlus Definitions of Health Terms XML file.

    Each <term-group> becomes one NormalizedDocument.
    """

    tree = ET.parse(xml_path)
    root = tree.getroot()

    documents: list[NormalizedDocument] = []

    for element in root.iter():

        if local_name(element.tag) != "term-group":
            continue

        term = clean_glossary_text(child_text(element, "term"))
        definition = clean_glossary_text(child_text(element, "definition"))
        definition_source = child_text(element, "source")

        if not term:
            continue

        if not definition:
            # An empty definition is not useful retrieval content.
            continue

        parent_id = glossary_parent_id(
            category=category,
            term=term,
        )

        document = NormalizedDocument(
            parent_id=parent_id,

            title=term,

            sections=[
                DocumentSection(
                    name="definition",
                    blocks=[definition],
                )
            ],

            source="MedlinePlus",
            source_type="glossary_definition",
            url=canonical_url,

            language="English",
            category=category,

            # The definition's NIH source is useful provenance,
            # but should not influence semantic similarity.
            organization=definition_source or None,

            source_file=xml_path.name,
        )

        documents.append(document)

    return documents