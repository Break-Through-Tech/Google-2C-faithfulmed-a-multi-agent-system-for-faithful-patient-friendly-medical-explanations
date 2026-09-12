from __future__ import annotations

from pathlib import Path
from typing import BinaryIO
import xml.etree.ElementTree as ET
import zipfile

from retrieval.ids import (
    health_topic_parent_id,
    short_hash,
)
from retrieval.schema import (
    DocumentSection,
    NormalizedDocument,
)
from retrieval.text_utils import (
    child_text,
    child_texts,
    extract_summary_blocks,
    local_name,
)


def _parse_xml_source(
    xml_path: Path,
) -> tuple[ET.ElementTree, str]:
    """
    Parse either an uncompressed .xml file or MedlinePlus's
    official compressed .zip distribution.

    Returns:
        (tree, source_filename)
    """

    if xml_path.suffix.lower() != ".zip":
        return ET.parse(xml_path), xml_path.name

    with zipfile.ZipFile(xml_path) as archive:

        xml_members = [
            name
            for name in archive.namelist()
            if name.lower().endswith(".xml")
        ]

        if not xml_members:
            raise ValueError(
                f"No XML file found inside {xml_path}"
            )

        # The bulk download should contain the Health Topic XML.
        # If multiple XML files appear, choose the largest.
        xml_member = max(
            xml_members,
            key=lambda name: archive.getinfo(name).file_size,
        )

        with archive.open(xml_member) as file_obj:
            tree = ET.parse(file_obj)

        return tree, f"{xml_path.name}:{xml_member}"


def _mesh_terms(topic: ET.Element) -> list[str]:

    terms: list[str] = []

    for element in topic.iter():

        if local_name(element.tag) != "descriptor":
            continue

        text = " ".join(element.itertext()).strip()

        if text:
            terms.append(text)

    return terms


def _related_topics(topic: ET.Element) -> list[str]:

    values: list[str] = []

    for child in topic:

        if local_name(child.tag) != "related-topic":
            continue

        text = " ".join(child.itertext()).strip()

        if text:
            values.append(text)

    return values


def parse_medlineplus_health_topics(
    xml_path: Path,
) -> list[NormalizedDocument]:
    """
    Parse the official MedlinePlus Health Topic XML.

    Only English topics are retained for the current FaithfulMed
    English-language index.
    """

    tree, source_filename = _parse_xml_source(xml_path)

    root = tree.getroot()

    source_generated_at = (
        root.attrib.get("date-generated")
        or root.attrib.get("dategenerated")
    )

    documents: list[NormalizedDocument] = []

    for topic in root:

        if local_name(topic.tag) != "health-topic":
            continue

        language = topic.attrib.get(
            "language",
            "English",
        )

        if language.lower() != "english":
            continue

        title = topic.attrib.get("title", "").strip()
        topic_id = topic.attrib.get("id", "").strip()
        url = topic.attrib.get("url", "").strip()

        if not title:
            continue

        if topic_id:
            parent_id = health_topic_parent_id(topic_id)
        else:
            # Defensive fallback. In normal MedlinePlus data,
            # health topics are expected to have IDs.
            parent_id = (
                "medlineplus:health_topic:"
                + short_hash(title, url)
            )

        summary_element = None

        for child in topic:
            if local_name(child.tag) == "full-summary":
                summary_element = child
                break

        from retrieval.text_utils import (element_text, parse_summary_html,)
        
        raw_summary_html = element_text(summary_element)
        parsed_sections = parse_summary_html(raw_summary_html)

        if not parsed_sections:
            continue

        sections = [
            DocumentSection(
                name=section_name,
                blocks=blocks,
            )
            for section_name, blocks
            in parsed_sections
        ]

        # A topic without human-readable summary text has nothing
        # useful to embed into the current RAG corpus.
        if not summary_blocks:
            continue

        synonyms = child_texts(
            topic,
            "also-called",
        )

        see_references = child_texts(
            topic,
            "see-reference",
        )

        groups = child_texts(
            topic,
            "group",
        )

        primary_institute = child_text(
            topic,
            "primary-institute",
        )

        document = NormalizedDocument(
            parent_id=parent_id,

            title=title,

            # Important:
            # MedlinePlus does NOT provide explicit Symptoms /
            # Treatment / Diagnosis headings inside full-summary.
            # So don't invent them.
            sections=sections,

            source="MedlinePlus",
            source_type="health_topic",
            url=url,

            language=language,

            topic_id=topic_id or None,

            synonyms=synonyms,
            see_references=see_references,
            mesh_terms=_mesh_terms(topic),

            groups=groups,
            related_topics=_related_topics(topic),

            primary_institute=(
                primary_institute or None
            ),

            organization=(
                primary_institute or None
            ),

            meta_description=(
                topic.attrib.get("meta-desc")
                or None
            ),

            date_created=(
                topic.attrib.get("date-created")
                or None
            ),

            source_generated_at=(
                source_generated_at
                or None
            ),

            source_file=source_filename,
        )

        documents.append(document)

    return documents