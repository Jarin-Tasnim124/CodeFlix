from __future__ import annotations

import copy
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET
from xml.sax.saxutils import escape


TEMPLATE_PATH = Path(r"g:\New folder\Portfolio 2.docx")
OUTPUT_PATH = Path(r"c:\Users\MrWolf\Desktop\CodeFlix\Portfolio 2 - AI Finder Updated.docx")

W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
ET.register_namespace("w", W_NS)


def w_tag(name: str) -> str:
    return f"{{{W_NS}}}{name}"


def make_run(text: str, bold: bool = False) -> ET.Element:
    run = ET.Element(w_tag("r"))
    if bold:
        r_pr = ET.SubElement(run, w_tag("rPr"))
        ET.SubElement(r_pr, w_tag("b"))
    t = ET.SubElement(run, w_tag("t"))
    if text.startswith(" ") or text.endswith(" "):
        t.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
    t.text = text
    return run


def make_paragraph(
    text: str = "",
    style: str | None = None,
    bold: bool = False,
    page_break_before: bool = False,
) -> ET.Element:
    p = ET.Element(w_tag("p"))
    if style or page_break_before:
        p_pr = ET.SubElement(p, w_tag("pPr"))
        if style:
            p_style = ET.SubElement(p_pr, w_tag("pStyle"))
            p_style.set(w_tag("val"), style)
        if page_break_before:
            ET.SubElement(p_pr, w_tag("pageBreakBefore"))
    if text:
        p.append(make_run(text, bold=bold))
    else:
        ET.SubElement(p, w_tag("r"))
    return p


def make_table(headers: list[str], rows: list[list[str]]) -> ET.Element:
    tbl = ET.Element(w_tag("tbl"))

    tbl_pr = ET.SubElement(tbl, w_tag("tblPr"))
    tbl_style = ET.SubElement(tbl_pr, w_tag("tblStyle"))
    tbl_style.set(w_tag("val"), "TableGrid")
    tbl_w = ET.SubElement(tbl_pr, w_tag("tblW"))
    tbl_w.set(w_tag("w"), "0")
    tbl_w.set(w_tag("type"), "auto")
    borders = ET.SubElement(tbl_pr, w_tag("tblBorders"))
    for side in ["top", "left", "bottom", "right", "insideH", "insideV"]:
        border = ET.SubElement(borders, w_tag(side))
        border.set(w_tag("val"), "single")
        border.set(w_tag("sz"), "4")
        border.set(w_tag("space"), "0")
        border.set(w_tag("color"), "auto")

    tbl_grid = ET.SubElement(tbl, w_tag("tblGrid"))
    col_count = max(len(headers), max((len(r) for r in rows), default=0))
    for _ in range(col_count):
        grid_col = ET.SubElement(tbl_grid, w_tag("gridCol"))
        grid_col.set(w_tag("w"), "2400")

    all_rows = [headers] + rows
    for row_index, row_values in enumerate(all_rows):
        tr = ET.SubElement(tbl, w_tag("tr"))
        for cell_value in row_values:
            tc = ET.SubElement(tr, w_tag("tc"))
            tc_pr = ET.SubElement(tc, w_tag("tcPr"))
            tc_w = ET.SubElement(tc_pr, w_tag("tcW"))
            tc_w.set(w_tag("w"), "2400")
            tc_w.set(w_tag("type"), "dxa")
            tc_p = ET.SubElement(tc, w_tag("p"))
            if row_index == 0:
                tc_p.append(make_run(cell_value, bold=True))
            else:
                tc_p.append(make_run(cell_value))
    return tbl


def build_document_xml(sect_pr: ET.Element | None) -> bytes:
    root = ET.Element(w_tag("document"))
    body = ET.SubElement(root, w_tag("body"))

    body.append(make_paragraph("Portfolio 2 - Advanced Prototype & Test Logs", style="Title"))
    body.append(make_paragraph("P Number: P2863788"))
    body.append(make_paragraph("Name: Tauheed Ahmed Nabil"))
    body.append(make_paragraph("Individual Component: AI Finder Advanced Features & OMDb Integration"))
    body.append(make_paragraph("Team Name: CodeFlix (Team-1)"))
    body.append(make_paragraph("GitHub Link: https://github.com/tauheednabil-TAN/CodeFlix.git"))
    body.append(make_paragraph("Prepared from Implemented_features.md and the current project source files."))

    body.append(make_paragraph("", page_break_before=True))
    body.append(make_paragraph("Contents", style="Heading1"))
    for line in [
        "1. SQLite Database Login Details",
        "2. Advanced Prototype Feature Summary",
        "3. Implementation Check Logs",
        "4. End-to-End Flow",
        "5. Primary Files and Code References",
    ]:
        body.append(make_paragraph(line))

    body.append(make_paragraph("", page_break_before=True))
    body.append(make_paragraph("1. SQLite Database Login Details", style="Heading1"))
    body.append(
        make_paragraph(
            "The current CodeFlix system uses a local SQLite database instead of SQL Server, so no username or password is required for normal use."
        )
    )
    body.append(
        make_table(
            ["Field", "Value"],
            [
                ["Database Type", "SQLite"],
                ["Database File", "movies.db"],
                ["Server Name", "Local SQLite - movies.db"],
                ["User Name", "Not Needed"],
                ["Password", "Not Needed"],
                ["Recommended Viewer", "DB Browser for SQLite"],
            ],
        )
    )
    body.append(make_paragraph("Current database tables", style="Heading2"))
    body.append(
        make_table(
            ["Table Name", "Purpose"],
            [
                [
                    "enhanced_movies",
                    "Main application table that stores movie records, watched status, rating, review, plot details, streaming flags and ticket availability.",
                ],
                [
                    "recommendation_feedback",
                    "Stores AI Finder feedback such as like and dislike votes, the source query and timestamps for personalization.",
                ],
                [
                    "app_metadata",
                    "Stores lightweight application metadata such as database version and seeded state.",
                ],
                [
                    "sqlite_sequence",
                    "SQLite system table created automatically for AUTOINCREMENT fields. This should not be edited manually.",
                ],
            ],
        )
    )
    body.append(
        make_paragraph(
            "Important note: the original Portfolio 2 template mentioned only three tables. The current prototype now includes a fourth functional table called recommendation_feedback for AI Finder personalization."
        )
    )

    body.append(make_paragraph("2. Advanced Prototype Feature Summary", style="Heading1"))
    body.append(
        make_paragraph(
            "This section is based on Implemented_features.md and updated code references from the current repository."
        )
    )
    body.append(
        make_table(
            ["Feature", "What Was Implemented", "Main Code References", "Status"],
            [
                [
                    "Explainable Recommendations",
                    "Each recommendation card shows a short 'Why this movie?' reason based on mood, genre, actor or theme signals from the user query.",
                    "app.py:1106, app.py:1596, recommender.py:247, recommender.py:295, recommender.py:352",
                    "Implemented",
                ],
                [
                    "Like/Dislike Feedback Loop",
                    "Recommendation cards include like and dislike controls; votes are stored in SQLite and used to rerank future suggestions.",
                    "app.py:1324, app.py:1338, app.py:1596, app.py:1737, recommender.py:116, recommender.py:186",
                    "Implemented",
                ],
                [
                    "Prompt Templates (Quick Actions)",
                    "One-click prompts help the user request mood-based, similar-title and anime-only recommendations without typing a full prompt.",
                    "app.py:1452, app.py:1543, app.py:2280, app.py:2412",
                    "Implemented",
                ],
                [
                    "Smart Fallback Recommender",
                    "If Gemini is unavailable or returns unusable output, the system falls back to TF-IDF recommendations and still shows meaningful results.",
                    "app.py:1467, app.py:1490, app.py:1543, recommender.py:366",
                    "Implemented",
                ],
                [
                    "Recommendation Filters",
                    "The AI Finder supports year range, minimum IMDb rating and content type filters to constrain both AI and fallback results.",
                    "app.py:1174, app.py:1200, app.py:1277, app.py:1490, app.py:2284",
                    "Implemented",
                ],
            ],
        )
    )

    body.append(make_paragraph("3. Implementation Check Logs", style="Heading1"))
    body.append(
        make_paragraph(
            "The environment used to prepare this document did not have Streamlit, requests or pandas installed, so the logs below are written as implementation checks verified from source code rather than live UI execution."
        )
    )

    check_entries = [
        (
            "Explainable Recommendations",
            [
                ["Description of Item to Be Tested", "Verify that recommendation cards display a short explanation showing why a movie was suggested."],
                ["Test Data / Trigger", "Example prompt: 'Sad mood, recommend 3 sci-fi movies'."],
                ["Expected Result", "The system should render recommendation cards with reasons such as mood match, genre match, actor match or theme similarity."],
                ["Evidence in Code", "app.py:1106 builds explainable results; app.py:1596 renders them; recommender.py:295 and recommender.py:352 generate explanation text."],
                ["Status", "Verified in code"],
            ],
        ),
        (
            "Like/Dislike Feedback Loop",
            [
                ["Description of Item to Be Tested", "Verify that the AI Finder stores user feedback and uses it to influence later ranking."],
                ["Test Data / Trigger", "User clicks Like or Dislike on a recommendation card."],
                ["Expected Result", "A feedback row should be written to SQLite and future recommendations should be reranked using the feedback profile."],
                ["Evidence in Code", "app.py:1338 saves votes; app.py:1737 creates recommendation_feedback; recommender.py:116 and recommender.py:186 build and use the feedback profile."],
                ["Status", "Verified in code"],
            ],
        ),
        (
            "Prompt Templates (Quick Actions)",
            [
                ["Description of Item to Be Tested", "Verify that mood, similar-title and anime-only quick actions submit through the normal AI Finder pipeline."],
                ["Test Data / Trigger", "Click the quick action buttons from the AI Finder page."],
                ["Expected Result", "The selected quick action should generate a ready-made prompt and run the same recommendation flow as typed input."],
                ["Evidence in Code", "app.py:1452 builds quick prompts; app.py:1543 handles the chat turn; app.py:2412-2418 triggers the quick action buttons."],
                ["Status", "Verified in code"],
            ],
        ),
        (
            "Smart Fallback Recommender",
            [
                ["Description of Item to Be Tested", "Verify that the system falls back to TF-IDF recommendations when Gemini fails or returns no usable titles."],
                ["Test Data / Trigger", "Simulate a blank Gemini reply, error response or zero extracted titles."],
                ["Expected Result", "The user should still receive a meaningful recommendation message and fallback recommendation cards."],
                ["Evidence in Code", "app.py:1467 detects fallback conditions; app.py:1490 builds fallback recommendations; recommender.py:366 provides TF-IDF ranking."],
                ["Status", "Verified in code"],
            ],
        ),
        (
            "Recommendation Filters",
            [
                ["Description of Item to Be Tested", "Verify that AI Finder filters affect prompt instructions and recommendation output."],
                ["Test Data / Trigger", "Set year range, minimum IMDb rating and content type before requesting recommendations."],
                ["Expected Result", "Returned titles should respect the selected filters in both the Gemini path and the fallback path."],
                ["Evidence in Code", "app.py:1174 defines defaults; app.py:1200 summarizes filters; app.py:1277 applies filters; app.py:1490 reuses filters in fallback ranking."],
                ["Status", "Verified in code"],
            ],
        ),
    ]

    for title, rows in check_entries:
        body.append(make_paragraph(title, style="Heading2"))
        body.append(make_table(["Field", "Details"], rows))

    body.append(make_paragraph("4. End-to-End Flow", style="Heading1"))
    for line in [
        "1. The user enters a prompt or clicks a quick action in the AI Finder interface.",
        "2. Active filter values are converted into prompt instructions and ranking constraints.",
        "3. Gemini returns answer text and candidate movie titles when available.",
        "4. If Gemini fails, the app switches to the TF-IDF fallback recommender.",
        "5. Candidate movies are filtered, enriched with explanations and reranked with feedback-aware logic.",
        "6. Recommendation cards are rendered with reasoning text, metadata and feedback buttons.",
    ]:
        body.append(make_paragraph(line))

    body.append(make_paragraph("5. Primary Files and Code References", style="Heading1"))
    body.append(
        make_table(
            ["File", "Purpose"],
            [
                ["app.py", "Main Streamlit UI, AI Finder flow, filter logic, feedback persistence and fallback control."],
                ["recommender.py", "Recommendation reasoning helpers, feedback profile building and TF-IDF fallback ranking."],
                ["movies.db", "SQLite database that stores movie collection data, metadata and recommendation feedback."],
            ],
        )
    )
    body.append(make_paragraph("Additional Notes / Instructions", style="Heading2"))
    body.append(
        make_paragraph(
            "To run the full Streamlit prototype and perform live UI testing, follow the steps in README.md and install the required dependencies before launching app.py."
        )
    )

    if sect_pr is not None:
        body.append(copy.deepcopy(sect_pr))

    xml_bytes = ET.tostring(root, encoding="utf-8", xml_declaration=True)
    return xml_bytes


def main() -> None:
    with zipfile.ZipFile(TEMPLATE_PATH, "r") as source_zip:
        original_xml = source_zip.read("word/document.xml")
        original_root = ET.fromstring(original_xml)
        sect_pr = original_root.find(f".//{w_tag('sectPr')}")
        new_document_xml = build_document_xml(sect_pr)

        with zipfile.ZipFile(OUTPUT_PATH, "w") as target_zip:
            for item in source_zip.infolist():
                data = source_zip.read(item.filename)
                if item.filename == "word/document.xml":
                    data = new_document_xml
                target_zip.writestr(item, data)

    print(f"CREATED: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
