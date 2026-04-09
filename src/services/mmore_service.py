"""
mmore document processing service for FastAPI

Handles text extraction from various file types (PDF, DOCX, TXT,
HTML, media, etc.) using mmore's auto-dispatching processor registry.
For image files without a dedicated processor, Ollama vision is used
to generate a textual description.
For xlsx files, each sheet and subtable is extracted individually.
"""

import base64
import io
import logging
import os
from typing import List, Dict, Any
from urllib.parse import urlparse

import pandas as pd
from unstructured.partition.xlsx import partition_xlsx

from mmore.process.processors.base import AutoProcessor, ProcessorConfig
from mmore.process.crawler import FileDescriptor, URLDescriptor
from mmore.type import MultimodalSample

from src.services.ollama_service import ollama_service, get_prompt_config

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.tiff'}
XLSX_EXTENSIONS = {'.xlsx', '.xls'}


# ── DataFrame cleaning helpers ──────────────────────────────────────────────

def _detect_horizontal_table(df: pd.DataFrame) -> bool:
    """Return True if *df* looks like a two-column key→value layout."""
    df_temp = df.dropna(how='all').dropna(axis=1, how='all')
    if df_temp.empty or df_temp.shape[1] != 2 or df_temp.shape[0] < 2:
        return False
    first_col = df_temp.iloc[:, 0].dropna()
    if len(first_col) == 0:
        return False
    string_count = sum(isinstance(val, str) for val in first_col)
    return string_count / len(first_col) > 0.6


def _transpose_horizontal_table(df: pd.DataFrame) -> pd.DataFrame:
    """Transpose a two-column key→value table into a single-row table."""
    df_temp = df.dropna(how='all').dropna(axis=1, how='all').reset_index(drop=True)
    headers = df_temp.iloc[:, 0].astype(str).str.strip().str.rstrip(':?').str.strip()
    values = df_temp.iloc[:, 1]

    clean_headers: list[str] = []
    header_counts: dict[str, int] = {}
    for i, h in enumerate(headers):
        h = h if h and h != 'nan' else f'Column_{i}'
        if h in header_counts:
            header_counts[h] += 1
            clean_headers.append(f"{h}_{header_counts[h]}")
        else:
            header_counts[h] = 0
            clean_headers.append(h)

    return pd.DataFrame([values.values], columns=clean_headers)


def _clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Remove empty rows/columns, fix headers, and handle horizontal tables."""
    df_clean = df.copy()
    df_clean = df_clean.dropna(how='all').dropna(axis=1, how='all').reset_index(drop=True)

    if df_clean.empty:
        return df_clean

    if _detect_horizontal_table(df_clean):
        return _transpose_horizontal_table(df_clean)

    # Promote first row to header when columns are unnamed
    current_cols = df_clean.columns.tolist()
    has_default_cols = all(
        isinstance(col, int) or str(col).startswith('Unnamed') or col == ''
        for col in current_cols
    )
    if has_default_cols and len(df_clean) > 0:
        new_headers = [
            str(h).strip() if pd.notna(h) else f'Column_{i}'
            for i, h in enumerate(df_clean.iloc[0].tolist())
        ]
        if any(h and not h.startswith('Column_') for h in new_headers):
            df_clean.columns = new_headers
            df_clean = df_clean.iloc[1:].reset_index(drop=True)

    df_clean = df_clean.dropna(how='all')
    df_clean.columns = [str(col).strip() for col in df_clean.columns]

    # Deduplicate column names
    cols = pd.Series(df_clean.columns)
    for dup in cols[cols.duplicated()].unique():
        cols[cols == dup] = [
            f"{dup}_{i}" if i != 0 else dup
            for i in range(sum(cols == dup))
        ]
    df_clean.columns = cols

    return df_clean


# ── XLSX subtable extraction ────────────────────────────────────────────────

def _parse_xlsx_subtables(
    filepath: str,
    min_table_rows: int = 2,
) -> List[Dict[str, Any]]:
    """Parse subtables from an XLSX file via *unstructured*.

    Returns a list of dicts, each with keys *dataframe*, *table_number*,
    *shape*, *columns*, and *was_transposed* for every valid table found.
    """
    elements = partition_xlsx(filename=filepath, infer_table_structure=True)

    subtables: list[dict[str, Any]] = []
    table_counter = 1

    for element in elements:
        if element.category != "Table":
            continue

        if not (hasattr(element, 'metadata') and hasattr(element.metadata, 'text_as_html')):
            continue

        try:
            df = pd.read_html(io.StringIO(element.metadata.text_as_html))[0]
            is_horizontal = _detect_horizontal_table(df)
            df = _clean_dataframe(df)

            if df.empty:
                continue
            if not is_horizontal and len(df) < min_table_rows:
                continue

            subtables.append({
                "dataframe": df,
                "table_number": table_counter,
                "shape": df.shape,
                "columns": df.columns.tolist(),
                "was_transposed": is_horizontal,
            })
            table_counter += 1
        except Exception as e:
            logger.warning("Failed to parse table element: %s", e)

    return subtables


# ── Service class ───────────────────────────────────────────────────────────

class MmoreService:
    """Service for document processing via mmore"""

    def __init__(self):
        pass

    def _make_config(self, output_path: str) -> ProcessorConfig:
        return ProcessorConfig(custom_config={"output_path": output_path})

    async def process_file(self, uri: str, tmp_dir: str | None = None) -> list[MultimodalSample]:
        """
        Process a file with the appropriate mmore processor.

        For image files that mmore cannot handle, the image is sent to
        Ollama for captioning and the description is returned as text.

        For xlsx/xls files, each sheet and subtable is extracted as a
        separate MultimodalSample with CSV data in metadata.

        Args:
            uri: Local file path or HTTP/HTTPS URL
            tmp_dir: Temporary directory for processor output (cleaned by caller)

        Returns:
            List of MultimodalSample objects
        """
        parsed = urlparse(uri)

        if parsed.scheme in ('http', 'https'):
            descriptor = URLDescriptor(uri)
        else:
            file_path = parsed.path if parsed.scheme == 'file' else uri
            descriptor = FileDescriptor.from_filename(file_path)
            if descriptor is None:
                raise FileNotFoundError(f"Could not access file: {uri}")

        ext = (descriptor.file_extension or '').lower()

        # if ext in XLSX_EXTENSIONS:
        #     return self._process_xlsx(descriptor.file_path)

        processor_class = AutoProcessor.from_file(descriptor)

        if processor_class is None:
            if ext in IMAGE_EXTENSIONS:
                description = await self._describe_image(descriptor.file_path)
                return [MultimodalSample(text=description, modalities=[])]
            else:
                raise ValueError(f"No processor found for file: {uri}")

        output_path = tmp_dir or os.path.dirname(descriptor.file_path)
        config = self._make_config(output_path)
        processor = processor_class(config=config)
        return [processor.process(descriptor.file_path)]

    def export_xlsx_sheet_csvs(self, file_path: str, tmp_dir: str) -> list[tuple[str, str]]:
        """Export each sheet of an xlsx file as a CSV in tmp_dir.

        Returns:
            List of (sheet_name, csv_path) tuples, one per non-empty sheet.
        """
        xl = pd.ExcelFile(file_path)
        results: list[tuple[str, str]] = []
        for sheet_name in xl.sheet_names:
            df = xl.parse(sheet_name, header=None)
            df = _clean_dataframe(df)
            if df.empty:
                logger.info("Skipping empty sheet '%s' in '%s'", sheet_name, file_path)
                continue
            safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in sheet_name)
            csv_path = os.path.join(tmp_dir, f"{safe_name}.csv")
            df.to_csv(csv_path, index=False)
            results.append((sheet_name, csv_path))
            logger.info("Exported sheet '%s' → '%s'", sheet_name, csv_path)
        return results

    def _process_xlsx(self, file_path: str) -> list[MultimodalSample]:
        """
        Extract each subtable from an xlsx file as a separate sample.

        Each MultimodalSample contains:
        - text: the table rendered as a string (for vectorization)
        - metadata.csv_data: the table as CSV (for structured Postgres storage)
        - metadata.table_number, metadata.columns, metadata.sheet info

        Args:
            file_path: Local path to the xlsx file

        Returns:
            List of MultimodalSample, one per subtable
        """
        subtables = _parse_xlsx_subtables(filepath=file_path)

        samples: list[MultimodalSample] = []

        for table in subtables:
            df = table["dataframe"]
            text = df.to_string(index=False)
            csv_data = df.to_csv(index=False)

            sample = MultimodalSample(
                text=text,
                modalities=[],
                metadata={
                    "table_number": table["table_number"],
                    "columns": table["columns"],
                    "shape": table["shape"],
                    "was_transposed": table["was_transposed"],
                    "csv_data": csv_data,
                },
            )
            samples.append(sample)

        if not samples:
            logger.warning("No subtables extracted from '%s'", file_path)
            samples.append(MultimodalSample(text="", modalities=[]))

        logger.info("Extracted %d subtable(s) from '%s'", len(samples), file_path)
        return samples

    async def _describe_image(self, file_path: str) -> str:
        """
        Send an image to Ollama and return a full textual description.

        Args:
            file_path: Local path to the image file

        Returns:
            Description string generated by the vision model
        """
        with open(file_path, 'rb') as f:
            image_b64 = base64.b64encode(f.read()).decode('utf-8')

        response = await ollama_service.generate(
            **get_prompt_config("describe_image"),
            images=[image_b64],
        )
        return response.response


# Global service instance
mmore_service = MmoreService()
