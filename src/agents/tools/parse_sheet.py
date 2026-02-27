from unstructured.partition.xlsx import partition_xlsx
from typing import List, Dict, Any, Optional
import pandas as pd
from pathlib import Path
import numpy as np
import io

def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean a DataFrame by removing empty rows/columns and ensuring proper headers.
    
    Args:
        df: Input DataFrame
        
    Returns:
        Cleaned DataFrame with proper column headers
    """
    # Make a copy to avoid modifying original
    df_clean = df.copy()
    
    # Remove completely empty rows
    df_clean = df_clean.dropna(how='all')
    
    # Remove completely empty columns
    df_clean = df_clean.dropna(axis=1, how='all')
    
    # Reset index after dropping rows
    df_clean = df_clean.reset_index(drop=True)
    
    if df_clean.empty:
        return df_clean
    
    # Check if table needs to be transposed (horizontal key-value format)
    needs_transpose = detect_horizontal_table(df_clean)
    
    if needs_transpose:
        df_clean = transpose_horizontal_table(df_clean)
        return df_clean
    
    # Check if first row should be headers
    current_cols = df_clean.columns.tolist()
    has_default_cols = all(isinstance(col, int) or 
                           str(col).startswith('Unnamed') or 
                           col == '' 
                           for col in current_cols)
    
    if has_default_cols and len(df_clean) > 0:
        # Use first row as column names
        new_headers = df_clean.iloc[0].tolist()
        
        # Clean up header names (handle NaN, convert to string)
        new_headers = [str(h).strip() if pd.notna(h) else f'Column_{i}' 
                      for i, h in enumerate(new_headers)]
        
        # Check if headers look valid (not all empty or numbers)
        if any(h and not h.startswith('Column_') for h in new_headers):
            df_clean.columns = new_headers
            df_clean = df_clean.iloc[1:].reset_index(drop=True)
    
    # Remove rows where all values are NaN after setting headers
    df_clean = df_clean.dropna(how='all')
    
    # Clean up column names - remove extra whitespace
    df_clean.columns = [str(col).strip() for col in df_clean.columns]
    
    # Handle duplicate column names by adding suffixes
    cols = pd.Series(df_clean.columns)
    for dup in cols[cols.duplicated()].unique():
        cols[cols == dup] = [f"{dup}_{i}" if i != 0 else dup 
                            for i in range(sum(cols == dup))]
    df_clean.columns = cols
    
    return df_clean


def detect_horizontal_table(df: pd.DataFrame) -> bool:
    """
    Detect if a table is in horizontal key-value format (needs transposing).
    
    Returns True if:
    - Table has 2 columns
    - First column contains labels/keys (mostly text)
    - Second column contains values
    - More rows than columns
    """
    # Remove empty rows/columns first for better detection
    df_temp = df.dropna(how='all').dropna(axis=1, how='all')
    
    if df_temp.empty:
        return False
    
    # Check if has exactly 2 columns with data
    if df_temp.shape[1] != 2:
        return False
    
    # Should have at least 2 rows
    if df_temp.shape[0] < 2:
        return False
    
    # Check if first column looks like keys (strings, often ending with :)
    first_col = df_temp.iloc[:, 0].dropna()
    if len(first_col) == 0:
        return False
    
    # Count how many entries in first column are strings
    string_count = sum(isinstance(val, str) for val in first_col)
    
    # If most of first column is strings, likely horizontal format
    if string_count / len(first_col) > 0.6:
        return True
    
    return False


def transpose_horizontal_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transpose a horizontal key-value table to proper column format.
    
    Converts:
        Key1    Value1
        Key2    Value2
    
    To:
        Key1    Key2
        Value1  Value2
    """
    # Remove empty rows first
    df_temp = df.dropna(how='all').dropna(axis=1, how='all').reset_index(drop=True)
    
    # Use first column as headers, second column as values
    headers = df_temp.iloc[:, 0].astype(str).str.strip()
    values = df_temp.iloc[:, 1]
    
    # Clean headers - remove trailing colons, extra whitespace, question marks
    headers = headers.str.rstrip(':?').str.strip()
    
    # Replace empty or NaN headers
    headers = [h if h and h != 'nan' else f'Column_{i}' for i, h in enumerate(headers)]
    
    # Handle duplicate headers
    header_counts = {}
    clean_headers = []
    for h in headers:
        if h in header_counts:
            header_counts[h] += 1
            clean_headers.append(f"{h}_{header_counts[h]}")
        else:
            header_counts[h] = 0
            clean_headers.append(h)
    
    # Create new DataFrame with one row
    transposed = pd.DataFrame([values.values], columns=clean_headers)
    
    return transposed


def parse_xlsx_subtables(
    filepath: str,
    output_dir: str = "output_tables",
    detect_subtables: bool = True,
    min_table_rows: int = 2,
    clean_tables: bool = True
) -> List[Dict[str, Any]]:
    """
    Parse subtables from an XLSX file using unstructured and save each as CSV.
    
    Args:
        filepath: Path to the XLSX file
        output_dir: Directory where CSV files will be saved
        detect_subtables: Whether to detect and separate subtables
        min_table_rows: Minimum number of rows to consider as a table
        clean_tables: Whether to remove empty rows/columns and fix headers
        
    Returns:
        List of dictionaries containing subtable data with metadata and CSV paths
    """
    # Create output directory if it doesn't exist
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Get base filename without extension
    base_filename = Path(filepath).stem
    
    # Partition the XLSX file
    elements = partition_xlsx(
        filename=filepath,
        infer_table_structure=True
    )
    
    subtables = []
    table_counter = 1
    
    for element in elements:
        # Check if element is a table
        if element.category == "Table":
            # Extract table metadata
            table_data = {
                "type": "table",
                "text": element.text,
                "table_number": table_counter,
                "metadata": {
                    "page_number": getattr(element.metadata, 'page_number', None),
                    "filename": getattr(element.metadata, 'filename', None),
                }
            }
            
            # Try to extract structured table data if available
            if hasattr(element, 'metadata') and hasattr(element.metadata, 'text_as_html'):
                # Convert HTML table to pandas DataFrame
                try:
                    text_as_html_io = io.StringIO(element.metadata.text_as_html)
                    df = pd.read_html(text_as_html_io)[0]
                    
                    # Store original shape for debugging
                    original_shape = df.shape
                    
                    # Check if it's a horizontal table before cleaning
                    is_horizontal = False
                    if clean_tables:
                        is_horizontal = detect_horizontal_table(df)
                    
                    # Clean the DataFrame
                    if clean_tables:
                        df = clean_dataframe(df)
                    
                    # For horizontal tables (transposed), they'll have 1 row, so skip min_table_rows check
                    # For normal tables, apply the min_table_rows requirement
                    # Also check if df is not empty after cleaning
                    if not df.empty:
                        meets_min_rows = is_horizontal or (len(df) >= min_table_rows)
                        
                        if meets_min_rows:
                            table_data["dataframe"] = df
                            table_data["shape"] = df.shape
                            table_data["columns"] = df.columns.tolist()
                            table_data["was_transposed"] = is_horizontal
                            
                            # Save to CSV
                            csv_filename = f"{base_filename}_table_{table_counter}.csv"
                            csv_path = output_path / csv_filename
                            df.to_csv(csv_path, index=False)
                            
                            table_data["csv_path"] = str(csv_path)
                            table_counter += 1
                    else:
                        table_data["warning"] = "Table was empty after cleaning"
                except Exception as e:
                    table_data["error"] = str(e)
            
            subtables.append(table_data)
    
    return subtables


def parse_xlsx_with_structure(filepath: str) -> Dict[str, Any]:
    """
    Parse XLSX file and return all elements with their structure.
    
    Args:
        filepath: Path to the XLSX file
        
    Returns:
        Dictionary containing all parsed elements organized by type
    """
    elements = partition_xlsx(
        filename=filepath,
        infer_table_structure=True
    )
    
    result = {
        "tables": [],
        "titles": [],
        "text": [],
        "other": []
    }
    
    for element in elements:
        element_data = {
            "content": element.text,
            "category": element.category,
            "metadata": vars(element.metadata) if hasattr(element, 'metadata') else {}
        }
        
        if element.category == "Table":
            result["tables"].append(element_data)
        elif element.category == "Title":
            result["titles"].append(element_data)
        elif element.category == "NarrativeText":
            result["text"].append(element_data)
        else:
            result["other"].append(element_data)
    
    return result


# Example usage
if __name__ == "__main__":
    # Parse subtables and save as CSVs
    xlsxfile = '/workspaces/canchat-v2-nss/nss/crewai/data/cache/100 Saint Joseph Road/surveys/DEC - 100 Rue Saint-Joseph Alma QC  -  RSD - Site Checklist and Intake Form.xlsx'

    subtables = parse_xlsx_subtables(xlsxfile, output_dir="output_tables")
    
    print(f"Found {len(subtables)} subtables\n")
    
    for i, table in enumerate(subtables):
        print(f"--- Subtable {i+1} ---")
        if "csv_path" in table:
            print(f"Saved to: {table['csv_path']}")
            print(f"Shape: {table['shape']} (rows x columns)")
            if table.get('was_transposed'):
                print(f"[Transposed from horizontal format]")
            print(f"Columns: {', '.join(table['columns'][:5])}" + 
                  (f"... (+{len(table['columns'])-5} more)" if len(table['columns']) > 5 else ""))
            print(f"\nPreview:")
            print(table['dataframe'].head())
            print()
        else:
            print(f"Text preview: {table['text'][:100]}...")
            if "error" in table:
                print(f"Error: {table['error']}")
            if "warning" in table:
                print(f"Warning: {table['warning']}")
            print()
    
    # Parse with full structure
    structured_data = parse_xlsx_with_structure(xlsxfile)
    print(f"\nTotal elements found:")
    print(f"  Tables: {len(structured_data['tables'])}")
    print(f"  Titles: {len(structured_data['titles'])}")
    print(f"  Text: {len(structured_data['text'])}")
    print(f"  Other: {len(structured_data['other'])}")