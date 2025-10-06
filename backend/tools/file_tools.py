"""
File Operation Tools
LangChain tools for file reading and writing operations
"""

import json
import csv
from pathlib import Path
from typing import Optional, List, Dict, Any
from langchain.tools import tool


# ============================================================================
# FILE READING TOOLS
# ============================================================================

@tool
def read_feedback_chunk(file_path: str, start_line: int = 0, num_lines: int = 10) -> str:
    """
    Read a chunk of feedback data from a file.
    
    Args:
        file_path: Path to the feedback file
        start_line: Starting line number (0-indexed)
        num_lines: Number of lines to read
    
    Returns:
        String containing the feedback chunk
    
    Example:
        >>> chunk = read_feedback_chunk("feedback_data.csv", start_line=0, num_lines=5)
    """
    try:
        file_path = Path(file_path)
        
        if not file_path.exists():
            return f"Error: File not found: {file_path}"
        
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        end_line = min(start_line + num_lines, len(lines))
        chunk = lines[start_line:end_line]
        
        return ''.join(chunk)
        
    except Exception as e:
        return f"Error reading file: {str(e)}"


@tool
def read_csv_file(file_path: str, max_rows: Optional[int] = None) -> str:
    """
    Read a CSV file and return as formatted string.
    
    Args:
        file_path: Path to the CSV file
        max_rows: Maximum number of rows to read (None for all)
    
    Returns:
        Formatted string representation of CSV data
    
    Example:
        >>> data = read_csv_file("feedback_data.csv", max_rows=10)
    """
    try:
        file_path = Path(file_path)
        
        if not file_path.exists():
            return f"Error: File not found: {file_path}"
        
        rows = []
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                if max_rows and i >= max_rows:
                    break
                rows.append(row)
        
        if not rows:
            return "No data found in CSV file"
        
        # Format as string
        result = f"Read {len(rows)} rows from {file_path}\n\n"
        result += "Columns: " + ", ".join(rows[0].keys()) + "\n\n"
        
        for i, row in enumerate(rows[:5], 1):  # Show first 5
            result += f"Row {i}: {row}\n"
        
        if len(rows) > 5:
            result += f"... and {len(rows) - 5} more rows"
        
        return result
        
    except Exception as e:
        return f"Error reading CSV file: {str(e)}"


@tool
def read_json_file(file_path: str) -> str:
    """
    Read a JSON file and return as formatted string.
    
    Args:
        file_path: Path to the JSON file
    
    Returns:
        Formatted string representation of JSON data
    
    Example:
        >>> data = read_json_file("results.json")
    """
    try:
        file_path = Path(file_path)
        
        if not file_path.exists():
            return f"Error: File not found: {file_path}"
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        return json.dumps(data, indent=2)
        
    except Exception as e:
        return f"Error reading JSON file: {str(e)}"


# ============================================================================
# FILE WRITING TOOLS
# ============================================================================

@tool
def write_feedback(file_path: str, content: str, mode: str = "w") -> str:
    """
    Write feedback or results to a file.
    
    Args:
        file_path: Path to the output file
        content: Content to write
        mode: Write mode ('w' for overwrite, 'a' for append)
    
    Returns:
        Success message or error
    
    Example:
        >>> result = write_feedback("output.txt", "Analysis results...")
    """
    try:
        file_path = Path(file_path)
        
        # Create parent directories if they don't exist
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(file_path, mode, encoding='utf-8') as f:
            f.write(content)
        
        return f"Successfully wrote to {file_path} ({len(content)} bytes)"
        
    except Exception as e:
        return f"Error writing to file: {str(e)}"


@tool
def write_json_file(file_path: str, data: Dict[str, Any]) -> str:
    """
    Write data to a JSON file.
    
    Args:
        file_path: Path to the output JSON file
        data: Dictionary data to write
    
    Returns:
        Success message or error
    
    Example:
        >>> result = write_json_file("results.json", {"status": "complete"})
    """
    try:
        file_path = Path(file_path)
        
        # Create parent directories if they don't exist
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, default=str)
        
        return f"Successfully wrote JSON to {file_path}"
        
    except Exception as e:
        return f"Error writing JSON file: {str(e)}"


# ============================================================================
# UTILITY TOOLS
# ============================================================================

@tool
def count_file_lines(file_path: str) -> str:
    """
    Count the number of lines in a file.
    
    Args:
        file_path: Path to the file
    
    Returns:
        String with line count or error
    
    Example:
        >>> count = count_file_lines("feedback_data.csv")
    """
    try:
        file_path = Path(file_path)
        
        if not file_path.exists():
            return f"Error: File not found: {file_path}"
        
        with open(file_path, 'r', encoding='utf-8') as f:
            line_count = sum(1 for _ in f)
        
        return f"File {file_path.name} has {line_count} lines"
        
    except Exception as e:
        return f"Error counting lines: {str(e)}"


@tool
def check_file_exists(file_path: str) -> str:
    """
    Check if a file exists.
    
    Args:
        file_path: Path to check
    
    Returns:
        String indicating if file exists
    
    Example:
        >>> exists = check_file_exists("feedback_data.csv")
    """
    file_path = Path(file_path)
    
    if file_path.exists():
        size = file_path.stat().st_size
        return f"File exists: {file_path} ({size} bytes)"
    else:
        return f"File does not exist: {file_path}"


@tool
def list_files_in_directory(directory: str, pattern: str = "*") -> str:
    """
    List files in a directory matching a pattern.
    
    Args:
        directory: Directory path
        pattern: File pattern (e.g., "*.csv", "*.json")
    
    Returns:
        String with list of files
    
    Example:
        >>> files = list_files_in_directory(".", pattern="*.csv")
    """
    try:
        directory = Path(directory)
        
        if not directory.exists():
            return f"Error: Directory not found: {directory}"
        
        if not directory.is_dir():
            return f"Error: Not a directory: {directory}"
        
        files = sorted(directory.glob(pattern))
        
        if not files:
            return f"No files matching '{pattern}' found in {directory}"
        
        result = f"Found {len(files)} file(s) in {directory}:\n"
        for file in files:
            size = file.stat().st_size
            result += f"  • {file.name} ({size} bytes)\n"
        
        return result
        
    except Exception as e:
        return f"Error listing files: {str(e)}"


# Create tool instances for easy import
read_feedback_chunk_tool = read_feedback_chunk
write_feedback_tool = write_feedback
read_csv_file_tool = read_csv_file
write_json_file_tool = write_json_file
read_json_file_tool = read_json_file


# ============================================================================
# DEMO / TESTING
# ============================================================================

def main():
    """Demo the file tools."""
    print("\n" + "=" * 80)
    print("FILE TOOLS DEMONSTRATION")
    print("=" * 80)
    
    # Test reading CSV
    print("\n1️⃣  Reading CSV file:")
    result = read_csv_file.invoke({"file_path": "feedback_data.csv", "max_rows": 3})
    print(result[:300] + "...")
    
    # Test counting lines
    print("\n2️⃣  Counting lines:")
    result = count_file_lines.invoke({"file_path": "feedback_data.csv"})
    print(result)
    
    # Test checking file existence
    print("\n3️⃣  Checking file existence:")
    result = check_file_exists.invoke({"file_path": "feedback_data.csv"})
    print(result)
    
    # Test listing files
    print("\n4️⃣  Listing CSV files:")
    result = list_files_in_directory.invoke({"directory": ".", "pattern": "*.csv"})
    print(result)
    
    print("\n" + "=" * 80)
    print("✅ FILE TOOLS DEMO COMPLETE")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()

