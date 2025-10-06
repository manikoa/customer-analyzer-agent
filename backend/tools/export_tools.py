"""
Export Operation Tools
LangChain tools for exporting data in various formats
"""

import json
import csv
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime
from langchain.tools import tool


# ============================================================================
# CSV EXPORT TOOLS
# ============================================================================

@tool
def export_to_csv(data: List[Dict[str, Any]], output_path: str, 
                  columns: List[str] = None) -> str:
    """
    Export data to CSV file.
    
    Args:
        data: List of dictionaries to export
        output_path: Output CSV file path
        columns: Optional list of column names (uses all keys if None)
    
    Returns:
        Success message or error
    
    Example:
        >>> export_to_csv([
        ...     {"id": 1, "text": "Great!", "sentiment": "positive"},
        ...     {"id": 2, "text": "Bad", "sentiment": "negative"}
        ... ], "results.csv")
    """
    try:
        if not data:
            return "Error: No data to export"
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Determine columns
        if columns is None:
            columns = list(data[0].keys())
        
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=columns, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(data)
        
        return f"Successfully exported {len(data)} rows to {output_path}"
        
    except Exception as e:
        return f"Error exporting to CSV: {str(e)}"


# ============================================================================
# JSON EXPORT TOOLS
# ============================================================================

@tool
def export_to_json(data: Any, output_path: str, pretty: bool = True) -> str:
    """
    Export data to JSON file.
    
    Args:
        data: Data to export (dict, list, etc.)
        output_path: Output JSON file path
        pretty: If True, use indentation for readability
    
    Returns:
        Success message or error
    
    Example:
        >>> export_to_json({"status": "complete", "results": [...]}, "results.json")
    """
    try:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            if pretty:
                json.dump(data, f, indent=2, default=str)
            else:
                json.dump(data, f, default=str)
        
        size = output_path.stat().st_size
        return f"Successfully exported to {output_path} ({size} bytes)"
        
    except Exception as e:
        return f"Error exporting to JSON: {str(e)}"


# ============================================================================
# MARKDOWN EXPORT TOOLS
# ============================================================================

@tool
def export_to_markdown(content: str, output_path: str, title: str = None) -> str:
    """
    Export content to Markdown file.
    
    Args:
        content: Markdown content to write
        output_path: Output .md file path
        title: Optional title to prepend
    
    Returns:
        Success message or error
    
    Example:
        >>> export_to_markdown("# Results\\n\\nAnalysis complete", "report.md")
    """
    try:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        markdown_content = ""
        
        if title:
            markdown_content += f"# {title}\n\n"
            markdown_content += f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n\n"
            markdown_content += "---\n\n"
        
        markdown_content += content
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(markdown_content)
        
        return f"Successfully exported to {output_path}"
        
    except Exception as e:
        return f"Error exporting to Markdown: {str(e)}"


@tool
def format_report_as_markdown(report_data: Dict[str, Any]) -> str:
    """
    Format a report dictionary as Markdown.
    
    Args:
        report_data: Report data dictionary
    
    Returns:
        Formatted Markdown string
    
    Example:
        >>> markdown = format_report_as_markdown({
        ...     "title": "Analysis Report",
        ...     "summary": "Complete",
        ...     "items": [...]
        ... })
    """
    try:
        lines = []
        
        # Title
        if 'title' in report_data:
            lines.append(f"# {report_data['title']}")
            lines.append("")
        
        # Metadata
        if 'generated_at' in report_data:
            lines.append(f"**Generated:** {report_data['generated_at']}")
        if 'total_items' in report_data:
            lines.append(f"**Total Items:** {report_data['total_items']}")
        lines.append("")
        lines.append("---")
        lines.append("")
        
        # Summary
        if 'summary' in report_data:
            lines.append("## Executive Summary")
            lines.append("")
            lines.append(report_data['summary'])
            lines.append("")
        
        # Action Items
        if 'action_items' in report_data:
            lines.append("## Action Items")
            lines.append("")
            
            for i, item in enumerate(report_data['action_items'], 1):
                lines.append(f"### {i}. {item.get('title', 'Untitled')}")
                lines.append("")
                lines.append(f"**Priority:** {item.get('priority', 'N/A')}")
                lines.append(f"**Category:** {item.get('category', 'N/A')}")
                lines.append("")
                
                if 'problem_statement' in item:
                    lines.append(f"**Problem:** {item['problem_statement']}")
                    lines.append("")
                
                if 'recommended_action' in item:
                    lines.append(f"**Action:** {item['recommended_action']}")
                    lines.append("")
                
                lines.append("---")
                lines.append("")
        
        return "\n".join(lines)
        
    except Exception as e:
        return f"Error formatting report: {str(e)}"


# ============================================================================
# PDF EXPORT TOOLS (placeholder)
# ============================================================================

@tool
def generate_pdf_report(markdown_content: str, output_path: str) -> str:
    """
    Generate PDF report from Markdown content.
    
    Note: This is a placeholder. Actual PDF generation requires
    additional libraries like weasyprint or reportlab.
    
    Args:
        markdown_content: Markdown content
        output_path: Output PDF file path
    
    Returns:
        Success message or informational message
    
    Example:
        >>> generate_pdf_report(markdown_text, "report.pdf")
    """
    return (
        f"PDF generation requires additional libraries.\n"
        f"To generate PDF:\n"
        f"1. Install: pip install markdown weasyprint\n"
        f"2. Use a PDF generation library\n"
        f"For now, content saved as Markdown to: {output_path}.md"
    )


# ============================================================================
# BATCH EXPORT TOOLS
# ============================================================================

@tool
def export_analysis_results(results: Dict[str, Any], output_dir: str, 
                           formats: List[str] = None) -> str:
    """
    Export analysis results in multiple formats.
    
    Args:
        results: Analysis results dictionary
        output_dir: Output directory
        formats: List of formats ('csv', 'json', 'markdown')
    
    Returns:
        Success message with list of generated files
    
    Example:
        >>> export_analysis_results(
        ...     results_dict,
        ...     "output/",
        ...     formats=["json", "markdown"]
        ... )
    """
    try:
        if formats is None:
            formats = ['json', 'markdown']
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        base_name = f"analysis_results_{timestamp}"
        
        generated_files = []
        
        # Export JSON
        if 'json' in formats:
            json_path = output_dir / f"{base_name}.json"
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, default=str)
            generated_files.append(str(json_path))
        
        # Export Markdown
        if 'markdown' in formats:
            md_path = output_dir / f"{base_name}.md"
            markdown_content = format_report_as_markdown(results)
            with open(md_path, 'w', encoding='utf-8') as f:
                f.write(markdown_content)
            generated_files.append(str(md_path))
        
        # Export CSV (if applicable)
        if 'csv' in formats and 'items' in results:
            csv_path = output_dir / f"{base_name}.csv"
            items = results['items']
            if items:
                with open(csv_path, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.DictWriter(f, fieldnames=items[0].keys())
                    writer.writeheader()
                    writer.writerows(items)
                generated_files.append(str(csv_path))
        
        result = f"Successfully exported analysis results to {len(generated_files)} file(s):\n"
        for file_path in generated_files:
            result += f"  • {file_path}\n"
        
        return result
        
    except Exception as e:
        return f"Error exporting analysis results: {str(e)}"


# Create tool instances for easy import
export_to_csv_tool = export_to_csv
export_to_json_tool = export_to_json
export_to_markdown_tool = export_to_markdown
generate_pdf_report_tool = generate_pdf_report


# ============================================================================
# DEMO / TESTING
# ============================================================================

def main():
    """Demo the export tools."""
    print("\n" + "=" * 80)
    print("EXPORT TOOLS DEMONSTRATION")
    print("=" * 80)
    
    # Test data
    test_data = [
        {"id": 1, "text": "Great app!", "sentiment": "positive", "score": 9},
        {"id": 2, "text": "Too slow", "sentiment": "negative", "score": 3},
        {"id": 3, "text": "It's okay", "sentiment": "neutral", "score": 6}
    ]
    
    # Test CSV export
    print("\n1️⃣  Exporting to CSV:")
    result = export_to_csv(test_data, "test_output/results.csv")
    print(result)
    
    # Test JSON export
    print("\n2️⃣  Exporting to JSON:")
    result = export_to_json(
        {"results": test_data, "total": len(test_data)},
        "test_output/results.json"
    )
    print(result)
    
    # Test Markdown export
    print("\n3️⃣  Exporting to Markdown:")
    markdown_content = "# Analysis Results\n\nAnalysis complete with 3 items."
    result = export_to_markdown(
        markdown_content,
        "test_output/report.md",
        title="Customer Feedback Analysis"
    )
    print(result)
    
    # Test batch export
    print("\n4️⃣  Batch export:")
    result = export_analysis_results(
        {
            "title": "Test Report",
            "generated_at": datetime.now().isoformat(),
            "items": test_data
        },
        "test_output/",
        formats=["json", "markdown"]
    )
    print(result)
    
    print("\n" + "=" * 80)
    print("✅ EXPORT TOOLS DEMO COMPLETE")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()

