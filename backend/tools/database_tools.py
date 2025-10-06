"""
Database Operation Tools
LangChain tools for database logging and retrieval operations
"""

import json
import sqlite3
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime
from langchain.tools import tool


# ============================================================================
# DATABASE CONFIGURATION
# ============================================================================

DEFAULT_DB_PATH = "feedback_analysis.db"


def _get_db_connection(db_path: str = DEFAULT_DB_PATH) -> sqlite3.Connection:
    """Get a database connection."""
    conn = sqlite3.Connection(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def _init_database(db_path: str = DEFAULT_DB_PATH):
    """Initialize database tables if they don't exist."""
    conn = _get_db_connection(db_path)
    cursor = conn.cursor()
    
    # Analysis results table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS analysis_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            feedback_id TEXT,
            feedback_text TEXT,
            sentiment TEXT,
            sentiment_score REAL,
            category TEXT,
            urgency TEXT,
            confidence REAL,
            timestamp TEXT,
            metadata TEXT
        )
    ''')
    
    # Action items table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS action_items (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT,
            category TEXT,
            priority TEXT,
            status TEXT DEFAULT 'open',
            created_at TEXT,
            metadata TEXT
        )
    ''')
    
    # Workflow logs table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS workflow_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            workflow_id TEXT,
            node_name TEXT,
            status TEXT,
            message TEXT,
            timestamp TEXT,
            details TEXT
        )
    ''')
    
    conn.commit()
    conn.close()


# ============================================================================
# DATABASE TOOLS
# ============================================================================

@tool
def log_to_db(result: Dict[str, Any], db_path: str = DEFAULT_DB_PATH) -> str:
    """
    Log analysis results to the database.
    
    Args:
        result: Dictionary containing analysis results
        db_path: Path to SQLite database
    
    Returns:
        Success message or error
    
    Example:
        >>> log_to_db({
        ...     "feedback_id": "1",
        ...     "feedback_text": "Great app!",
        ...     "sentiment": "positive",
        ...     "sentiment_score": 9,
        ...     "category": "General Feedback",
        ...     "urgency": "low"
        ... })
    """
    try:
        _init_database(db_path)
        conn = _get_db_connection(db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO analysis_results 
            (feedback_id, feedback_text, sentiment, sentiment_score, 
             category, urgency, confidence, timestamp, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            result.get('feedback_id', ''),
            result.get('feedback_text', ''),
            result.get('sentiment', ''),
            result.get('sentiment_score', 0),
            result.get('category', ''),
            result.get('urgency', ''),
            result.get('confidence', 0),
            datetime.now().isoformat(),
            json.dumps(result.get('metadata', {}))
        ))
        
        conn.commit()
        row_id = cursor.lastrowid
        conn.close()
        
        return f"Successfully logged result to database (ID: {row_id})"
        
    except Exception as e:
        return f"Error logging to database: {str(e)}"


@tool
def retrieve_from_db(query_params: Dict[str, Any], db_path: str = DEFAULT_DB_PATH) -> str:
    """
    Retrieve analysis results from the database.
    
    Args:
        query_params: Dictionary with query parameters (e.g., {"category": "Bug Report"})
        db_path: Path to SQLite database
    
    Returns:
        String with query results or error
    
    Example:
        >>> results = retrieve_from_db({"urgency": "critical"})
    """
    try:
        _init_database(db_path)
        conn = _get_db_connection(db_path)
        cursor = conn.cursor()
        
        # Build WHERE clause
        where_clauses = []
        params = []
        
        for key, value in query_params.items():
            if key in ['sentiment', 'category', 'urgency']:
                where_clauses.append(f"{key} = ?")
                params.append(value)
        
        where_sql = " AND ".join(where_clauses) if where_clauses else "1=1"
        
        query = f"SELECT * FROM analysis_results WHERE {where_sql} ORDER BY timestamp DESC LIMIT 10"
        cursor.execute(query, params)
        
        rows = cursor.fetchall()
        conn.close()
        
        if not rows:
            return "No results found"
        
        result = f"Found {len(rows)} result(s):\n\n"
        for row in rows:
            result += f"ID: {row['id']}\n"
            result += f"Feedback: {row['feedback_text'][:60]}...\n"
            result += f"Sentiment: {row['sentiment']} ({row['sentiment_score']})\n"
            result += f"Category: {row['category']} (Urgency: {row['urgency']})\n"
            result += f"Timestamp: {row['timestamp']}\n"
            result += "-" * 40 + "\n"
        
        return result
        
    except Exception as e:
        return f"Error retrieving from database: {str(e)}"


@tool
def update_db(record_id: int, updates: Dict[str, Any], db_path: str = DEFAULT_DB_PATH) -> str:
    """
    Update a database record.
    
    Args:
        record_id: ID of the record to update
        updates: Dictionary with fields to update
        db_path: Path to SQLite database
    
    Returns:
        Success message or error
    
    Example:
        >>> update_db(1, {"sentiment": "neutral", "sentiment_score": 6})
    """
    try:
        _init_database(db_path)
        conn = _get_db_connection(db_path)
        cursor = conn.cursor()
        
        # Build UPDATE clause
        set_clauses = []
        params = []
        
        for key, value in updates.items():
            if key in ['sentiment', 'sentiment_score', 'category', 'urgency', 'confidence']:
                set_clauses.append(f"{key} = ?")
                params.append(value)
        
        if not set_clauses:
            return "No valid fields to update"
        
        set_sql = ", ".join(set_clauses)
        params.append(record_id)
        
        query = f"UPDATE analysis_results SET {set_sql} WHERE id = ?"
        cursor.execute(query, params)
        
        conn.commit()
        rows_updated = cursor.rowcount
        conn.close()
        
        if rows_updated == 0:
            return f"No record found with ID: {record_id}"
        
        return f"Successfully updated record {record_id}"
        
    except Exception as e:
        return f"Error updating database: {str(e)}"


@tool
def query_db(sql_query: str, db_path: str = DEFAULT_DB_PATH) -> str:
    """
    Execute a custom SQL query (SELECT only for safety).
    
    Args:
        sql_query: SQL SELECT query
        db_path: Path to SQLite database
    
    Returns:
        String with query results or error
    
    Example:
        >>> results = query_db("SELECT category, COUNT(*) as count FROM analysis_results GROUP BY category")
    """
    try:
        # Safety check - only allow SELECT queries
        if not sql_query.strip().upper().startswith('SELECT'):
            return "Error: Only SELECT queries are allowed"
        
        _init_database(db_path)
        conn = _get_db_connection(db_path)
        cursor = conn.cursor()
        
        cursor.execute(sql_query)
        rows = cursor.fetchall()
        conn.close()
        
        if not rows:
            return "Query returned no results"
        
        # Format results
        result = f"Query returned {len(rows)} row(s):\n\n"
        
        # Get column names
        columns = rows[0].keys()
        result += " | ".join(columns) + "\n"
        result += "-" * 60 + "\n"
        
        for row in rows[:20]:  # Limit to 20 rows
            result += " | ".join(str(row[col]) for col in columns) + "\n"
        
        if len(rows) > 20:
            result += f"\n... and {len(rows) - 20} more rows"
        
        return result
        
    except Exception as e:
        return f"Error executing query: {str(e)}"


@tool
def log_workflow_event(workflow_id: str, node_name: str, status: str, 
                       message: str, details: Optional[Dict] = None,
                       db_path: str = DEFAULT_DB_PATH) -> str:
    """
    Log a workflow event to the database.
    
    Args:
        workflow_id: Unique workflow identifier
        node_name: Name of the node
        status: Status (started, completed, failed)
        message: Log message
        details: Optional additional details
        db_path: Path to SQLite database
    
    Returns:
        Success message or error
    
    Example:
        >>> log_workflow_event("wf_123", "sentiment_node", "completed", "Analyzed 10 items")
    """
    try:
        _init_database(db_path)
        conn = _get_db_connection(db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO workflow_logs 
            (workflow_id, node_name, status, message, timestamp, details)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            workflow_id,
            node_name,
            status,
            message,
            datetime.now().isoformat(),
            json.dumps(details or {})
        ))
        
        conn.commit()
        conn.close()
        
        return f"Logged workflow event: {workflow_id} - {node_name} - {status}"
        
    except Exception as e:
        return f"Error logging workflow event: {str(e)}"


@tool
def get_database_stats(db_path: str = DEFAULT_DB_PATH) -> str:
    """
    Get statistics about the database.
    
    Args:
        db_path: Path to SQLite database
    
    Returns:
        String with database statistics
    
    Example:
        >>> stats = get_database_stats()
    """
    try:
        _init_database(db_path)
        conn = _get_db_connection(db_path)
        cursor = conn.cursor()
        
        # Count records in each table
        cursor.execute("SELECT COUNT(*) as count FROM analysis_results")
        analysis_count = cursor.fetchone()['count']
        
        cursor.execute("SELECT COUNT(*) as count FROM action_items")
        action_count = cursor.fetchone()['count']
        
        cursor.execute("SELECT COUNT(*) as count FROM workflow_logs")
        log_count = cursor.fetchone()['count']
        
        # Get sentiment distribution
        cursor.execute("""
            SELECT sentiment, COUNT(*) as count 
            FROM analysis_results 
            GROUP BY sentiment
        """)
        sentiment_dist = cursor.fetchall()
        
        conn.close()
        
        result = f"Database Statistics ({db_path}):\n\n"
        result += f"Analysis Results: {analysis_count}\n"
        result += f"Action Items: {action_count}\n"
        result += f"Workflow Logs: {log_count}\n\n"
        
        if sentiment_dist:
            result += "Sentiment Distribution:\n"
            for row in sentiment_dist:
                result += f"  {row['sentiment']}: {row['count']}\n"
        
        return result
        
    except Exception as e:
        return f"Error getting database stats: {str(e)}"


# Create tool instances for easy import
log_to_db_tool = log_to_db
retrieve_from_db_tool = retrieve_from_db
update_db_tool = update_db
query_db_tool = query_db


# ============================================================================
# DEMO / TESTING
# ============================================================================

def main():
    """Demo the database tools."""
    print("\n" + "=" * 80)
    print("DATABASE TOOLS DEMONSTRATION")
    print("=" * 80)
    
    # Test logging
    print("\n1️⃣  Logging to database:")
    result = log_to_db({
        "feedback_id": "test_1",
        "feedback_text": "This is a test feedback",
        "sentiment": "positive",
        "sentiment_score": 8,
        "category": "General Feedback",
        "urgency": "low",
        "confidence": 0.9
    })
    print(result)
    
    # Test retrieval
    print("\n2️⃣  Retrieving from database:")
    result = retrieve_from_db({"urgency": "low"})
    print(result[:300] + "...")
    
    # Test stats
    print("\n3️⃣  Getting database statistics:")
    result = get_database_stats()
    print(result)
    
    # Test workflow logging
    print("\n4️⃣  Logging workflow event:")
    result = log_workflow_event(
        "test_workflow_001",
        "sentiment_node",
        "completed",
        "Successfully analyzed 1 item"
    )
    print(result)
    
    print("\n" + "=" * 80)
    print("✅ DATABASE TOOLS DEMO COMPLETE")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()

