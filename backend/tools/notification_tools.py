"""
Notification and Alerting Tools
LangChain tools for sending notifications and creating alerts
"""

import json
from typing import Dict, Any, Optional, List
from datetime import datetime
from pathlib import Path
from langchain.tools import tool


# ============================================================================
# EMAIL NOTIFICATION TOOLS
# ============================================================================

@tool
def send_email_notification(recipient: str, subject: str, body: str, 
                           priority: str = "normal") -> str:
    """
    Send an email notification.
    
    Note: This is a placeholder. Actual email sending requires
    SMTP configuration and email service setup.
    
    Args:
        recipient: Email recipient address
        subject: Email subject
        body: Email body content
        priority: Priority level (low, normal, high, critical)
    
    Returns:
        Success message or informational message
    
    Example:
        >>> send_email_notification(
        ...     "pm@example.com",
        ...     "Critical Issue Alert",
        ...     "10 critical bugs detected"
        ... )
    """
    # Log the notification instead of actually sending
    log_entry = {
        "type": "email",
        "timestamp": datetime.now().isoformat(),
        "recipient": recipient,
        "subject": subject,
        "body": body[:100] + "..." if len(body) > 100 else body,
        "priority": priority,
        "status": "logged"
    }
    
    # Save to notifications log
    log_file = Path("notifications_log.json")
    
    logs = []
    if log_file.exists():
        with open(log_file, 'r') as f:
            logs = json.load(f)
    
    logs.append(log_entry)
    
    with open(log_file, 'w') as f:
        json.dump(logs, f, indent=2)
    
    return (
        f"Email notification logged:\n"
        f"  To: {recipient}\n"
        f"  Subject: {subject}\n"
        f"  Priority: {priority}\n"
        f"  Note: Configure SMTP to actually send emails"
    )


# ============================================================================
# SLACK NOTIFICATION TOOLS
# ============================================================================

@tool
def send_slack_notification(channel: str, message: str, 
                           mention_users: Optional[List[str]] = None) -> str:
    """
    Send a Slack notification.
    
    Note: This is a placeholder. Actual Slack integration requires
    Slack API token and webhook setup.
    
    Args:
        channel: Slack channel name (e.g., "#alerts")
        message: Message content
        mention_users: Optional list of users to mention
    
    Returns:
        Success message or informational message
    
    Example:
        >>> send_slack_notification(
        ...     "#product-alerts",
        ...     "Critical issues detected in latest feedback",
        ...     mention_users=["@pm", "@eng-lead"]
        ... )
    """
    # Log the notification instead of actually sending
    log_entry = {
        "type": "slack",
        "timestamp": datetime.now().isoformat(),
        "channel": channel,
        "message": message,
        "mentions": mention_users or [],
        "status": "logged"
    }
    
    # Save to notifications log
    log_file = Path("notifications_log.json")
    
    logs = []
    if log_file.exists():
        with open(log_file, 'r') as f:
            logs = json.load(f)
    
    logs.append(log_entry)
    
    with open(log_file, 'w') as f:
        json.dump(logs, f, indent=2)
    
    mentions_str = ", ".join(mention_users) if mention_users else "None"
    
    return (
        f"Slack notification logged:\n"
        f"  Channel: {channel}\n"
        f"  Message: {message[:60]}...\n"
        f"  Mentions: {mentions_str}\n"
        f"  Note: Configure Slack API to actually send notifications"
    )


# ============================================================================
# ALERT CREATION TOOLS
# ============================================================================

@tool
def create_alert(title: str, description: str, severity: str, 
                category: str, metadata: Optional[Dict] = None) -> str:
    """
    Create an alert for critical issues.
    
    Args:
        title: Alert title
        description: Detailed description
        severity: Severity level (low, medium, high, critical)
        category: Alert category (bug, performance, security, etc.)
        metadata: Optional additional metadata
    
    Returns:
        Success message with alert ID
    
    Example:
        >>> create_alert(
        ...     "Data Loss Bug Detected",
        ...     "Export feature causing data corruption",
        ...     "critical",
        ...     "bug"
        ... )
    """
    try:
        alerts_file = Path("alerts.json")
        
        # Load existing alerts
        alerts = []
        if alerts_file.exists():
            with open(alerts_file, 'r') as f:
                alerts = json.load(f)
        
        # Create new alert
        alert_id = f"ALERT_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{len(alerts) + 1}"
        
        alert = {
            "id": alert_id,
            "title": title,
            "description": description,
            "severity": severity,
            "category": category,
            "status": "open",
            "created_at": datetime.now().isoformat(),
            "metadata": metadata or {}
        }
        
        alerts.append(alert)
        
        # Save alerts
        with open(alerts_file, 'w') as f:
            json.dump(alerts, f, indent=2)
        
        return (
            f"Alert created successfully:\n"
            f"  ID: {alert_id}\n"
            f"  Title: {title}\n"
            f"  Severity: {severity}\n"
            f"  Status: open"
        )
        
    except Exception as e:
        return f"Error creating alert: {str(e)}"


@tool
def log_critical_issue(issue_data: Dict[str, Any]) -> str:
    """
    Log a critical issue that requires immediate attention.
    
    Args:
        issue_data: Dictionary containing issue details
    
    Returns:
        Success message with issue ID
    
    Example:
        >>> log_critical_issue({
        ...     "feedback_id": "123",
        ...     "feedback_text": "System down!",
        ...     "category": "Bug Report",
        ...     "urgency": "critical",
        ...     "sentiment_score": 1
        ... })
    """
    try:
        issues_file = Path("critical_issues.json")
        
        # Load existing issues
        issues = []
        if issues_file.exists():
            with open(issues_file, 'r') as f:
                issues = json.load(f)
        
        # Create issue entry
        issue_id = f"ISSUE_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{len(issues) + 1}"
        
        issue = {
            "id": issue_id,
            "timestamp": datetime.now().isoformat(),
            "status": "needs_attention",
            **issue_data
        }
        
        issues.append(issue)
        
        # Save issues
        with open(issues_file, 'w') as f:
            json.dump(issues, f, indent=2)
        
        # Also create an alert
        create_alert(
            title=f"Critical Issue: {issue_data.get('category', 'Unknown')}",
            description=issue_data.get('feedback_text', 'No description')[:200],
            severity="critical",
            category=issue_data.get('category', 'unknown'),
            metadata={"issue_id": issue_id}
        )
        
        return (
            f"Critical issue logged:\n"
            f"  ID: {issue_id}\n"
            f"  Category: {issue_data.get('category', 'N/A')}\n"
            f"  Urgency: {issue_data.get('urgency', 'N/A')}\n"
            f"  Alert created for immediate attention"
        )
        
    except Exception as e:
        return f"Error logging critical issue: {str(e)}"


# ============================================================================
# ESCALATION TOOLS
# ============================================================================

@tool
def escalate_to_team(team: str, issue_summary: str, priority: str, 
                    contact_info: Optional[str] = None) -> str:
    """
    Escalate an issue to a specific team.
    
    Args:
        team: Team name (e.g., "engineering", "product", "support")
        issue_summary: Summary of the issue
        priority: Priority level
        contact_info: Optional contact information
    
    Returns:
        Success message
    
    Example:
        >>> escalate_to_team(
        ...     "engineering",
        ...     "Critical bugs affecting 60% of users",
        ...     "critical"
        ... )
    """
    escalation = {
        "type": "escalation",
        "timestamp": datetime.now().isoformat(),
        "team": team,
        "issue_summary": issue_summary,
        "priority": priority,
        "contact_info": contact_info,
        "status": "escalated"
    }
    
    # Save escalation
    escalations_file = Path("escalations.json")
    
    escalations = []
    if escalations_file.exists():
        with open(escalations_file, 'r') as f:
            escalations = json.load(f)
    
    escalations.append(escalation)
    
    with open(escalations_file, 'w') as f:
        json.dump(escalations, f, indent=2)
    
    return (
        f"Issue escalated to {team} team:\n"
        f"  Priority: {priority}\n"
        f"  Summary: {issue_summary[:80]}...\n"
        f"  Status: Escalated for immediate review"
    )


@tool
def get_active_alerts(severity: Optional[str] = None) -> str:
    """
    Get list of active alerts.
    
    Args:
        severity: Optional filter by severity
    
    Returns:
        String with active alerts
    
    Example:
        >>> alerts = get_active_alerts(severity="critical")
    """
    try:
        alerts_file = Path("alerts.json")
        
        if not alerts_file.exists():
            return "No alerts file found"
        
        with open(alerts_file, 'r') as f:
            alerts = json.load(f)
        
        # Filter by severity if specified
        if severity:
            alerts = [a for a in alerts if a.get('severity') == severity]
        
        # Filter to only open alerts
        active_alerts = [a for a in alerts if a.get('status') == 'open']
        
        if not active_alerts:
            return f"No active alerts{' with severity: ' + severity if severity else ''}"
        
        result = f"Active Alerts ({len(active_alerts)}):\n\n"
        
        for alert in active_alerts[:10]:  # Show first 10
            result += f"ID: {alert['id']}\n"
            result += f"Title: {alert['title']}\n"
            result += f"Severity: {alert['severity']}\n"
            result += f"Category: {alert['category']}\n"
            result += f"Created: {alert['created_at']}\n"
            result += "-" * 40 + "\n"
        
        return result
        
    except Exception as e:
        return f"Error retrieving alerts: {str(e)}"


# Create tool instances for easy import
send_email_notification_tool = send_email_notification
send_slack_notification_tool = send_slack_notification
create_alert_tool = create_alert
log_critical_issue_tool = log_critical_issue


# ============================================================================
# DEMO / TESTING
# ============================================================================

def main():
    """Demo the notification tools."""
    print("\n" + "=" * 80)
    print("NOTIFICATION TOOLS DEMONSTRATION")
    print("=" * 80)
    
    # Test email notification
    print("\n1️⃣  Sending email notification:")
    result = send_email_notification(
        "pm@example.com",
        "Critical Issues Alert",
        "10 critical bugs detected in latest feedback analysis",
        priority="high"
    )
    print(result)
    
    # Test Slack notification
    print("\n2️⃣  Sending Slack notification:")
    result = send_slack_notification(
        "#product-alerts",
        "Critical feedback analysis complete: 5 urgent issues require attention",
        mention_users=["@pm", "@eng-lead"]
    )
    print(result)
    
    # Test alert creation
    print("\n3️⃣  Creating alert:")
    result = create_alert(
        "Data Export Bug",
        "Users reporting data corruption during CSV export",
        "critical",
        "bug",
        metadata={"affected_users": 60, "first_reported": "2024-01-15"}
    )
    print(result)
    
    # Test critical issue logging
    print("\n4️⃣  Logging critical issue:")
    result = log_critical_issue({
        "feedback_id": "FB001",
        "feedback_text": "System crashed and lost all my data!",
        "category": "Bug Report",
        "urgency": "critical",
        "sentiment_score": 1,
        "affected_feature": "export"
    })
    print(result)
    
    # Test escalation
    print("\n5️⃣  Escalating to team:")
    result = escalate_to_team(
        "engineering",
        "Critical export bug affecting 60% of users",
        "critical",
        contact_info="bug-reports@example.com"
    )
    print(result)
    
    # Test getting active alerts
    print("\n6️⃣  Getting active alerts:")
    result = get_active_alerts(severity="critical")
    print(result[:300] + "...")
    
    print("\n" + "=" * 80)
    print("✅ NOTIFICATION TOOLS DEMO COMPLETE")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()

