"""
Connection failure detection and error reporting utilities.

This module provides tools for detecting WebSocket connection failures,
analyzing connection patterns, and generating detailed error reports
for troubleshooting bot connectivity issues.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass, field
import json

from .websocket_monitor import (
    WebSocketConnectionMonitor, WebSocketConnectionInfo, WebSocketState,
    ConnectionFailureReason, get_websocket_monitor
)
from .diagnostics import (
    BotDiagnosticTracker, BotLifecycleEvent, DiagnosticLevel,
    get_diagnostic_tracker
)


class ConnectionIssueType(Enum):
    """Types of connection issues that can be detected."""
    FREQUENT_DISCONNECTIONS = "frequent_disconnections"
    SLOW_CONNECTION = "slow_connection"
    HIGH_LATENCY = "high_latency"
    MESSAGE_LOSS = "message_loss"
    AUTHENTICATION_FAILURES = "authentication_failures"
    SERVER_UNAVAILABLE = "server_unavailable"
    TIMEOUT_ISSUES = "timeout_issues"
    PROTOCOL_ERRORS = "protocol_errors"
    RECONNECTION_FAILURES = "reconnection_failures"
    NO_ACTIVITY = "no_activity"


class IssueSeverity(Enum):
    """Severity levels for connection issues."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class ConnectionIssue:
    """Represents a detected connection issue."""
    issue_id: str
    bot_id: str
    issue_type: ConnectionIssueType
    severity: IssueSeverity
    detected_at: datetime
    description: str
    details: Dict[str, Any] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)
    resolved: bool = False
    resolved_at: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'issue_id': self.issue_id,
            'bot_id': self.bot_id,
            'issue_type': self.issue_type.value,
            'severity': self.severity.value,
            'detected_at': self.detected_at.isoformat(),
            'description': self.description,
            'details': self.details,
            'recommendations': self.recommendations,
            'resolved': self.resolved,
            'resolved_at': self.resolved_at.isoformat() if self.resolved_at else None
        }


@dataclass
class ConnectionPattern:
    """Represents connection behavior patterns for analysis."""
    bot_id: str
    analysis_period: timedelta
    total_connections: int
    successful_connections: int
    failed_connections: int
    average_connection_time: float
    average_connection_duration: float
    disconnection_frequency: float
    common_failure_reasons: List[Tuple[ConnectionFailureReason, int]]
    message_patterns: Dict[str, int]
    latency_stats: Dict[str, float]
    
    @property
    def success_rate(self) -> float:
        """Calculate connection success rate."""
        if self.total_connections == 0:
            return 0.0
        return self.successful_connections / self.total_connections
    
    @property
    def failure_rate(self) -> float:
        """Calculate connection failure rate."""
        return 1.0 - self.success_rate
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'bot_id': self.bot_id,
            'analysis_period': str(self.analysis_period),
            'total_connections': self.total_connections,
            'successful_connections': self.successful_connections,
            'failed_connections': self.failed_connections,
            'success_rate': self.success_rate,
            'failure_rate': self.failure_rate,
            'average_connection_time': self.average_connection_time,
            'average_connection_duration': self.average_connection_duration,
            'disconnection_frequency': self.disconnection_frequency,
            'common_failure_reasons': [(reason.value, count) for reason, count in self.common_failure_reasons],
            'message_patterns': self.message_patterns,
            'latency_stats': self.latency_stats
        }


class ConnectionDiagnostics:
    """
    Connection failure detection and error reporting system.
    
    Analyzes WebSocket connection patterns, detects issues, and provides
    detailed error reports with recommendations for troubleshooting.
    """
    
    def __init__(self, websocket_monitor: Optional[WebSocketConnectionMonitor] = None,
                 diagnostic_tracker: Optional[BotDiagnosticTracker] = None,
                 analysis_window: timedelta = timedelta(hours=1),
                 issue_detection_interval: float = 60.0):
        """
        Initialize connection diagnostics.
        
        Args:
            websocket_monitor: WebSocket monitor instance
            diagnostic_tracker: Diagnostic tracker instance
            analysis_window: Time window for pattern analysis
            issue_detection_interval: Interval between issue detection runs (seconds)
        """
        self.websocket_monitor = websocket_monitor or get_websocket_monitor()
        self.diagnostic_tracker = diagnostic_tracker or get_diagnostic_tracker()
        self.analysis_window = analysis_window
        self.issue_detection_interval = issue_detection_interval
        
        # Issue tracking
        self._detected_issues: Dict[str, List[ConnectionIssue]] = {}  # bot_id -> issues
        self._issue_counter = 0
        
        # Pattern analysis
        self._connection_patterns: Dict[str, ConnectionPattern] = {}
        
        # Background tasks
        self._detection_task: Optional[asyncio.Task] = None
        self._running = False
        
        self.logger = logging.getLogger(__name__)
    
    async def start(self) -> None:
        """Start connection diagnostics."""
        if self._running:
            return
        
        self._running = True
        
        # Start issue detection task
        self._detection_task = asyncio.create_task(self._periodic_issue_detection())
        
        self.logger.info("Connection diagnostics started")
    
    async def stop(self) -> None:
        """Stop connection diagnostics."""
        if not self._running:
            return
        
        self._running = False
        
        # Cancel detection task
        if self._detection_task:
            self._detection_task.cancel()
            try:
                await self._detection_task
            except asyncio.CancelledError:
                pass
        
        self.logger.info("Connection diagnostics stopped")
    
    async def analyze_bot_connection(self, bot_id: str) -> Optional[ConnectionPattern]:
        """
        Analyze connection patterns for a specific bot.
        
        Args:
            bot_id: Bot identifier
            
        Returns:
            ConnectionPattern or None if insufficient data
        """
        # Get connection info
        connection_info = self.websocket_monitor.get_connection_info(bot_id)
        if not connection_info:
            return None
        
        # Get diagnostic events for analysis
        events = self.diagnostic_tracker.get_diagnostic_events(
            bot_id=bot_id,
            limit=1000  # Analyze last 1000 events
        )
        
        # Filter events within analysis window
        cutoff_time = datetime.now() - self.analysis_window
        recent_events = [e for e in events if e.timestamp >= cutoff_time]
        
        if not recent_events:
            return None
        
        # Analyze connection events
        connection_events = [e for e in recent_events if 'websocket' in e.event_type.value.lower()]
        connect_events = [e for e in connection_events if 'connect' in e.event_type.value]
        disconnect_events = [e for e in connection_events if 'disconnect' in e.event_type.value or 'failed' in e.event_type.value]
        
        # Calculate connection statistics
        total_connections = len(connect_events)
        successful_connections = len([e for e in connect_events if e.level != DiagnosticLevel.ERROR])
        failed_connections = total_connections - successful_connections
        
        # Calculate timing statistics
        connection_times = []
        connection_durations = []
        
        for event in connect_events:
            if 'connection_time_seconds' in event.details:
                connection_times.append(event.details['connection_time_seconds'])
        
        # Analyze failure reasons
        failure_reasons = {}
        for event in disconnect_events:
            if 'failure_reason' in event.details:
                reason_str = event.details['failure_reason']
                try:
                    reason = ConnectionFailureReason(reason_str)
                    failure_reasons[reason] = failure_reasons.get(reason, 0) + 1
                except ValueError:
                    pass
        
        # Analyze message patterns
        message_history = self.websocket_monitor.get_message_history(bot_id, limit=500)
        message_patterns = {}
        for msg in message_history:
            message_patterns[msg.message_type] = message_patterns.get(msg.message_type, 0) + 1
        
        # Calculate latency statistics
        latency_stats = {}
        if connection_info.latency_ms is not None:
            latency_stats = {
                'current_latency_ms': connection_info.latency_ms,
                'avg_latency_ms': connection_info.latency_ms  # Would need historical data for true average
            }
        
        # Create connection pattern
        pattern = ConnectionPattern(
            bot_id=bot_id,
            analysis_period=self.analysis_window,
            total_connections=total_connections,
            successful_connections=successful_connections,
            failed_connections=failed_connections,
            average_connection_time=sum(connection_times) / len(connection_times) if connection_times else 0.0,
            average_connection_duration=sum(connection_durations) / len(connection_durations) if connection_durations else 0.0,
            disconnection_frequency=len(disconnect_events) / self.analysis_window.total_seconds() * 3600,  # per hour
            common_failure_reasons=sorted(failure_reasons.items(), key=lambda x: x[1], reverse=True),
            message_patterns=message_patterns,
            latency_stats=latency_stats
        )
        
        self._connection_patterns[bot_id] = pattern
        return pattern
    
    async def detect_connection_issues(self, bot_id: str) -> List[ConnectionIssue]:
        """
        Detect connection issues for a specific bot.
        
        Args:
            bot_id: Bot identifier
            
        Returns:
            List of detected connection issues
        """
        issues = []
        
        # Analyze connection pattern
        pattern = await self.analyze_bot_connection(bot_id)
        if not pattern:
            return issues
        
        # Get current connection info
        connection_info = self.websocket_monitor.get_connection_info(bot_id)
        connection_health = self.websocket_monitor.get_connection_health(bot_id)
        
        # Detect frequent disconnections
        if pattern.disconnection_frequency > 5:  # More than 5 disconnections per hour
            issue = self._create_issue(
                bot_id=bot_id,
                issue_type=ConnectionIssueType.FREQUENT_DISCONNECTIONS,
                severity=IssueSeverity.HIGH,
                description=f"Bot experiencing frequent disconnections ({pattern.disconnection_frequency:.1f} per hour)",
                details={
                    'disconnection_frequency': pattern.disconnection_frequency,
                    'failure_rate': pattern.failure_rate,
                    'common_failures': pattern.common_failure_reasons[:3]
                },
                recommendations=[
                    "Check network stability",
                    "Verify WebSocket server availability",
                    "Review bot reconnection logic",
                    "Consider increasing reconnection delays"
                ]
            )
            issues.append(issue)
        
        # Detect slow connections
        if pattern.average_connection_time > 10.0:  # More than 10 seconds to connect
            issue = self._create_issue(
                bot_id=bot_id,
                issue_type=ConnectionIssueType.SLOW_CONNECTION,
                severity=IssueSeverity.MEDIUM,
                description=f"Bot connections are slow (avg {pattern.average_connection_time:.1f}s)",
                details={
                    'average_connection_time': pattern.average_connection_time,
                    'total_connections': pattern.total_connections
                },
                recommendations=[
                    "Check network latency to game server",
                    "Verify DNS resolution speed",
                    "Review server load and capacity",
                    "Consider connection timeout settings"
                ]
            )
            issues.append(issue)
        
        # Detect high latency
        if connection_info and connection_info.latency_ms and connection_info.latency_ms > 500:
            issue = self._create_issue(
                bot_id=bot_id,
                issue_type=ConnectionIssueType.HIGH_LATENCY,
                severity=IssueSeverity.MEDIUM,
                description=f"High connection latency ({connection_info.latency_ms:.1f}ms)",
                details={
                    'current_latency_ms': connection_info.latency_ms,
                    'latency_stats': pattern.latency_stats
                },
                recommendations=[
                    "Check network path to game server",
                    "Verify server geographic location",
                    "Review network quality and congestion",
                    "Consider using different network connection"
                ]
            )
            issues.append(issue)
        
        # Detect authentication failures
        auth_failures = [reason for reason, count in pattern.common_failure_reasons 
                        if reason == ConnectionFailureReason.AUTHENTICATION_FAILED]
        if auth_failures:
            issue = self._create_issue(
                bot_id=bot_id,
                issue_type=ConnectionIssueType.AUTHENTICATION_FAILURES,
                severity=IssueSeverity.HIGH,
                description="Bot experiencing authentication failures",
                details={
                    'auth_failure_count': auth_failures[0][1] if auth_failures else 0,
                    'failure_rate': pattern.failure_rate
                },
                recommendations=[
                    "Verify bot credentials and tokens",
                    "Check room access permissions",
                    "Review authentication flow",
                    "Ensure room passwords are correct"
                ]
            )
            issues.append(issue)
        
        # Detect server unavailability
        server_failures = [reason for reason, count in pattern.common_failure_reasons 
                          if reason == ConnectionFailureReason.SERVER_UNAVAILABLE]
        if server_failures:
            issue = self._create_issue(
                bot_id=bot_id,
                issue_type=ConnectionIssueType.SERVER_UNAVAILABLE,
                severity=IssueSeverity.CRITICAL,
                description="Game server appears to be unavailable",
                details={
                    'server_failure_count': server_failures[0][1] if server_failures else 0,
                    'failure_rate': pattern.failure_rate
                },
                recommendations=[
                    "Check game server status and availability",
                    "Verify server URL and port configuration",
                    "Review server logs for issues",
                    "Consider server restart or maintenance"
                ]
            )
            issues.append(issue)
        
        # Detect no activity issues
        if connection_info and connection_info.connection_state == WebSocketState.CONNECTED:
            if connection_info.last_message_received:
                time_since_message = datetime.now() - connection_info.last_message_received
                if time_since_message > timedelta(minutes=10):
                    issue = self._create_issue(
                        bot_id=bot_id,
                        issue_type=ConnectionIssueType.NO_ACTIVITY,
                        severity=IssueSeverity.HIGH,
                        description=f"No messages received for {time_since_message}",
                        details={
                            'time_since_last_message': str(time_since_message),
                            'messages_received': connection_info.messages_received,
                            'messages_sent': connection_info.messages_sent
                        },
                        recommendations=[
                            "Check if bot is properly joined to game",
                            "Verify game state synchronization",
                            "Review message handling logic",
                            "Check for game server issues"
                        ]
                    )
                    issues.append(issue)
        
        # Store detected issues
        if bot_id not in self._detected_issues:
            self._detected_issues[bot_id] = []
        
        # Add new issues
        for issue in issues:
            # Check if similar issue already exists
            existing_issue = self._find_similar_issue(bot_id, issue)
            if not existing_issue:
                self._detected_issues[bot_id].append(issue)
                
                # Log issue detection
                self.diagnostic_tracker.log_event(
                    bot_id=bot_id,
                    event_type=BotLifecycleEvent.BOT_ERROR,
                    level=DiagnosticLevel.WARN,
                    message=f"Connection issue detected: {issue.description}",
                    details={
                        'issue_type': issue.issue_type.value,
                        'severity': issue.severity.value,
                        'recommendations': issue.recommendations
                    }
                )
        
        return issues
    
    def get_bot_issues(self, bot_id: str, include_resolved: bool = False) -> List[ConnectionIssue]:
        """
        Get connection issues for a specific bot.
        
        Args:
            bot_id: Bot identifier
            include_resolved: Whether to include resolved issues
            
        Returns:
            List of connection issues
        """
        if bot_id not in self._detected_issues:
            return []
        
        issues = self._detected_issues[bot_id]
        if not include_resolved:
            issues = [issue for issue in issues if not issue.resolved]
        
        return issues
    
    def get_all_issues(self, include_resolved: bool = False) -> Dict[str, List[ConnectionIssue]]:
        """
        Get all connection issues across all bots.
        
        Args:
            include_resolved: Whether to include resolved issues
            
        Returns:
            Dictionary mapping bot_id to list of issues
        """
        result = {}
        for bot_id, issues in self._detected_issues.items():
            if not include_resolved:
                issues = [issue for issue in issues if not issue.resolved]
            if issues:
                result[bot_id] = issues
        
        return result
    
    def resolve_issue(self, bot_id: str, issue_id: str) -> bool:
        """
        Mark an issue as resolved.
        
        Args:
            bot_id: Bot identifier
            issue_id: Issue identifier
            
        Returns:
            True if issue was found and resolved
        """
        if bot_id not in self._detected_issues:
            return False
        
        for issue in self._detected_issues[bot_id]:
            if issue.issue_id == issue_id:
                issue.resolved = True
                issue.resolved_at = datetime.now()
                
                # Log issue resolution
                self.diagnostic_tracker.log_event(
                    bot_id=bot_id,
                    event_type=BotLifecycleEvent.BOT_ACTIVE,
                    level=DiagnosticLevel.INFO,
                    message=f"Connection issue resolved: {issue.description}",
                    details={
                        'issue_type': issue.issue_type.value,
                        'issue_id': issue_id
                    }
                )
                
                return True
        
        return False
    
    def generate_connection_report(self, bot_id: str) -> Dict[str, Any]:
        """
        Generate comprehensive connection report for a bot.
        
        Args:
            bot_id: Bot identifier
            
        Returns:
            Detailed connection report
        """
        # Get connection information
        connection_info = self.websocket_monitor.get_connection_info(bot_id)
        connection_health = self.websocket_monitor.get_connection_health(bot_id)
        pattern = self._connection_patterns.get(bot_id)
        issues = self.get_bot_issues(bot_id)
        
        # Build report
        report = {
            'bot_id': bot_id,
            'report_generated_at': datetime.now().isoformat(),
            'connection_status': {
                'current_state': connection_info.connection_state.value if connection_info else 'unknown',
                'is_healthy': connection_health.get('healthy', False) if connection_health else False,
                'connection_info': connection_info.to_dict() if connection_info else None,
                'health_details': connection_health
            },
            'connection_pattern': pattern.to_dict() if pattern else None,
            'active_issues': [issue.to_dict() for issue in issues],
            'issue_summary': {
                'total_issues': len(issues),
                'critical_issues': len([i for i in issues if i.severity == IssueSeverity.CRITICAL]),
                'high_issues': len([i for i in issues if i.severity == IssueSeverity.HIGH]),
                'medium_issues': len([i for i in issues if i.severity == IssueSeverity.MEDIUM]),
                'low_issues': len([i for i in issues if i.severity == IssueSeverity.LOW])
            },
            'recommendations': self._generate_recommendations(bot_id, issues, pattern)
        }
        
        return report
    
    def _create_issue(self, bot_id: str, issue_type: ConnectionIssueType,
                     severity: IssueSeverity, description: str,
                     details: Dict[str, Any], recommendations: List[str]) -> ConnectionIssue:
        """Create a new connection issue."""
        self._issue_counter += 1
        issue_id = f"issue_{self._issue_counter}_{int(datetime.now().timestamp())}"
        
        return ConnectionIssue(
            issue_id=issue_id,
            bot_id=bot_id,
            issue_type=issue_type,
            severity=severity,
            detected_at=datetime.now(),
            description=description,
            details=details,
            recommendations=recommendations
        )
    
    def _find_similar_issue(self, bot_id: str, new_issue: ConnectionIssue) -> Optional[ConnectionIssue]:
        """Find if a similar issue already exists."""
        if bot_id not in self._detected_issues:
            return None
        
        for existing_issue in self._detected_issues[bot_id]:
            if (existing_issue.issue_type == new_issue.issue_type and 
                not existing_issue.resolved and
                (datetime.now() - existing_issue.detected_at) < timedelta(hours=1)):
                return existing_issue
        
        return None
    
    def _generate_recommendations(self, bot_id: str, issues: List[ConnectionIssue],
                                pattern: Optional[ConnectionPattern]) -> List[str]:
        """Generate overall recommendations based on issues and patterns."""
        recommendations = set()
        
        # Add recommendations from issues
        for issue in issues:
            recommendations.update(issue.recommendations)
        
        # Add pattern-based recommendations
        if pattern:
            if pattern.failure_rate > 0.5:
                recommendations.add("Consider reviewing overall bot connection logic")
            
            if pattern.total_connections > 20:
                recommendations.add("Monitor for connection resource exhaustion")
        
        return sorted(list(recommendations))
    
    async def _periodic_issue_detection(self) -> None:
        """Periodic issue detection for all monitored bots."""
        while self._running:
            try:
                # Get all monitored connections
                connections = self.websocket_monitor.get_all_connections()
                
                # Detect issues for each bot
                for bot_id in connections.keys():
                    try:
                        await self.detect_connection_issues(bot_id)
                    except Exception as e:
                        self.logger.error(f"Error detecting issues for bot {bot_id}: {e}")
                
                # Wait for next detection cycle
                await asyncio.sleep(self.issue_detection_interval)
                
            except Exception as e:
                self.logger.error(f"Error in periodic issue detection: {e}")
                await asyncio.sleep(self.issue_detection_interval)


# Global connection diagnostics instance
_connection_diagnostics: Optional[ConnectionDiagnostics] = None


def get_connection_diagnostics() -> ConnectionDiagnostics:
    """Get the global connection diagnostics instance."""
    global _connection_diagnostics
    if _connection_diagnostics is None:
        _connection_diagnostics = ConnectionDiagnostics()
    return _connection_diagnostics


def set_connection_diagnostics(diagnostics: ConnectionDiagnostics) -> None:
    """Set the global connection diagnostics instance."""
    global _connection_diagnostics
    _connection_diagnostics = diagnostics