"""
Operations Audit Logger
"""

import logging
import json
from typing import Dict, Any, List
from datetime import datetime
from dataclasses import dataclass, asdict
from enum import Enum
import os

class OperationType(Enum):
    RESTART = "restart"
    RESYNC = "resync"
    RETRAIN = "retrain"

class OperationStatus(Enum):
    INITIATED = "initiated"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"

@dataclass
class OperationContext:
    user_id: str
    timestamp: str
    operation_type: str
    operation_id: str
    reason: str
    status: str

class OpsAuditLogger:
    def __init__(self, log_file: str = "logs/ops_audit.log"):
        self.logger = logging.getLogger(__name__)
        self.log_file = log_file
        self.operations = {}
        self.session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        self._setup_audit_logger()
    
    def _setup_audit_logger(self):
        audit_logger = logging.getLogger('ops_audit')
        audit_logger.setLevel(logging.INFO)
        
        for handler in audit_logger.handlers[:]:
            audit_logger.removeHandler(handler)
        
        file_handler = logging.FileHandler(self.log_file)
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(message)s')
        file_handler.setFormatter(formatter)
        audit_logger.addHandler(file_handler)
        audit_logger.propagate = False
    
    def log_operation_start(self, operation_type: OperationType, reason: str) -> str:
        try:
            operation_id = f"{operation_type.value}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            operation_context = OperationContext(
                user_id=os.getenv('USER', 'unknown'),
                timestamp=datetime.now().isoformat(),
                operation_type=operation_type.value,
                operation_id=operation_id,
                reason=reason,
                status=OperationStatus.INITIATED.value
            )
            
            self.operations[operation_id] = operation_context
            self._log_audit_event(operation_context)
            
            self.logger.info(f"🔧 OPERATION_STARTED: {operation_type.value} - {reason} (ID: {operation_id})")
            return operation_id
            
        except Exception as e:
            self.logger.error(f"❌ Error logging operation start: {e}")
            return ""
    
    def log_operation_complete(self, operation_id: str, status: OperationStatus = OperationStatus.COMPLETED):
        try:
            if operation_id not in self.operations:
                return
            
            operation = self.operations[operation_id]
            operation.status = status.value
            
            self._log_audit_event(operation)
            
            status_emoji = "✅" if status == OperationStatus.COMPLETED else "❌"
            self.logger.info(f"{status_emoji} OPERATION_COMPLETE: {operation_id} - {status.value}")
            
        except Exception as e:
            self.logger.error(f"❌ Error logging operation completion: {e}")
    
    def _log_audit_event(self, operation: OperationContext):
        try:
            audit_logger = logging.getLogger('ops_audit')
            audit_event = {
                "event_type": "operation_audit",
                "operation_context": asdict(operation)
            }
            audit_logger.info(json.dumps(audit_event))
        except Exception as e:
            self.logger.error(f"❌ Error logging audit event: {e}")
    
    def get_operation_statistics(self) -> Dict[str, Any]:
        return {
            "total_operations": len(self.operations),
            "session_id": self.session_id,
            "last_updated": datetime.now().isoformat()
        }

def demo_ops_audit_logger():
    print("📋 Operations Audit Logger Demo")
    print("=" * 50)
    
    audit_logger = OpsAuditLogger("logs/demo_audit.log")
    
    # Simulate operations
    restart_id = audit_logger.log_operation_start(OperationType.RESTART, "System restart")
    audit_logger.log_operation_complete(restart_id, OperationStatus.COMPLETED)
    
    resync_id = audit_logger.log_operation_start(OperationType.RESYNC, "WebSocket reconnection")
    audit_logger.log_operation_complete(resync_id, OperationStatus.COMPLETED)
    
    # Show statistics
    stats = audit_logger.get_operation_statistics()
    print(f"📊 Operation Statistics: {stats}")
    
    print("✅ Operations Audit Logger Demo Complete")

if __name__ == "__main__":
    demo_ops_audit_logger()
