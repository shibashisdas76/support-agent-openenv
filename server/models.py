from pydantic import BaseModel
from typing import Optional, Dict

class SupportAction(BaseModel):
    tool_name: str  # 'search_kb', 'query_db', 'issue_refund', 'route_ticket', 'reply'
    tool_args: Dict[str, str]

class SupportObservation(BaseModel):
    ticket_text: str
    tool_output: Optional[str] = None
    step_count: int