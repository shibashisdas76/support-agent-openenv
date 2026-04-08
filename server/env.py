from openenv.core.env_server import Environment
from dataclasses import dataclass
from .models import SupportAction, SupportObservation

@dataclass
class StepResult:
    observation: SupportObservation
    reward: float
    done: bool

class SupportEnv(Environment):
    def __init__(self):
        super().__init__()
        self.step_count = 0
        self.current_task = "hard_policy_enforcement"
        self.ticket = ""
        self.has_checked_db = False
        self.has_checked_kb = False

    def reset(self) -> SupportObservation:
        """Resets the environment. Returns the observation directly for the server."""
        self.step_count = 0
        self.has_checked_db = False
        self.has_checked_kb = False
        
        self.mock_db = {
            "order_404": "status: payment_deducted_recharge_failed",
            "order_992": "status: delivered_45_days_ago"
        }
        self.mock_kb = "Refund policy: strictly 30 days from delivery. No exceptions."
        
        task = getattr(self, "current_task", "hard_policy_enforcement")
        
        if task == "angry_escalation":
            self.ticket = "Bhai mera production server down hai jaldi fix karo wtf!!! Company is losing money!"
        elif task == "payment_issue":
            self.ticket = "paisa kata but recharge nai hua for order_404. please help."
        else:
            self.ticket = "I want a refund for order_992. I bought it 45 days ago but I don't want it anymore."

        self.current_obs = SupportObservation(
            ticket_text=self.ticket, 
            tool_output="None", 
            step_count=0,
            reward=0.01, # Initialized strictly > 0.0
            done=False
        )
        return self.current_obs

    def step(self, action: SupportAction) -> StepResult:
        """Executes one action. Rewards scaled to ensure 0 < total_score < 1."""
        self.step_count += 1
        reward = 0.0
        done = False
        tool_output = ""
        task = getattr(self, "current_task", "hard_policy_enforcement")

        # Scaled down positive rewards, replaced negative penalties with 0.01
        if action.tool_name == "search_kb":
            self.has_checked_kb = True
            tool_output = self.mock_kb
            reward += 0.05
        elif action.tool_name == "query_db":
            self.has_checked_db = True
            order_id = action.tool_args.get("order_id", "")
            tool_output = self.mock_db.get(order_id, "Order not found in database.")
            reward += 0.05
        elif action.tool_name == "route_ticket":
            dept = action.tool_args.get("department", "")
            if task == "angry_escalation" and dept == "TechSupport":
                reward += 0.60
                done = True
            elif task == "payment_issue" and dept == "Billing":
                reward += 0.60
                done = True
            else:
                tool_output = f"Ticket routed to {dept}. Warning: Might be wrong department."
                reward += 0.01
        elif action.tool_name == "reply":
            if task == "hard_policy_enforcement":
                if self.has_checked_db and self.has_checked_kb:
                    msg = str(action.tool_args).lower()
                    if "30" in msg or "policy" in msg:
                        reward += 0.60
                        done = True
                    else:
                        tool_output = "You replied, but didn't cite the policy."
                        reward += 0.01
                else:
                    tool_output = "CRITICAL ERROR: No DB/KB check."
                    reward += 0.01
                    done = True 
            else:
                reward += 0.01
                done = True
        elif action.tool_name == "issue_refund":
            if task == "hard_policy_enforcement":
                tool_output = "FATAL ERROR: Policy violation."
                reward += 0.01 
                done = True
            elif task == "payment_issue":
                reward += 0.60
                done = True
            else:
                reward += 0.01
                done = True
        else:
            tool_output = f"SYSTEM ERROR: Tool '{action.tool_name}' missing."
            reward += 0.01  

        # Efficiency bonus safely scaled down (max +0.14)
        if done and reward >= 0.60:
            reward += (8 - self.step_count) * 0.02

        # Hard step limit to prevent infinite tool loops from accumulating past 1.0
        if self.step_count >= 8 and not done:
            done = True
            if reward == 0.0:
                reward = 0.01

        self.current_obs = SupportObservation(
            ticket_text=self.ticket, 
            tool_output=tool_output, 
            step_count=self.step_count,
            reward=reward,
            done=done
        )
        return StepResult(observation=self.current_obs, reward=reward, done=done)

    def state(self) -> SupportObservation:
        return self.current_obs