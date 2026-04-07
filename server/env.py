from openenv.core.env_server import Environment
from dataclasses import dataclass
from .models import SupportAction, SupportObservation

@dataclass
class StepResult:
    observation: SupportObservation
    reward: float
    done: bool

class SupportEnv(Environment):
    def reset(self) -> SupportObservation:
        """
        Resets the environment for a new episode.
        IMPORTANT: Must return SupportObservation, NOT StepResult.
        """
        self.step_count = 0
        self.has_checked_db = False
        self.has_checked_kb = False
        
        self.mock_db = {
            "order_404": "status: payment_deducted_recharge_failed",
            "order_992": "status: delivered_45_days_ago"
        }
        self.mock_kb = "Refund policy: strictly 30 days from delivery. No exceptions."
        
        # Determine ticket based on task
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
            step_count=0
        )
        
        # The server calls .model_dump() on this return value.
        # SupportObservation (Pydantic) has it; StepResult (Dataclass) does not.
        return self.current_obs

    def step(self, action: SupportAction) -> StepResult:
        """
        Executes one action in the environment.
        Returns StepResult (Observation, Reward, Done).
        """
        self.step_count += 1
        reward = 0.0
        done = False
        tool_output = ""
        task = getattr(self, "current_task", "hard_policy_enforcement")

        if action.tool_name == "search_kb":
            self.has_checked_kb = True
            tool_output = self.mock_kb
            reward += 0.1
            
        elif action.tool_name == "query_db":
            self.has_checked_db = True
            order_id = action.tool_args.get("order_id", "")
            tool_output = self.mock_db.get(order_id, "Order not found in database.")
            reward += 0.1
            
        elif action.tool_name == "route_ticket":
            dept = action.tool_args.get("department", "")
            if task == "angry_escalation" and dept == "TechSupport":
                reward += 0.8
                done = True
            elif task == "payment_issue" and dept == "Billing":
                reward += 0.8
                done = True
            else:
                tool_output = f"Ticket routed to {dept}. Warning: Might be wrong department."
                reward -= 0.3

        elif action.tool_name == "reply":
            if task == "hard_policy_enforcement":
                if self.has_checked_db and self.has_checked_kb:
                    msg = str(action.tool_args).lower()
                    if "30" in msg or "policy" in msg:
                        reward += 0.8
                        done = True
                    else:
                        tool_output = "You replied, but didn't cite the policy."
                        reward -= 0.3
                else:
                    tool_output = "CRITICAL ERROR: You replied without checking the Database and Knowledge Base first."
                    reward -= 0.5
            else:
                done = True
                
        elif action.tool_name == "issue_refund":
            if task == "hard_policy_enforcement":
                tool_output = "FATAL ERROR: You violated company policy and issued a refund after 30 days."
                reward -= 1.0 
                done = True
            elif task == "payment_issue":
                reward += 0.8
                done = True
                
        else:
            tool_output = f"SYSTEM ERROR: Tool '{action.tool_name}' does not exist."
            reward -= 0.2  

        # Efficiency Bonus
        if done and reward > 0.5:
            efficiency_bonus = (8 - self.step_count) * 0.05
            reward += efficiency_bonus

        self.current_obs = SupportObservation(
            ticket_text=self.ticket, 
            tool_output=tool_output, 
            step_count=self.step_count
        )
        return StepResult(observation=self.current_obs, reward=reward, done=done)

    def state(self) -> SupportObservation:
        return getattr(self, 'current_obs', None)