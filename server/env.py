import random
import re
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
        self.last_action = None

    # --------------------------
    # 🔀 NOISE GENERATOR
    # --------------------------
    def _add_noise(self, text: str) -> str:
        if random.random() < 0.3:
            text = text.upper()
        if random.random() < 0.3:
            text = text.replace("a", "@")
        if random.random() < 0.2:
            text = text.replace("o", "0")
        return text

    # --------------------------
    # 🔎 ORDER ID EXTRACTION
    # --------------------------
    def _extract_order_id(self, text: str):
        match = re.search(r'order_\d+', text)
        return match.group(0) if match else None

    def reset(self) -> SupportObservation:
        self.step_count = 0
        self.has_checked_db = False
        self.has_checked_kb = False
        self.last_action = None

        # --------------------------
        # 🔀 STOCHASTIC DB
        # --------------------------
        self.mock_db = {
            "order_404": "status: payment_deducted_recharge_failed",
            "order_992": "status: delivered_45_days_ago",
            "order_777": "status: delivered_10_days_ago"
        }

        self.mock_kb = "Refund policy: strictly 30 days from delivery. No exceptions."

        task = getattr(self, "current_task", "hard_policy_enforcement")

        # --------------------------
        # 🔀 MULTIPLE TICKET VARIANTS
        # --------------------------
        if task == "angry_escalation":
            tickets = [
                "Bhai server down hai jaldi fix karo!!!",
                "Production is down!!! losing money fast",
                "wtf server not working fix ASAP"
            ]
        elif task == "payment_issue":
            tickets = [
                "paisa kata but recharge nai hua for order_404",
                "money deducted recharge failed order_404",
                "amount debited but no recharge happened"
            ]
        else:
            tickets = [
                "refund chahiye for order_992 bought 45 days ago",
                "I want refund for my order after 45 days",
                "pls refund order_992 I don't need it anymore"
            ]

        self.ticket = self._add_noise(random.choice(tickets))

        self.current_obs = SupportObservation(
            ticket_text=self.ticket,
            tool_output="None",
            step_count=0,
            reward=0.01,
            done=False
        )
        return self.current_obs

    def step(self, action: SupportAction) -> StepResult:
        self.step_count += 1
        reward = 0.0
        done = False
        tool_output = ""
        task = getattr(self, "current_task", "hard_policy_enforcement")

        tool_args = action.tool_args or {}

        # --------------------------
        # 🚫 ANTI-LOOP PENALTY
        # --------------------------
        if self.last_action == action.tool_name:
            reward += 0.01
            tool_output += "Repeated action detected. "

        self.last_action = action.tool_name

        # --------------------------
        # 🧠 TOOL LOGIC
        # --------------------------
        if action.tool_name == "search_kb":
            self.has_checked_kb = True
            tool_output = self.mock_kb
            reward += 0.05

        elif action.tool_name == "query_db":
            self.has_checked_db = True

            order_id = tool_args.get("order_id") or self._extract_order_id(self.ticket)

            if not order_id:
                tool_output = "ERROR: Missing order_id. Try extracting from ticket."
                reward += 0.01
            else:
                tool_output = self.mock_db.get(order_id, "Order not found in database.")
                reward += 0.05

        elif action.tool_name == "route_ticket":
            dept = tool_args.get("department", "")

            if task == "angry_escalation" and dept == "TechSupport":
                tool_output = "Ticket routed to TechSupport successfully."
                reward += 0.60
                done = True

            elif task == "payment_issue" and dept == "Billing":
                tool_output = "Ticket routed to Billing."
                reward += 0.60
                done = True

            else:
                tool_output = f"Wrong department: {dept}. Hint: Check intent."
                reward += 0.01

        elif action.tool_name == "reply":
            if task == "hard_policy_enforcement":
                if self.has_checked_db and self.has_checked_kb:
                    msg = str(tool_args).lower()

                    if "30" in msg or "policy" in msg:
                        tool_output = "Correct policy enforcement."
                        reward += 0.60
                        done = True
                    else:
                        tool_output = "Missing policy reference. Mention 30-day rule."
                        reward += 0.01
                else:
                    tool_output = "ERROR_NO_CONTEXT: Must check DB and KB first."
                    reward += 0.01
                    done = True
            else:
                tool_output = "Reply sent."
                reward += 0.01
                done = True

        elif action.tool_name == "issue_refund":
            order_id = tool_args.get("order_id")

            if task == "hard_policy_enforcement":
                tool_output = "FATAL: Refund violates policy."
                reward += 0.01
                done = True

            elif task == "payment_issue":
                tool_output = f"Refund issued for {order_id}."
                reward += 0.60
                done = True

            else:
                tool_output = "Refund not applicable."
                reward += 0.01
                done = True

        else:
            tool_output = f"INVALID_TOOL: {action.tool_name}"
            reward += 0.01

        # --------------------------
        # ⚡ EFFICIENCY BONUS
        # --------------------------
        if done and reward >= 0.60:
            reward += (8 - self.step_count) * 0.02

        # --------------------------
        # ⛔ HARD STEP LIMIT
        # --------------------------
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

        return StepResult(
            observation=self.current_obs,
            reward=reward,
            done=done
        )

    def state(self) -> SupportObservation:
        return self.current_obs