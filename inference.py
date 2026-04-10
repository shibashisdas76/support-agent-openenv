import asyncio
import os
import json
import re  # Added for safer JSON extraction
from openai import OpenAI
from server.env import SupportEnv
from server.models import SupportAction

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
API_KEY = os.getenv("HF_TOKEN")

# MANDATORY CHECK: Required by Hackathon Guidelines
if API_KEY is None:
    raise ValueError("HF_TOKEN environment variable is required")

# UPDATED: Added the specific conclusion logic to prevent the "infinite loop"
SYSTEM_PROMPT = """You are an expert SaaS customer support agent. 

CRITICAL RULES:
1. NEVER use the 'reply' tool as your first action.
2. For 'angry_escalation' (Server Down): You MUST use 'route_ticket' with department='TechSupport' immediately.
3. For 'payment_issue': You MUST use 'query_db' to find order details, then 'issue_refund'.
4. For 'hard_policy_enforcement': You MUST use 'query_db' AND 'search_kb' before taking any action. If an order violates the 30-day refund policy, you MUST NOT use 'issue_refund'. You must use 'reply' to deny it.
5. CONCLUSION: Once you have found the information you need, you MUST conclude the task by using 'issue_refund' (if valid) or 'reply' (if denying/explaining) to set 'done' to true.

AVAILABLE TOOLS:
1. {"tool_name": "search_kb", "tool_args": {"query": "..."}}
2. {"tool_name": "query_db", "tool_args": {"order_id": "..."}}
3. {"tool_name": "route_ticket", "tool_args": {"department": "TechSupport/Billing"}}
4. {"tool_name": "reply", "tool_args": {"message": "..."}}
5. {"tool_name": "issue_refund", "tool_args": {"order_id": "..."}}

INSTRUCTIONS:
Think step-by-step. Write your reasoning first, then the JSON tool call.
"""

def extract_json(llm_output: str) -> dict:
    try:
        # First, try to use Regex to find the tool dictionary exactly
        match = re.search(r'(\{.*"tool_name".*\})', llm_output, re.DOTALL)
        if match:
            return json.loads(match.group(1))
            
        # Fallback to your original logic
        start_idx = llm_output.find('{')
        end_idx = llm_output.rfind('}')
        if start_idx != -1 and end_idx != -1:
            clean_json = llm_output[start_idx:end_idx+1]
            return json.loads(clean_json)
        return json.loads(llm_output)
    except Exception:
        return {"tool_name": "reply", "tool_args": {"message": "Invalid JSON format."}}

async def main():
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    env = SupportEnv()
    
    tasks = ["angry_escalation", "payment_issue", "hard_policy_enforcement"]
    
    for task in tasks:
        env.current_task = task
        
        # FORMAT FIX: Removed '\n' and added 'model=' per guidelines
        print(f"[START] task={task} env=support_agent model={MODEL_NAME}", flush=True)
        
        # NOTE: reset() and step() are synchronous based on our previous server-compatibility fixes
        obs = env.reset() 
        done = False
        step = 0
        rewards = []
        
        # Initialize conversation memory outside the loop
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        
        while not done and step < 8:
            step += 1
            # Tool output and ticket text accessed directly from the Pydantic observation object
            prompt = f"Ticket: {obs.ticket_text}\nLast Tool Output: {obs.tool_output}\nWhat is your next step?"
            
            # Add user prompt to memory
            messages.append({"role": "user", "content": prompt})
            
            try:
                response = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=messages,  # Send the full memory history
                    temperature=0.1
                )
                raw_reply = response.choices[0].message.content.strip()
                
                # Add agent's reply to memory so it remembers what it just did
                messages.append({"role": "assistant", "content": raw_reply})
                
                action_dict = extract_json(raw_reply)
                action = SupportAction(**action_dict)
                error_val = "null"
            except Exception:
                action = SupportAction(tool_name="reply", tool_args={"message": "Error parsing response."})
                error_val = "json_parse_error"

            # Execute step (Synchronous)
            result = env.step(action)
            obs = result.observation
            rewards.append(result.reward)
            done = result.done
            
            print(f"[STEP] step={step} action={action.tool_name} reward={result.reward:.2f} done={str(done).lower()} error={error_val}", flush=True)
            
        score = max(0.0, min(sum(rewards), 1.0))
        success = score > 0.5
        
        # FORMAT FIX: Created comma-separated rewards string formatted to 2 decimal places
        rewards_str = ",".join([f"{r:.2f}" for r in rewards])
        print(f"[END] success={str(success).lower()} steps={step} rewards={rewards_str}", flush=True)

if __name__ == "__main__":
    asyncio.run(main())