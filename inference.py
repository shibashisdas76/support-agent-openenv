import asyncio
import os
import json
from openai import OpenAI
from server.env import SupportEnv
from server.models import SupportAction

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
API_KEY = os.getenv("HF_TOKEN")

# UPDATED: Added the specific conclusion logic to prevent the "infinite loop"
SYSTEM_PROMPT = """You are an expert SaaS customer support agent. 

CRITICAL RULES:
1. NEVER use the 'reply' tool as your first action.
2. For 'angry_escalation' (Server Down): You MUST use 'route_ticket' with department='TechSupport' immediately.
3. For 'payment_issue': You MUST use 'query_db' to find order details, then 'issue_refund'.
4. For 'hard_policy_enforcement': You MUST use 'query_db' AND 'search_kb' before you even think about using 'reply'.
5. CONCLUSION: Once you have found the information you need from the tools, you MUST conclude the task by using 'issue_refund' for payments or 'reply' for policy questions to set 'done' to true.

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
        print(f"\n[START] task={task} env=support_agent", flush=True)
        
        # NOTE: reset() and step() are synchronous based on our previous server-compatibility fixes
        obs = env.reset() 
        done = False
        step = 0
        rewards = []
        
        while not done and step < 8:
            step += 1
            # Tool output and ticket text accessed directly from the Pydantic observation object
            prompt = f"Ticket: {obs.ticket_text}\nLast Tool Output: {obs.tool_output}\nWhat is your next step?"
            
            try:
                response = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.1
                )
                raw_reply = response.choices[0].message.content.strip()
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
        print(f"[END] success={str(success).lower()} steps={step} score={score:.3f}", flush=True)

if __name__ == "__main__":
    asyncio.run(main())