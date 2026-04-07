import asyncio
import os
import json
from openai import OpenAI
from server.env import SupportEnv
from server.models import SupportAction

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
API_KEY = os.getenv("HF_TOKEN")

SYSTEM_PROMPT = """You are an expert SaaS customer support agent. You handle global and Indian users (Hinglish).

AVAILABLE TOOLS:
1. {"tool_name": "search_kb", "tool_args": {"query": "..."}}
2. {"tool_name": "query_db", "tool_args": {"order_id": "..."}}
3. {"tool_name": "route_ticket", "tool_args": {"department": "TechSupport/Billing"}}
4. {"tool_name": "reply", "tool_args": {"message": "..."}}
5. {"tool_name": "issue_refund", "tool_args": {"order_id": "..."}}

STANDARD OPERATING PROCEDURE (SOP):
- Server Down / Outages -> Route to TechSupport immediately.
- Recharge failed but payment deducted -> Issue refund.
- Policy/Return questions -> MUST query_db AND search_kb first, then reply enforcing the 30-day policy.

INSTRUCTIONS:
You must think step-by-step. First, write down your reasoning. Then, provide the exact JSON tool call.

Example:
I see the user has an issue with a billing duplicate. I need to check the database first.
{"tool_name": "query_db", "tool_args": {"order_id": "123"}}
"""

# Safe JSON Extractor (No Regex errors in VS Code!)
def extract_json(llm_output: str) -> dict:
    start_idx = llm_output.find('{')
    end_idx = llm_output.rfind('}')
    if start_idx != -1 and end_idx != -1:
        clean_json = llm_output[start_idx:end_idx+1]
        return json.loads(clean_json)
    return json.loads(llm_output)

async def main():
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    env = SupportEnv()
    
    # Updated tasks to match our new SaaS Scenarios
    tasks = ["angry_escalation", "payment_issue", "hard_policy_enforcement"]
    
    for task in tasks:
        env.current_task = task
        print(f"[START] task={task} env=support_agent model={MODEL_NAME}", flush=True)
        
        result = await env.reset()
        done = False
        step = 0
        rewards = []
        
        while not done and step < 8:
            step += 1
            obs = result.observation
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
            except Exception as e:
                action = SupportAction(tool_name="reply", tool_args={"message": "Error parsing."})
                error_val = "json_parse_error"

            result = await env.step(action)
            rewards.append(result.reward)
            done = result.done
            
            print(f"[STEP] step={step} action={action.tool_name} reward={result.reward:.2f} done={str(done).lower()} error={error_val}", flush=True)
            
        score = min(sum(rewards), 1.0) 
        score = max(score, 0.0) # Prevent negative scores
        success = score > 0.5
        rewards_str = ",".join(f"{r:.2f}" for r in rewards)
        print(f"[END] success={str(success).lower()} steps={step} score={score:.3f} rewards={rewards_str}", flush=True)

if __name__ == "__main__":
    asyncio.run(main())