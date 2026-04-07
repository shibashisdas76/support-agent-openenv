import asyncio
import os
import json
from openai import OpenAI
from server.env import SupportEnv, StepResult
from server.models import SupportAction

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
API_KEY = os.getenv("HF_TOKEN")

SYSTEM_PROMPT = """You are an expert SaaS customer support agent. Handle Hinglish/English.
SOP:
- Outages -> Route TechSupport.
- Failed recharge (order_404) -> Issue refund.
- Refunds (order_992) -> Check DB & KB, then Reply enforcing 30-day policy.
JSON ONLY."""

def extract_json(llm_output: str) -> dict:
    start_idx = llm_output.find('{')
    end_idx = llm_output.rfind('}')
    if start_idx != -1 and end_idx != -1:
        return json.loads(llm_output[start_idx:end_idx+1])
    return json.loads(llm_output)

async def main():
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    env = SupportEnv()
    tasks = ["angry_escalation", "payment_issue", "hard_policy_enforcement"]
    
    for task in tasks:
        env.current_task = task
        print(f"[START] task={task} env=support_agent", flush=True)
        
        # Fixed: env.reset() is now sync
        obs = env.reset()
        done = False
        step = 0
        rewards = []
        
        while not done and step < 8:
            step += 1
            prompt = f"Ticket: {obs.ticket_text}\nLast Tool Output: {obs.tool_output}\nAction?"
            
            try:
                response = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[{"role": "system", "content": SYSTEM_PROMPT},
                              {"role": "user", "content": prompt}],
                    temperature=0.1
                )
                action_dict = extract_json(response.choices[0].message.content)
                action = SupportAction(**action_dict)
                error_val = "null"
            except:
                action = SupportAction(tool_name="reply", tool_args={"message": "Error"})
                error_val = "json_parse_error"

            # Fixed: env.step() is now sync
            result = env.step(action)
            obs = result.observation
            rewards.append(result.reward)
            done = result.done
            
            print(f"[STEP] step={step} action={action.tool_name} reward={result.reward:.2f} done={str(done).lower()}", flush=True)
            
        score = max(0.0, min(sum(rewards), 1.0))
        print(f"[END] success={str(score > 0.5).lower()} score={score:.3f}", flush=True)

if __name__ == "__main__":
    asyncio.run(main())