---
title: Support Agent OpenEnv
emoji: 🚀
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
license: mit
---
🏆 Tier-1 SaaS Support Resolver (OpenEnv)
OpenEnv Compliant Difficulty Domain

🚀 Environment Description & Motivation
Customer support automation is one of the highest ROI applications for LLMs, representing a multi-billion dollar real-world use case. However, deploying agents in production is risky: they often hallucinate policies, blindly issue refunds, or fail to understand noisy, multilingual user inputs.

The Support Resolver Environment simulates a high-stakes, Tier-1 Enterprise SaaS support desk. It forces agents to navigate strict Standard Operating Procedures (SOPs), query mock databases, reference knowledge bases, and handle culturally realistic noisy inputs (Hinglish). It rigorously evaluates an agent's ability to use Chain-of-Thought reasoning and penalizes destructive actions (e.g., unauthorized refunds).

⚙️ Observation & Action Spaces
This environment strictly implements the OpenEnv specification using Pydantic typed models.

Observation Space (SupportObservation)
At each step, the agent receives a clean, deterministic state:

ticket_text (str): The raw, incoming message from the customer (includes capitalization noise and Hinglish).
tool_output (str): The result of the previous action (e.g., database records, policy text, or error messages).
step_count (int): The current step number in the episode (max 8).
Action Space (SupportAction)
The agent interacts with the SaaS ecosystem via 5 discrete tools:

search_kb({"query": str}): Searches the internal policy document.
query_db({"order_id": str}): Fetches real-time status from the mock database.
route_ticket({"department": str}): Escalates the ticket (e.g., "TechSupport", "Billing").
reply({"message": str}): Sends a direct text response to the customer.
issue_refund({"order_id": str}): Executes a financial refund.
🎯 Task Progression & Grader Design
The environment contains 3 meticulously graded tasks with programmatic success/failure criteria.

1. angry_escalation (Difficulty: Easy)

Objective: Recognize an out-of-scope, high-severity issue (Server Down).
Logic: The agent must ignore the aggressive tone and immediately use route_ticket to "TechSupport".
2. payment_issue (Difficulty: Medium)

Objective: Handle noisy/multilingual inputs ("paisa kata but recharge nai hua").
Logic: The agent must map the Hinglish complaint to a failed transaction for order_404, cross-reference the SOP, and confidently execute issue_refund.
3. hard_policy_enforcement (Difficulty: Hard)

Objective: Strict SOP dependency adherence.
Logic: The customer demands a refund for a 45-day-old order. The agent MUST execute query_db AND search_kb first. If it replies without checking both, it receives a -0.5 penalty. If it issues a refund, it receives a fatal -1.0 penalty. It must correctly cite the 30-day policy via reply to succeed.
🧠 Advanced Reward Shaping & Penalties
To prevent reward hacking and encourage optimal agentic behavior, the reward function is highly dynamic:

Information Gathering: +0.1 for successfully querying the DB or KB.
Hallucination Penalty: -0.2 for attempting to use a tool that doesn't exist.
SOP Violation Penalty: -0.5 to -1.0 for acting without gathering required context.
Efficiency Bonus: (8 - steps_taken) * 0.05 added to the final score for resolving the ticket swiftly, incentivizing direct logic over endless looping.
📊 Baseline Scores
Evaluated using the baseline inference script with Qwen/Qwen2.5-72B-Instruct (Temperature: 0.1, Max Steps: 8).

Task 1 (Easy): Score: 1.000 (Resolved in 1 step via routing).
Task 2 (Medium): Score: 1.000 (Resolved in 1 step via confident refund logic).
Task 3 (Hard): Score: 1.000 (Resolved in 3 steps: DB Check -> KB Check -> Policy Enforcement Reply).
Note: The environment is highly sensitive to the agent's system prompt. Without strict Chain-of-Thought instructions, frontier models frequently fail the Hard task by issuing unauthorized refunds or looping infinitely.

💻 Setup and Usage Instructions
Local Execution
Prerequisites: Python 3.10+
Install Dependencies:
pip install openenv-core openai pydantic
Set Environment Variables:

Bash export HF_TOKEN="your_huggingface_access_token" (On Windows PowerShell use: $env:HF_TOKEN="your_token")

Run the Baseline Evaluator:

Bash python inference.py Docker Execution The environment is fully containerized for isolated testing and Hugging Face Space deployment.

Bash

Build the image
docker build -t openenv-support-agent .

Run the container
docker run openenv-support-agent