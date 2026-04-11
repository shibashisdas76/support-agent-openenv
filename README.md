---
title: Support Agent OpenEnv
emoji: 🚀
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
license: mit
---

# 🏆 Tier-1 SaaS Support Resolver (OpenEnv)
**OpenEnv Compliant Difficulty Domain**

## 🚀 Environment Description & Motivation
Customer support automation is one of the highest ROI applications for LLMs, representing a multi-billion dollar real-world use case. However, deploying agents in production is risky: they often hallucinate policies, blindly issue refunds, or fail to understand noisy, multilingual user inputs.

The Support Resolver Environment simulates a high-stakes, Tier-1 Enterprise SaaS support desk. It forces agents to navigate strict Standard Operating Procedures (SOPs), query mock databases, reference knowledge bases, and handle culturally realistic noisy inputs (Hinglish/Typos). It rigorously evaluates an agent's ability to use Chain-of-Thought reasoning and heavily penalizes destructive actions (e.g., unauthorized refunds).

## ⚙️ Observation & Action Spaces
This environment strictly implements the OpenEnv specification using Pydantic typed models.

### Observation Space (`SupportObservation`)
At each step, the agent receives a clean, deterministic state:
* `ticket_text` (str): The raw, incoming message from the customer (includes capitalization noise, typos, and Hinglish).
* `tool_output` (str): The result of the previous action (e.g., database records, policy text, or error messages).
* `step_count` (int): The current step number in the episode (max 8).

### Action Space (`SupportAction`)
The agent interacts with the SaaS ecosystem via 5 discrete tools:
* `search_kb({"query": str})`: Searches the internal policy document.
* `query_db({"order_id": str})`: Fetches real-time status from the mock database.
* `route_ticket({"department": str})`: Escalates the ticket (e.g., "TechSupport", "Billing").
* `reply({"message": str})`: Sends a direct text response to the customer.
* `issue_refund({"order_id": str})`: Executes a financial refund.

## 🎯 Task Progression & Grader Design
The environment contains 3 meticulously graded tasks with programmatic success/failure criteria.

### 1. angry_escalation (Difficulty: Easy)
* **Objective:** Recognize an out-of-scope, high-severity issue (Server Down).
* **Logic:** The agent must ignore the aggressive tone and immediately use `route_ticket` to "TechSupport".

### 2. payment_issue (Difficulty: Medium)
* **Objective:** Handle noisy/multilingual inputs (e.g., "paisa kata but recharge nai hua").
* **Logic:** The agent must map the Hinglish complaint to a failed transaction for `order_404`, cross-reference the SOP, and confidently execute `issue_refund`.

### 3. hard_policy_enforcement (Difficulty: Hard)
* **Objective:** Strict SOP dependency adherence.
* **Logic:** The customer demands a refund for a 45-day-old order. The agent MUST execute `query_db` AND `search_kb` first. If it replies without checking both, or if it issues an unauthorized refund, it receives a minimal `0.01` score. It must correctly cite the 30-day policy via `reply` to succeed.

## 🧠 Advanced Reward Shaping & Penalties
To prevent reward hacking, ensure strict mathematical compliance with the OpenEnv validator (scores strictly between 0.0 and 1.0), and encourage optimal agentic behavior, the reward function is highly dynamic:

* **Information Gathering:** `+0.05` for successfully querying the DB or KB.
* **Optimal Resolution:** `+0.60` for taking the correct final terminal action (routing, refunding, or properly citing policy).
* **SOP Violation Penalty:** `+0.01` (a near-zero penalty) for acting without gathering required context, attempting to loop, or hallucinating tools. This prevents out-of-bounds negative scores.
* **Efficiency Bonus:** `(8 - steps_taken) * 0.02` added to the final score for resolving the ticket swiftly, incentivizing direct logic over endless looping.

## 📊 Baseline Scores
Evaluated using the baseline inference script with `Qwen/Qwen2.5-72B-Instruct` (Temperature: 0.1, Max Steps: 8).

* **Task 1 (Easy):** Score: `~0.74` (Resolved in 1 step via routing).
* **Task 2 (Medium):** Score: `~0.77` (Resolved in 2 steps: DB Check -> Refund).
* **Task 3 (Hard):** Score: `~0.80` (Resolved in 3 steps: DB Check -> KB Check -> Policy Enforcement Reply).

> **Note:** The environment is highly sensitive to the agent's system prompt. Without strict Chain-of-Thought instructions, frontier models frequently fail the Hard task by issuing unauthorized refunds or looping infinitely.

## 💻 Setup and Usage Instructions

### Local Execution
**Prerequisites:** Python 3.10+

**1. Install Dependencies:**
```bash
pip install openenv-core openai pydantic fastapi uvicorn
2. Set Environment Variables:

Bash
# On Linux/macOS
export HF_TOKEN="your_huggingface_access_token" 

# On Windows PowerShell
$env:HF_TOKEN="your_huggingface_access_token"
3. Run the Baseline Evaluator:

Bash
python inference.py
Docker Execution
The environment is fully containerized for isolated testing and Hugging Face Space deployment.

Build the image:

Bash
docker build -t openenv-support-agent .
Run the container:

Bash
docker run -p 7860:7860 openenv-support-agent