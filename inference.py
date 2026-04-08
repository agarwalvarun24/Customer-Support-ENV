import os
from openai import OpenAI
from my_env.env import SupportEnv
from my_env.models import Action

API_KEY = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"

client = None
if API_KEY:
    try:
        client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    except:
        client = None


def log_start(task, env, model):
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step, action, reward, done):
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error=null",
        flush=True
    )

def log_end(success, steps, score, rewards):
    rewards_str = ",".join([f"{r:.2f}" for r in rewards])
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}",
        flush=True
    )


def rule_based_action(ticket, step):
    ticket = ticket.lower()

    if step == 1:
        if "payment" in ticket or "charged" in ticket:
            return Action(category="billing", priority="", response="")
        elif "crash" in ticket or "slow" in ticket:
            return Action(category="technical", priority="", response="")
        else:
            return Action(category="general", priority="", response="")

    elif step == 2:
        if "failed" in ticket or "crash" in ticket:
            return Action(category="", priority="high", response="")
        elif "slow" in ticket:
            return Action(category="", priority="medium", response="")
        else:
            return Action(category="", priority="low", response="")

    else:
        return Action(
            category="",
            priority="",
            response="We understand your issue and will resolve it soon."
        )


def get_action(ticket, step):
    if client is None:
        return rule_based_action(ticket, step)

    try:
        prompt = f"Ticket: {ticket}. Step {step}. Give category|priority|response"
        res = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=100
        )

        text = res.choices[0].message.content.strip()
        parts = text.split("|")

        if len(parts) != 3:
            raise ValueError

        return Action(
            category=parts[0].strip().lower(),
            priority=parts[1].strip().lower(),
            response=parts[2].strip()
        )

    except:
        return rule_based_action(ticket, step)


def run():
    env = SupportEnv()
    rewards = []

    log_start("customer-support", "support-env", MODEL_NAME)

    obs = env.reset(task_index=0)

    for step in range(1, 4):
        action = get_action(obs.ticket, step)
        result = env.step(action)

        rewards.append(result.reward)

        action_str = f"{action.category},{action.priority}"
        log_step(step, action_str, result.reward, result.done)

        if result.done:
            break

    score = min(max(sum(rewards), 0.0), 1.0)
    success = score >= 0.5

    log_end(success, step, score, rewards)

    env.close()

from fastapi import FastAPI
from my_env.env import SupportEnv

app = FastAPI()

env = SupportEnv()

@app.get("/")
def home():
    return {"message": "API is running"}

@app.post("/reset")
def reset():
    obs = env.reset(task_index=0)
    return {"ticket": obs.ticket}

@app.post("/step")
def step(action: dict):
    result = env.step(action)
    return {
        "reward": result.reward,
        "done": result.done
    }
