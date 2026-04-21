import json
from config import client, MODEL
from tools import scripture_search, reflect, pray, respond, finish


# =========================
# THINK
# =========================
def think(goal, memory):
    prompt = f"""
You are a Scripture Companion AI.

Goal:
{goal}

Memory:
{memory}

You MUST follow this reasoning process:
1. Find a relevant scripture
2. Reflect on it
3. Optionally pray
4. Then finish with a clear final answer

Available actions:
- scripture
- reflect
- pray
- respond
- finish

STRICT RULES:
- NEVER skip directly to finish
- ALWAYS produce meaningful output
- DO NOT say "already generated"
- Output ONLY JSON

Format:
{{
  "action": "...",
  "input": "..."
}}
"""

    response = client.models.generate_content(
        model=MODEL,
        contents=prompt
    )

    text = response.text.strip()

    try:
        json_start = text.find("{")
        json_end = text.rfind("}") + 1
        clean_json = text[json_start:json_end]
        return json.loads(clean_json)
    except:
        return {"action": "respond", "input": text}


# =========================
# ACT
# =========================
def act(action, input_text):
    if action == "scripture":
        return scripture_search(input_text)
    elif action == "reflect":
        return reflect(input_text)
    elif action == "pray":
        return pray(input_text)
    elif action == "respond":
        return respond(input_text)
    elif action == "finish":
        return finish(input_text)
    else:
        return "Unknown action"


# =========================
# AGENT LOOP
# =========================
def run_agent(goal, max_steps=5):
    memory = []

    for step in range(max_steps):
        decision = think(goal, memory)

        action = decision.get("action")
        input_text = decision.get("input")

        print(f"\n[Step {step+1}] {action} → {input_text}")

        result = act(action, input_text)

        print(f"Result: {result}")

        memory.append({
            "step": step + 1,
            "action": action,
            "input": input_text,
            "result": result
        })

        if action == "finish":
            print("\n✅ FINAL ANSWER:\n", result)
            break