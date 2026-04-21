import google.generativeai as genai
import json

# =========================
# CONFIG
# =========================
genai.configure(api_key="AIzaSyCbnPOJhEIFCXKJkl0afGm9-EhX5JbryJ4")

model = genai.GenerativeModel("gemini-2.5-flash")

# =========================
# TOOLS
# =========================
def calculator(expression):
    try:
        return str(eval(expression))
    except Exception as e:
        return f"Error: {str(e)}"


def fake_search(query):
    # mock search tool (replace later with real API)
    return f"Search results for '{query}': Example data about {query}"


# =========================
# THINK (DECISION MAKING)
# =========================
def think(goal, memory):
    prompt = f"""
You are an AI agent.

Goal: {goal}

Previous steps:
{memory}

Decide the next action.

Available tools:
1. calculator → for math expressions
2. search → for general knowledge
3. finish → if task is complete

Return ONLY JSON like this:
{{
  "action": "calculator" | "search" | "finish",
  "input": "your input here"
}}
"""

    response = model.generate_content(prompt)
    text = response.text.strip()

    # Try to extract JSON safely
    try:
        json_start = text.find("{")
        json_end = text.rfind("}") + 1
        clean_json = text[json_start:json_end]
        return json.loads(clean_json)
    except:
        return {"action": "finish", "input": text}


# =========================
# ACT (EXECUTE TOOLS)
# =========================
def act(action, input_text):
    if action == "calculator":
        return calculator(input_text)

    elif action == "search":
        return fake_search(input_text)

    elif action == "finish":
        return input_text

    else:
        return "Unknown action"


# =========================
# AGENT LOOP
# =========================
def agent(goal, max_steps=5):
    memory = []

    print(f"\n🎯 Goal: {goal}\n")

    for step in range(max_steps):
        print(f"--- Step {step+1} ---")

        decision = think(goal, memory)
        action = decision.get("action")
        input_text = decision.get("input")

        print("Action:", action)
        print("Input:", input_text)

        result = act(action, input_text)
        print("Result:", result)

        memory.append({
            "action": action,
            "input": input_text,
            "result": result
        })

        if action == "finish":
            print("\n✅ Final Answer:", result)
            return result

    print("\n⚠️ Reached max steps. Last result:", result)
    return result


# =========================
# RUN
# =========================
if __name__ == "__main__":
    agent("Give me a bible Verse about love and explain it")
