import google.generativeai as genai
import json

# =========================
# CONFIG
# =========================
genai.configure(api_key="AIzaSyCbnPOJhEIFCXKJkl0afGm9-EhX5JbryJ4")

model = genai.GenerativeModel("gemini-2.5-flash")
def bible_search(query):
    return f"search results for '{query}':Example data about {query}"
def think(goal,memory):
    prompt=f"""
    You are an AI agent.
    Goal:{goal}
    Previous steps:
   {memory}
You are a Scripture Companion AI.

Your role:
- Help users with Bible-based guidance
- Suggest relevant scriptures
- Provide reflections and prayers
- Speak with wisdom, calmness, and clarity

User Goal:
{goal}

Conversation Memory:
{memory}

Decide the next action.

Available actions:
1. scripture → find relevant Bible verses
2. reflect → explain or reflect on a passage
3. pray → generate a prayer
4. respond → normal conversational response
5. finish → if task is complete

Rules:
- If the user expresses emotion → prioritize scripture + comfort
- If they ask about a passage → use reflect
- If they ask for prayer → use pray
- Stay spiritually grounded (Bible-centered)
- Do NOT use calculator unless explicitly needed

Return ONLY JSON in this format:
{{
  "action": "scripture | reflect | pray | respond | finish",
  "input": "what you will use or generate"
}}
"""
    response=model.generate_content(prompt)
    text=response.text.strip()
    try:
        json_start=text.find("{")
        json_end=text.rfind("}")+1
        clean_json=text[json_start:json_end]
        return json.loads(clean_json)
    except Exception as e:
        return {"action":"respond","input":f"Error parsing JSON: {str(e)}. Original response: {text}"}
    def act(scripture_search,reflect,pray,respond,finish):
        if action=="scripture":
            return scripture_search(input)
        elif action=="reflect":
            return reflect(input)
        elif action=="pray":
            return pray(input)
        elif action=="respond":
            return respond(input)
        elif action=="finish":
            return finish(input)
        else:
            return "Unknown action"
# =========================
# AGENT LOOP
# =========================
    def agent(goal,max_steps=5):
        memory=[]
        for step in range(max_steps):
            decision=think(goal,memory)
            action=decision.get("action")
            input_text=decision.get("input")
            result=act(bible_search,reflect,pray,respond,finish)
            memory.append(f"Step {step+1}: Action: {action}, Input: {input_text}, Result: {result}")
            if action=="finish":
                print("Final Result:", result)
                break


        
# =========================
# RUN
# =========================
if __name__=="__main__":
    agent("Give me a bible verse about love and explain love")

