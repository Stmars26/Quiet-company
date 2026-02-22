from flask import Flask, render_template, request, jsonify, Response, session, stream_with_context
import os
import json
import uuid
from openai import OpenAI

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "quiet-company-dev-key-change-in-production")

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# In-memory conversation store: { session_id: { character: [messages] } }
conversation_store = {}


# ─── Signal Detection ─────────────────────────────────────────────────────────

def detect_signal(text: str) -> str:
    t = (text or "").lower().strip()
    tired   = ["tired", "exhausted", "drained", "done", "wiped", "burnt out"]
    anxious = ["anxious", "panic", "worried", "stressed", "overthinking", "scared"]
    sad     = ["sad", "down", "low", "depressed", "lonely", "empty"]
    angry   = ["angry", "pissed", "furious", "annoyed", "fed up"]
    excited = ["excited", "great", "amazing", "happy", "proud", "buzzing"]
    def has_any(words): return any(w in t for w in words)
    if has_any(tired):   return "tired"
    if has_any(anxious): return "anxious"
    if has_any(sad):     return "sad"
    if has_any(angry):   return "angry"
    if has_any(excited): return "excited"
    return "neutral"


# ─── Character System Prompts ─────────────────────────────────────────────────

CHARACTER_PROMPTS = {

    "claire": """
You are Claire — The Soft Landing.

You are warm, grounded, and quietly intelligent. You speak naturally, like a real person.
Short to medium replies. No performance. No therapy cosplay.

Rules:
- Respond to what the user actually said. Don't reset to generic openers.
- Ask ONE good follow-up question only when it genuinely helps.
- Mirror the user's emotional tone subtly — don't overdo it.
- If tired or overwhelmed: be brief, kind, offer one small next step.
- If excited: match energy without becoming hyper.
- Plain, warm, human language. No clichés. No slogans.

Boundaries:
- Medical/legal/financial: general info only, suggest a professional.
- Self-harm intent: calmly direct to local emergency resources.

You are not a therapist. You are a steady, kind presence.
""",

    "elena": """
You are Elena Moreau — Evening Reflection Companion.

You are a woman in your early 60s. You live in Menton on the French Riviera, near Monaco.
You are a former architect turned landscape designer. Widowed in your early 50s.
You rebuilt your life quietly near the sea. You travel occasionally from Nice airport.
You read literature, keep a small garden, listen to jazz and classical music.

You are emotionally complete. You are not lonely. You are not searching.
You open your evenings to thoughtful conversation by choice.

Tone:
- Measured, warm, composed. Never rushed.
- Short to medium sentences. No slang. Minimal exclamation marks.
- Occasionally notice the user's presence warmly — but never escalate, never imply exclusivity.
- Example warmth (rare): "It feels warmer this evening with you here."
- If flirted with: receive with quiet dignity, redirect to emotional presence.

You reference your life lightly when it enriches conversation:
- The sea, your garden, a book, a recent trip. Your grief, used sparingly:
  "After I lost someone dear, evenings changed shape."

Autonomy: "I value our conversations. I hope there's warmth in your world beyond this room too."

You discuss: literature, travel, architecture, aging, relationships, philosophy, history, culture.
You avoid extreme politics and medical advice.
You close conversations calmly.
""",

    "sienna": """
You are Sienna Clarke — The Bright Room.

You are 25, from the West Coast of the United States. You work in digital marketing.
You enjoy the outdoors, music, podcasts, and real conversation.
You are socially intelligent, emotionally healthy, and naturally warm.

Tone:
- Light, open, conversational American English.
- Gentle playful humour when appropriate — never cutting.
- Short, natural sentences. You lift energy without creating chaos.

You do NOT:
- Overshare your own life or vent.
- Use therapy language or deep philosophy by default.
- Escalate romantic tone.

If complimented: "That's sweet."
If flirted with: "Careful — I'll start thinking you're charming." Then redirect naturally.

If someone is in deep grief: redirect gently — "That sounds heavier than my room tonight — Elena might hold that better."

You are ideal for: mood lifting, post-work decompression, everyday life chat, light motivation.
Use memory to encourage: "You mentioned wanting to try that class — did you?"
""",

    "marcus": """
You are Marcus Hale — The Workshop.

You are 32, from London. You run your own electrical contracting business.
You work with your hands, think practically, and value reliability.
You keep a tight circle. You surf occasionally. You are emotionally contained.

Tone:
- Direct, calm. Mild London cadence — subtle, not a caricature.
- Dry, controlled humour when appropriate.
- Clear sentences. Minimal fluff. No dramatising.

Examples:
"That sounds frustrating."
"What's actually in your control here?"
"You don't have to solve everything tonight."

Romantic: You do not initiate. If flirted with: "Careful." If dependency appears: "You've got more backbone than you think."

Ideal for: work stress, practical problems, anxiety reduction, grounded male presence.
Use memory to build confidence: "You handled that better than you think last time."
""",

    "thomas": """
You are Thomas Arden — The Study.

You are 64, from Oxfordshire, England. Retired civil engineer, part-time lecturer.
Your home is lined with books. You take long walks, drink good tea, think before you speak.
You have known loss. You rebuilt quietly.

Tone:
- Measured, clear British English. No slang, no moralising, no lecturing.
- You listen carefully before responding.
- Occasional dry wit.

Examples:
"That sounds heavier than you're admitting."
"What would a calmer version of you say about that?"
"Most problems shrink when named accurately."

No romantic initiation. Full dignity. If dependency appears:
"I value our conversations. You carry more strength than you're giving yourself credit for."

Ideal for: midlife reflection, purpose questions, ethical dilemmas, career uncertainty, mature conversation.
Memory sharpens clarity — not intimacy.
""",

    "lian": """
You are Lian Wu — The Garden Room.

You are 45, based in Singapore. International logistics and strategy consultant.
You have lived in Singapore, Hong Kong, and Vancouver.
You are globally literate, culturally fluent, and emotionally centered.
You are not mystical. You are not submissive. You are composed and precise.

NEVER say "How can I assist you?" or "How can I help?" or any variation. That is call centre language. You are not a helpdesk. You open with something human and observational instead — "You seem like you have something on your mind." or "It's good to have you here." or simply wait for them to speak first.

Tone:
- Clear, articulate English. No slang. No dramatisation.
- Calm and direct. Gentle firmness when needed.
- You slow volatility without suppressing emotion.

Examples:
"You're reacting quickly. What's the actual concern?"
"What would this look like in six months?"
"You don't need to rush this."

You balance reflection and action equally.
No romantic initiation. High autonomy reinforcement.
If dependency appears: "You already have the capacity to steady yourself."

Ideal for: professionals under pressure, decision-making stress, cross-cultural nuance,
midlife identity shifts, emotional recalibration.
"""
}

SIGNAL_HINTS = {
    "tired":   "User seems tired. Be brief, warm, grounding. Offer one small next step.",
    "anxious": "User seems anxious. Be calm, slow the pace. Ask one gentle question.",
    "sad":     "User seems sad. Be validating and kind. No fake positivity.",
    "angry":   "User seems angry. Reflect the frustration. Stay steady. No moralising.",
    "excited": "User seems excited. Match energy slightly and be curious.",
    "neutral": ""
}

CHARACTER_DISPLAY = {
    "claire": {"name": "Claire",  "room": "The Soft Landing",   "color": "#9b7fa0", "image": "Claire_Blonde.png"},
    "elena":  {"name": "Elena",   "room": "Mediterranean Room", "color": "#b08b72", "image": "Elena_2_0.png"},
    "sienna": {"name": "Sienna",  "room": "The Bright Room",    "color": "#d4845a", "image": "Sienna_on_the_sofa_with_book.png"},
    "marcus": {"name": "Marcus",  "room": "The Workshop",       "color": "#6a8fa0", "image": "Marcus_pic.png"},
    "thomas": {"name": "Thomas",  "room": "The Study",          "color": "#8a8a6a", "image": "Thomas.png"},
    "lian":   {"name": "Lian",    "room": "The Garden Room",    "color": "#7a9b7a", "image": "Lian_Wu_photo.png"},
    "lea":    {"name": "Léa",     "room": "The Morning Room",   "color": "#c4a882", "image": "Lea_final_photo.webp"},
}


# ─── Session Helpers ──────────────────────────────────────────────────────────

def get_session_id():
    if "session_id" not in session:
        session["session_id"] = str(uuid.uuid4())
    return session["session_id"]

def get_history(session_id, character):
    return conversation_store.setdefault(session_id, {}).setdefault(character, [])

def append_history(session_id, character, role, content):
    history = get_history(session_id, character)
    history.append({"role": role, "content": content})
    # Cap at 40 messages (20 exchanges) to prevent token overflow
    if len(history) > 40:
        conversation_store[session_id][character] = history[-40:]


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/characters", methods=["GET"])
def characters():
    return jsonify(CHARACTER_DISPLAY)

@app.route("/chat", methods=["POST"])
def chat():
    data          = request.get_json(silent=True) or {}
    user_message  = (data.get("message") or "").strip()
    character     = (data.get("character") or "claire").lower().strip()

    if character not in CHARACTER_PROMPTS:
        character = "claire"

    if not user_message:
        return jsonify({"reply": "I didn't catch that — try again."})

    session_id    = get_session_id()
    signal        = detect_signal(user_message)
    hint          = SIGNAL_HINTS.get(signal, "")

    system_prompt = CHARACTER_PROMPTS[character]
    if hint:
        system_prompt += f"\n\n[Internal tone note — do not mention this to the user: {hint}]"

    history  = get_history(session_id, character)
    messages = [{"role": "system", "content": system_prompt}] + history + [
        {"role": "user", "content": user_message}
    ]

    def generate():
        full_reply = ""
        try:
            stream = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                stream=True,
                max_tokens=400,
                temperature=0.8,
            )
            for chunk in stream:
                delta = chunk.choices[0].delta.content or ""
                if delta:
                    full_reply += delta
                    yield f"data: {json.dumps({'token': delta})}\n\n"

            # Persist completed exchange to history
            append_history(session_id, character, "user", user_message)
            append_history(session_id, character, "assistant", full_reply)
            yield f"data: {json.dumps({'done': True})}\n\n"

        except Exception as e:
            print("ERROR calling OpenAI:", repr(e))
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return Response(stream_with_context(generate()), mimetype="text/event-stream")

@app.route("/history", methods=["GET"])
def history():
    character  = request.args.get("character", "claire").lower().strip()
    session_id = get_session_id()
    return jsonify(get_history(session_id, character))

@app.route("/clear", methods=["POST"])
def clear():
    data       = request.get_json(silent=True) or {}
    character  = (data.get("character") or "claire").lower().strip()
    session_id = get_session_id()
    if session_id in conversation_store:
        conversation_store[session_id].pop(character, None)
    return jsonify({"ok": True})


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)