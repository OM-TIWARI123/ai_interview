import os, asyncio, keyboard, speech_recognition as sr, threading, time
from pathlib import Path
from dotenv import load_dotenv
from graph import create_graph, resume_processor, ask_question, eval_answer

load_dotenv()

rec = sr.Recognizer()
mic = sr.Microphone()

def listen_once():
    """Blocking helper: push-to-talk via ENTER."""
    print("\n🔴 Recording … press ENTER to start, ENTER again to stop.")
    keyboard.wait('enter')          # start recording
    print("🎙️  Recording … press ENTER to stop.")
    with mic as source:
        rec.adjust_for_ambient_noise(source, duration=1)
        try:
            audio = rec.listen(source, timeout=None)
        except sr.WaitTimeoutError:
            return None
    keyboard.wait('enter')  # wait for second ENTER to stop
    print("⏹️  Stopped.")
    try:
        text = rec.recognize_google(audio)
        return text
    except sr.UnknownValueError:
        return ""
    except Exception as e:
        print(f"Error: {e}")
        return ""

async def main():
    resume_path = input("📄 Resume path (PDF/DOCX/TXT): ").strip()
    roles = ["SDE", "Data Scientist", "Product Manager"]
    for i, r in enumerate(roles, 1):
        print(f"{i}. {r}")
    role = roles[int(input("🎯 Select role (1-3): ").strip()) - 1]

    # ---- 1. Resume ingest (blocking) ----
    state = {"resume_path": resume_path, "role": role}
    state.update(resume_processor(state))

    # ---- 2. Introduction ----
    intro = state["intro"]
    print("\n🎤", intro)
    await state["speak"](intro)

    # ---- 3. User intro ----
    user_intro = listen_once()
    print(f"👤 You said: {user_intro or 'No answer'}")
    state["last_q"] = "Please introduce yourself"
    result = eval_answer(state, user_intro or "No answer")
    print(f"🤖 Feedback: {result}")
    state["history"].append(result)

    # ---- 4. Questions ----
    for q in state["questions"]:
        print(f"\n🎤 {q}")
        await state["speak"](q)
        state["last_q"] = q
        answer = listen_once()
        print(f"👤 You said: {answer or 'No answer'}")
        result = eval_answer(state, answer or "No answer")
        print(f"🤖 Feedback: {result}")
        state["history"].append(result)

    print("\n✅ Interview complete")
    for h in state["history"]:
        print(h)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 bye")