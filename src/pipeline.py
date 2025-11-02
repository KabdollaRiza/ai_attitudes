# src/pipeline.py
import subprocess

steps = [
    ["python", "-m", "src.analyze.sentiment"],
    ["python", "-m", "src.analyze.emotions"],
    ["python", "-m", "src.analyze.topics"],
    ["python", "-m", "src.analyze.toxicity"],
]

def run_step(cmd):
    print("\n🚀 Running:", " ".join(cmd))
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print("❌ Step failed:", e)

def main():
    for step in steps:
        run_step(step)
    print("\n✅ All analysis steps completed successfully!")

if __name__ == "__main__":
    main()
