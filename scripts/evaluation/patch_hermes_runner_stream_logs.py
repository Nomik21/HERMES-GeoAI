from pathlib import Path

path = Path("scripts/evaluation/run_hermes_smoke_evaluation.py")
text = path.read_text(encoding="utf-8")

if "import threading" not in text:
    text = text.replace("import subprocess", "import subprocess\nimport threading")

helper = r'''
def run_subprocess_streaming(cmd, cwd, timeout, env):
    """
    Run subprocess while streaming stdout/stderr live to terminal
    and also returning captured stdout/stderr for log files.
    """
    stdout_chunks = []
    stderr_chunks = []

    proc = subprocess.Popen(
        cmd,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
        bufsize=1,
    )

    def pump(stream, chunks, writer):
        try:
            for line in iter(stream.readline, ""):
                chunks.append(line)
                writer.write(line)
                writer.flush()
        finally:
            try:
                stream.close()
            except Exception:
                pass

    stdout_thread = threading.Thread(
        target=pump,
        args=(proc.stdout, stdout_chunks, sys.stdout),
        daemon=True,
    )
    stderr_thread = threading.Thread(
        target=pump,
        args=(proc.stderr, stderr_chunks, sys.stderr),
        daemon=True,
    )

    stdout_thread.start()
    stderr_thread.start()

    try:
        returncode = proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        stdout_thread.join(timeout=5)
        stderr_thread.join(timeout=5)

        raise subprocess.TimeoutExpired(
            cmd=cmd,
            timeout=timeout,
            output="".join(stdout_chunks),
            stderr="".join(stderr_chunks),
        )

    stdout_thread.join(timeout=5)
    stderr_thread.join(timeout=5)

    return subprocess.CompletedProcess(
        args=cmd,
        returncode=returncode,
        stdout="".join(stdout_chunks),
        stderr="".join(stderr_chunks),
    )

'''

if "def run_subprocess_streaming(" not in text:
    marker = "def main():"
    text = text.replace(marker, helper + "\n\n" + marker)

old = '''            proc = subprocess.run(
                cmd,
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=args.timeout,
                env=env,
            )'''

new = '''            proc = run_subprocess_streaming(
                cmd=cmd,
                cwd=PROJECT_ROOT,
                timeout=args.timeout,
                env=env,
            )'''

if old not in text:
    raise SystemExit("Expected subprocess.run block not found. No patch applied.")

text = text.replace(old, new)

path.write_text(text, encoding="utf-8")
print(f"Patched live CrewAI log streaming in: {path}")
