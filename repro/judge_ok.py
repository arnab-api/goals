"""Exit 0 when the judge answers, 1 when it does not: one 1-token call to the judge model on the key in env.yml.
The lane runner (repro/eval_lib.sh) polls it after a judge failure and reruns the command once it passes; on
2026-11-05 the key ran out of API credit and every lane died at once. Run from the repository root."""
import sys

import anthropic
import yaml

key = yaml.safe_load(open("env.yml"))["CLAUDE_KEY"]
try:
    anthropic.Anthropic(api_key=key, max_retries=0).messages.create(
        model="claude-sonnet-4-6", max_tokens=1, messages=[{"role": "user", "content": "hi"}])
except Exception as e:  # any API error counts as down; the lane polls again later
    print(f"judge down: {type(e).__name__}: {str(e)[:200]}", file=sys.stderr)
    sys.exit(1)
