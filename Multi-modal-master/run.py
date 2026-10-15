"""TraceLearn — development entry point.

    python run.py

Serves the app on http://127.0.0.1:5000.
"""
from __future__ import annotations

import os

from tracelearn import create_app
from tracelearn.config import settings

app = create_app()

BANNER = r"""
  ______                   __
 /_  __/________ _________/ /__   __  ___________ ____
  / / / ___/ __ `/ ___/ _ \/ / _ \ / / / / ___/ _ `/ _ \
 / / / /  / /_/ / /__/  __/ /  __// /_/ / /  /  __/  __/
/_/ /_/   \__,_/\___/\___/_/\___/ \__, /_/   \___/\___/
                                 /____/
"""


def main() -> None:
    port = int(os.getenv("PORT", "5000"))
    state = settings.summary()
    print(BANNER)
    print("  Multimodal AI study companion — grounded tutor, verified quizzes, BKT learner model\n")
    print(f"  mode ......... {'DEMO (no model key)' if state['demo_mode'] else 'LIVE'}")
    print(f"  database ..... {state['db']}")
    print(f"  storage ...... {state['storage']}")
    print(f"  llm .......... {state['llm']}")
    print(f"  embeddings ... {state['embeddings']}")
    print(f"\n  →  http://127.0.0.1:{port}\n")
    app.run(host="127.0.0.1", port=port, debug=True)


if __name__ == "__main__":
    main()
