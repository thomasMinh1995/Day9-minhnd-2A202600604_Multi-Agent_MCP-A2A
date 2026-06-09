"""CLI entrypoint for the Supervisor-Workers legal system."""

from __future__ import annotations

import argparse
import json

from .supervisor import LegalSupervisorSystem


def main() -> None:
    parser = argparse.ArgumentParser(description="Legal Supervisor-Workers CLI")
    parser.add_argument("query", nargs="?", default="Hình phạt tàng trữ trái phép chất ma tuý là gì?")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--json", action="store_true", help="Print full JSON response")
    args = parser.parse_args()

    result = LegalSupervisorSystem().ask(args.query, top_k=args.top_k)
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    print("\nLEGAL SUPERVISOR-WORKERS ANSWER")
    print("=" * 72)
    print(result.get("answer", ""))
    print("\nSupervisor:")
    print(json.dumps(result.get("supervisor", {}), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
