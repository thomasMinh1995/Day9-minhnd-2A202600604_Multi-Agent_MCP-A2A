# Legal Supervisor-Workers Lab Assignment

This folder contains the Supervisor-Workers version of the legal multi-agent system.

## Pattern

```text
User
  -> legal_supervisor
      -> evidence_worker
      -> legal_analysis_worker
      -> drafting_worker
      -> compliance_worker
  -> final cited answer
```

The supervisor owns planning, worker selection, message routing, trace collection, and final response assembly. Workers only perform their assigned specialty.

## Run

```bash
.venv/bin/python -m lab_assignment.cli "Hình phạt tàng trữ trái phép chất ma tuý là gì?"
.venv/bin/python -m lab_assignment.cli "Điều 249 quy định gì?" --json
streamlit run lab_assignment/app.py
```

## Main Files

- `protocol.py`: message schema and in-memory worker bus.
- `workers.py`: evidence, legal analysis, drafting, and compliance workers.
- `supervisor.py`: Supervisor-Workers orchestration.
- `cli.py`: command-line interface.
- `app.py`: Streamlit interface.
