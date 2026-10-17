# Invoice Approval Agent — Single Python File

All application code is in **invoiceprocessagent.py**: shared state, PDF reading,
Gemini calls, validation, email delivery, routing, graph construction and terminal
approval. Every graph node and edge is declared individually with `add_node()`,
`add_edge()` and `add_conditional_edges()`; graph construction contains no loops.

## Run

The existing `.env` is preserved. Fill in your credentials there:

```dotenv
GEMINI_API_KEY=
SMTP_EMAIL=
SMTP_APP_PASSWORD=
ACCOUNT_TEAM_EMAIL=yadavbeauty74@gmail.com
```

Place `vendor1.pdf` in `incoming_invoice_folder`, then run:

```bash
source .venv/bin/activate
python invoiceprocessagent.py
```

To install on a fresh machine (Python 3.13 recommended):

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows, activate with `.venv\Scripts\activate`.
Select the virtual environment as the VS Code Python interpreter.

## Workflow

1. PyPDF reads the text-based PDF. Gemini extracts the requested invoice fields;
   missing or ambiguous values remain null.
2. Python and Gemini validate the invoice. Invalid invoices trigger a customized
   vendor email when a vendor email address is available.
3. Valid INR invoices up to and including ₹25,000 are emailed to accounts with
   the original PDF attached.
4. Larger invoices pause with LangGraph `interrupt()`. Enter **Yes** or **No**;
   the terminal runner resumes with `Command(resume=...)`.
5. Approved invoices are emailed to accounts. Rejected invoices trigger a vendor
   notification.
6. After a valid invoice's accounts email is confirmed by SMTP, its PDF is moved
   to **output_invoice_folder**. Invalid, rejected, failed and uncertain-delivery
   invoices remain in **incoming_invoice_folder**.

The output normally uses `vendor1.pdf`. If that name already belongs to a different
invoice, a content hash is added to the new filename to preserve both files.
The same PDF already in output is not duplicated. If a move fails after email
success, rerun with the incoming PDF still present: the saved workflow retries
archiving without resending. The final workflow state retains the original input
path; the terminal separately prints the saved output location.

## Persistence and email safety

`workflow.sqlite` stores checkpoints. The PDF content hash is the thread ID;
rerun with the same input PDF to resume after Ctrl+C or terminal closure. Completed
workflows do not resend emails. Keep the PDF unchanged during execution.

`email_delivery.sqlite` reserves a notification before sending. `Sent` means SMTP
accepted the message, not payment or guaranteed inbox delivery. `Failed` means
email preparation failed. `DeliveryUnknown` means delivery could not be confirmed.
SMTP and SQLite cannot commit atomically, so an uncertain send is never retried
automatically. Check the sender's mailbox/server logs before manual recovery.
Email failures end the workflow; simply rerunning does not retry a completed
failed send. API errors before email can be retried by rerunning.

Do not delete the SQLite files for live invoices: that removes checkpoint and
email duplicate protection. The databases and terminal output contain invoice
details. Credentials are read from `.env`, never logged. Invoice text is sent to
Gemini. Shell environment variables take precedence over `.env`.

## Configuration and limitations

- Uses exactly `gemini-3.1-flash-lite-preview`; no fallback model.
- Default SMTP: Gmail, port 587, STARTTLS and an app password. Optional
  `SMTP_HOST` / `SMTP_PORT` support other STARTTLS providers.
- Only INR, text-based, unlocked PDFs are supported. OCR is not included.
- No invented recipient addresses. A missing vendor email prevents notification.
- Application code uses no LangChain APIs. Official LangGraph installs
  `langchain-core` transitively as a required dependency.
- AI consistency validation is probabilistic; it does not replace accounting review.

The single-file version was verified with temporary offline tests covering
mocked Gemini/SMTP calls, thresholds, approval/rejection, restart recovery,
terminal input, attachments, duplicate prevention, error handling, output moves,
failed-move retry and filename collisions. No real emails were sent during tests.
Only one project Python source file is retained for teaching.
