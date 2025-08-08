
**MemoraX** is a long-text summarisation and quiz-generation app built with **Streamlit** and **Hugging Face Transformers**.  

It can handle PDFs, Word docs, or pasted text, chunk them safely for large language models, and output concise summaries with optional auto-generated multiple-choice quizzes.


![App Screenshot](<img width="1477" height="1098" alt="image" src="https://github.com/user-attachments/assets/f699d202-f812-436b-8634-5219b99993ba" />
)  

Features

- ** Multi-format input** — Upload PDF (`.pdf`) or Word (`.docx`) files, or paste text directly.
- ** Model selection** — Choose between:
  - **BART** (`facebook/bart-large-cnn`) — fast, ~1k token context.
  - **LED** (`allenai/led-base-16384`) — long-form, ~16k token context.
- ** Safe chunking** — Automatically splits text by token count to avoid model truncation.
- ** Summarisation** — Produces concise summaries within user-set length bounds.
- ** Quiz generation** — Creates multiple-choice questions from the summary using keyword extraction.
- ** Progress feedback** — Chunk progress, ETA, and debug view of sub-summaries.
