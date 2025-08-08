import streamlit as st
from transformers import pipeline, AutoTokenizer
from PyPDF2 import PdfReader
import docx
import re, random, time
from typing import List, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer

# ------------------------
# File reading functions
# ------------------------
def read_pdf(file):
    pdf = PdfReader(file)
    text = ""
    for page in pdf.pages:
        text += page.extract_text() + "\n"
    return text

def read_docx(file):
    doc = docx.Document(file)
    return "\n".join([para.text for para in doc.paragraphs if para.text.strip()])

# ------------------------
# Text cleaning
# ------------------------
def clean_text(text: str) -> str:
    text = re.sub(r"\bPage\s+\d+(\s+of\s+\d+)?\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^\s*\d+\s*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"Chapter\s+\d+", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    text = re.sub(r"(\w+)-\n(\w+)", r"\1\2", text)
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)
    return text.strip()

# ------------------------
# Quiz helpers
# ------------------------
def split_sentences(text: str) -> List[str]:
    return re.split(r'(?<=[.!?])\s+', text.strip())

def top_keywords(text: str, k: int = 12) -> List[str]:
    vec = TfidfVectorizer(stop_words="english", ngram_range=(1,2), min_df=1)
    X = vec.fit_transform([text])
    scores = X.toarray()[0]
    feats = vec.get_feature_names_out()
    pairs = sorted(zip(feats, scores), key=lambda x: x[1], reverse=True)
    cleaned = [w for w,_ in pairs if len(w) > 2 and not re.fullmatch(r'\d+(\.\d+)?', w)]
    return cleaned[:k]

def find_sentence_with(term: str, sentences: List[str]) -> str | None:
    pat = re.compile(rf'\b{re.escape(term)}\b', re.IGNORECASE)
    for s in sentences:
        if pat.search(s):
            return s
    return None

def make_mcq(term: str, sentence: str, distract_pool: List[str], n_opts: int = 4) -> Tuple[str, List[str], str]:
    blanked = re.sub(rf'(?i)\b{re.escape(term)}\b', "_____", sentence)
    correct = term
    pool = [w for w in distract_pool if w.lower() != term.lower()]
    random.shuffle(pool)
    distractors = []
    for w in pool:
        if len(distractors) >= (n_opts - 1):
            break
        if w.lower() != correct.lower() and w not in distractors:
            distractors.append(w)
    fallback = ["context", "process", "feature", "model", "system", "method"]
    for f in fallback:
        if len(distractors) >= (n_opts - 1): break
        if f.lower() != correct.lower() and f not in distractors:
            distractors.append(f)
    options = distractors + [correct]
    random.shuffle(options)
    q = f"In the sentence: “{blanked}” — what word/phrase best fills the blank?"
    return q, options, correct

# ------------------------
# Streamlit UI setup
# ------------------------
st.title("MemoraX")
st.caption("Long-text summariser with safe chunking, file upload, and quiz generation")

MODEL_CHOICES = {
    "BART (fast, ~1k token context)": "facebook/bart-large-cnn",
    "LED (long, ~16k token context)": "allenai/led-base-16384",
}
choice = st.selectbox("Model", list(MODEL_CHOICES.keys()), index=0)
model_name = MODEL_CHOICES[choice]

@st.cache_resource(show_spinner=False)
def load_pipe_and_tok(name: str):
    pipe = pipeline("summarization", model=name)
    tok = AutoTokenizer.from_pretrained(name)
    return pipe, tok

summariser, tok = load_pipe_and_tok(model_name)

uploaded_file = st.file_uploader("Upload a PDF or Word document", type=["pdf", "docx"])
manual_text = st.text_area("Or paste text manually:", height=220)

text_input = ""
if uploaded_file:
    if uploaded_file.name.lower().endswith(".pdf"):
        text_input = read_pdf(uploaded_file)
    elif uploaded_file.name.lower().endswith(".docx"):
        text_input = read_docx(uploaded_file)
elif manual_text.strip():
    text_input = manual_text

if text_input.strip():
    text_input = clean_text(text_input)

col1, col2 = st.columns(2)
with col1:
    out_max = st.slider("Output max length", 80, 280, 160, 10)
with col2:
    out_min = st.slider("Output min length", 20, 120, 60, 10)

# ------------------------
# Summarisation helpers
# ------------------------
def model_context_limit(tok):
    m = tok.model_max_length
    if m is None or m > 200_000_000_000:
        return 16_384 if "led" in model_name.lower() else 1024
    return int(m)

def safe_chunk_size(tok):
    limit = model_context_limit(tok)
    return max(400, min(900, limit - 80))

def count_tokens(text):
    return len(tok.encode(text, add_special_tokens=False))

def chunk_by_tokens(text, max_tokens, overlap=80):
    ids = tok.encode(text, add_special_tokens=False)
    n = len(ids)
    chunks, i = [], 0
    while i < n:
        j = min(i + max_tokens, n)
        chunk_ids = ids[i:j]
        chunks.append(tok.decode(chunk_ids))
        if j == n:
            break
        i = max(0, j - overlap)
    return chunks

def summarise_once(c):
    return summariser(c, max_length=out_max, min_length=out_min, do_sample=False, truncation=True)[0]["summary_text"]

def summarise_hierarchical(text):
    size = safe_chunk_size(tok)
    total_tokens = count_tokens(text)
    chunks = chunk_by_tokens(text, max_tokens=size, overlap=80)

    st.info(f"**Model:** {choice} • **Tokens:** ~{total_tokens:,} • **Chunks:** {len(chunks)} (≈{size} tokens each)")
    progress = st.progress(0)
    status = st.empty()

    first_pass, times = [], []
    for i, c in enumerate(chunks, 1):
        t0 = time.time()
        try:
            first_pass.append(summarise_once(c))
        except Exception:
            for smaller in (size - 200, size - 350):
                smaller = max(400, smaller)
                for sub in chunk_by_tokens(c, max_tokens=smaller, overlap=60):
                    first_pass.append(summarise_once(sub))
        dt = time.time() - t0
        times.append(dt)
        avg = sum(times[-3:]) / min(len(times), 3)
        remaining = max(0, len(chunks) - i) * avg
        progress.progress(i / len(chunks))
        status.write(f"Chunk {i}/{len(chunks)} done • {dt:.1f}s • ETA {remaining:.1f}s")

    status.write("Combining summaries…")
    joined = " ".join(first_pass)
    final = summariser(joined, max_length=out_max, min_length=out_min, do_sample=False, truncation=True)[0]["summary_text"]
    progress.progress(1.0)
    status.write("All done ✅")
    return final, first_pass

def summarise_long(text):
    if "led" in model_name.lower():
        total_tokens = count_tokens(text)
        st.info(f"**Model:** {choice} • **Tokens:** ~{total_tokens:,} • Single-pass")
        return summariser(text, max_length=out_max, min_length=out_min, truncation=True, do_sample=False)[0]["summary_text"], None
    return summarise_hierarchical(text)

# ------------------------
# Run summarisation
# ------------------------
if st.button("Summarise"):
    if not text_input.strip():
        st.warning("Please paste or upload text first.")
    else:
        with st.spinner("Summarising…"):
            final_summary, partials = summarise_long(text_input)
        st.session_state.final_summary = final_summary
        st.session_state.partials = partials
        st.subheader("Summary")
        st.write(final_summary)
# ------------------------
# Quiz generation
# ------------------------
if "final_summary" in st.session_state:
    st.subheader("Generate Quiz")
    num_q = st.slider("Number of questions", 3, 10, 5, 1)

    if st.button("Create Questions"):
        final_summary = st.session_state.final_summary
        sentences = split_sentences(final_summary)
        kws = top_keywords(final_summary, k=max(12, num_q * 3))
        questions = []
        used_terms = set()

        for kw in kws:
            if kw.lower() in used_terms:
                continue
            s = find_sentence_with(kw, sentences)
            if not s or len(s.split()) < 6:
                continue
            q, opts, ans = make_mcq(kw, s, distract_pool=kws)
            questions.append((q, opts, ans))
            used_terms.add(kw.lower())
            if len(questions) >= num_q:
                break

        st.session_state.questions = questions    # <-- save for reruns
        st.session_state.quiz_checked = False     # reset check state

    # Render questions if we have them
    if st.session_state.get("questions"):
        score = 0
        for i, (q, opts, ans) in enumerate(st.session_state.questions):
            st.markdown(f"**Q{i+1}. {q}**")
            key = f"q_{i}"
            st.radio("Choose one:", opts, key=key, index=None)

        if st.button("Check Answers"):
            st.session_state.quiz_checked = True

        if st.session_state.get("quiz_checked"):
            for i, (_, opts, ans) in enumerate(st.session_state.questions):
                choice = st.session_state.get(f"q_{i}")
                correct = (choice == ans)
                if correct:
                    score += 1
                verdict = "✅ Correct" if correct else f"❌ Incorrect (Answer: **{ans}**)"
                st.write(f"Q{i+1}: {verdict}")
            st.success(f"Score: {score} / {len(st.session_state.questions)}")
    else:
        st.info("Click **Create Questions** to generate a quiz.")
