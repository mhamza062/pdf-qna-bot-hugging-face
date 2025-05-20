import streamlit as st
from utils import get_falcon_pipeline, extract_text_from_pdf  # ✅ Updated import

def main():
    st.set_page_config(page_title="📄 PDF Q&A Bot", layout="wide")
    st.title("📄 PDF Question Answering Bot")

    if "pipe" not in st.session_state:
        with st.spinner("Loading model... (may take 30-60 secs)"):
            st.session_state.pipe = get_falcon_pipeline()  # ✅ Updated function

    uploaded_file = st.file_uploader("Upload your PDF", type=["pdf"])

    if uploaded_file:
        with st.spinner("Extracting text..."):
            pdf_text = extract_text_from_pdf(uploaded_file)

        st.text_area("📄 Extracted PDF Text (first 1000 characters)", pdf_text[:1000])

        context = pdf_text[:700]  # ✅ Falcon can't handle long context
        question = st.text_input("❓ Ask a question based on the PDF")

        if question:
            with st.spinner("Answering..."):
                prompt = f"{context}\n\nQuestion: {question}\nAnswer:"
                answer = st.session_state.pipe(
                    prompt,
                    max_new_tokens=30,
                    do_sample=False,
                    temperature=0.7,
                    top_k=50,
                    top_p=0.95,
                    repetition_penalty=1.2
                )
                st.success(answer[0]['generated_text'].split("Answer:")[-1].strip())

# ✅ CORRECT name guard
if __name__ == "__main__":
    main()
