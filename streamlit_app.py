import streamlit as st
from pipeline import run, TemplateBackend, to_markdown

st.set_page_config(page_title="Teacher Story & Lesson Planner", page_icon="📚", layout="wide")
st.title("📚 AI Story & Lesson-Plan Generator")
st.write("Create a classroom story and lesson plan from a topic and keywords.")
st.info("This first deployment checks the website using the project's offline template backend. It does not load the fine-tuned Qwen model yet.")

with st.form("lesson_form"):
    grade = st.selectbox("Grade", list(range(1, 9)), index=2)
    subject = st.selectbox("Subject", ["Science", "Math", "EVS", "Social Science", "Values", "English"])
    topic = st.text_input("Topic", placeholder="Water cycle")
    keywords = st.text_input("Keywords (comma-separated)", placeholder="rain, clouds, evaporation")
    submitted = st.form_submit_button("Generate", type="primary")

if submitted:
    if not topic.strip() or not keywords.strip():
        st.warning("Please enter a topic and at least one keyword.")
    else:
        with st.spinner("Creating your story and lesson plan..."):
            result = run(TemplateBackend(), int(grade), subject, topic.strip(), keywords.strip(), max_retries=2)
        st.subheader("Story")
        st.write(result["story"] or "No story was returned.")
        st.subheader("Lesson Plan")
        st.write(result["lesson"] or "No lesson plan was returned.")
        st.subheader("Quality checks")
        if result["passed"]:
            st.success("The output passed the project's checks.")
        else:
            st.warning("Please review these items:")
            for issue in result["issues"]:
                st.write(f"- {issue}")
        st.write("Metrics", result["metrics"])
        st.download_button("Download lesson plan (.md)", to_markdown(result["plan"], result["story"], result["lesson"]), file_name="lesson_plan.md", mime="text/markdown")
