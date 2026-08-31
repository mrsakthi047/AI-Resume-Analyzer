
import streamlit as st
import re

from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# -----------------------------
# Page Configuration
# -----------------------------

st.set_page_config(
    page_title="AI Resume Analyzer",
    page_icon="📄",
    layout="wide"
)


# -----------------------------
# Functions
# -----------------------------

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def extract_resume_text(uploaded_file):
    reader = PdfReader(uploaded_file)

    text = ""

    for page in reader.pages:
        page_text = page.extract_text()

        if page_text:
            text += page_text + "\n"

    return text


# -----------------------------
# Skills Database
# -----------------------------

skills = [
    "python",
    "java",
    "sql",
    "machine learning",
    "artificial intelligence",
    "pandas",
    "numpy",
    "scikit-learn",
    "tensorflow",
    "deep learning",
    "html",
    "css",
    "javascript",
    "git",
    "github",
    "power bi",
    "excel",
    "communication",
    "teamwork"
]


# -----------------------------
# Recommendations
# -----------------------------

skill_recommendations = {
    "python": "Improve Python programming, functions, OOP and problem solving.",
    "java": "Learn Java fundamentals and Object-Oriented Programming.",
    "sql": "Practice SQL queries, joins, subqueries and databases.",
    "machine learning": "Learn supervised and unsupervised ML algorithms.",
    "artificial intelligence": "Study AI fundamentals and intelligent systems.",
    "pandas": "Practice data cleaning and analysis using Pandas.",
    "numpy": "Learn NumPy arrays and numerical operations.",
    "scikit-learn": "Build ML models using Scikit-learn.",
    "tensorflow": "Learn neural networks and deep learning using TensorFlow.",
    "deep learning": "Study neural networks, CNNs and deep learning.",
    "html": "Learn HTML structure and semantic elements.",
    "css": "Practice CSS layouts and responsive design.",
    "javascript": "Learn JavaScript fundamentals and DOM manipulation.",
    "git": "Learn Git commits, branches and version control.",
    "github": "Practice GitHub repositories and collaboration.",
    "power bi": "Learn dashboards and data visualization using Power BI.",
    "excel": "Improve formulas, charts and pivot tables.",
    "communication": "Practice technical communication and presentations.",
    "teamwork": "Build collaboration skills through projects."
}


# -----------------------------
# User Interface
# -----------------------------

st.title("📄 AI Resume Analyzer")

st.write(
    "Analyze your resume against a job description using "
    "NLP, TF-IDF and skill matching."
)

st.divider()

uploaded_file = st.file_uploader(
    "📤 Upload your Resume",
    type=["pdf"]
)

job_description = st.text_area(
    "💼 Paste Job Description",
    height=220,
    placeholder="Paste the job description here..."
)


# -----------------------------
# Analyze Button
# -----------------------------

if st.button("🚀 Analyze Resume"):

    if uploaded_file is None:
        st.warning("Please upload your resume.")

    elif not job_description.strip():
        st.warning("Please enter a job description.")

    else:

        # Extract Resume Text
        resume_text = extract_resume_text(uploaded_file)

        # Clean Text
        cleaned_resume = clean_text(resume_text)
        cleaned_job = clean_text(job_description)

        # Find Resume Skills
        found_skills = []

        for skill in skills:
            if skill in cleaned_resume:
                found_skills.append(skill)

        # Find Required Skills
        required_skills = []

        for skill in skills:
            if skill in cleaned_job:
                required_skills.append(skill)

        # Matched Skills
        matched_skills = []

        for skill in required_skills:
            if skill in found_skills:
                matched_skills.append(skill)

        # Missing Skills
        missing_skills = []

        for skill in required_skills:
            if skill not in found_skills:
                missing_skills.append(skill)

        # Skill Match Score
        if len(required_skills) > 0:
            skill_match_score = (
                len(matched_skills) / len(required_skills)
            ) * 100
        else:
            skill_match_score = 0

        # TF-IDF Similarity
        documents = [cleaned_resume, cleaned_job]

        vectorizer = TfidfVectorizer()

        tfidf_matrix = vectorizer.fit_transform(documents)

        similarity = cosine_similarity(
            tfidf_matrix[0:1],
            tfidf_matrix[1:2]
        )

        similarity_score = similarity[0][0] * 100

        # Final Score
        final_score = (
            skill_match_score * 0.6
        ) + (
            similarity_score * 0.4
        )

        # -----------------------------
        # Results
        # -----------------------------

        st.divider()

        st.subheader("📊 Resume Analysis")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Skill Match",
                f"{skill_match_score:.1f}%"
            )

        with col2:
            st.metric(
                "NLP Similarity",
                f"{similarity_score:.1f}%"
            )

        with col3:
            st.metric(
                "Final Match",
                f"{final_score:.1f}%"
            )

        # Matched Skills
        st.subheader("✅ Matched Skills")

        if matched_skills:
            for skill in matched_skills:
                st.success(skill.title())
        else:
            st.info("No matching skills found.")

        # Missing Skills
        st.subheader("❌ Missing Skills")

        if missing_skills:

            for skill in missing_skills:
                st.error(skill.title())

            st.subheader("💡 Recommendations")

            for skill in missing_skills:
                recommendation = skill_recommendations.get(
                    skill,
                    "Develop this skill through practical projects."
                )

                st.write(
                    f"**{skill.title()}** → {recommendation}"
                )

        else:
            st.success(
                "🎉 No major missing skills detected!"
            )

        # Final Recommendation
        st.subheader("🎯 Overall Recommendation")

        if final_score >= 80:
            st.success(
                "Excellent match! Your resume strongly matches this job."
            )

        elif final_score >= 60:
            st.info(
                "Good match! Improve a few missing skills."
            )

        elif final_score >= 40:
            st.warning(
                "Moderate match. Consider improving your technical skills."
            )

        else:
            st.error(
                "Low match. Develop more skills required for this role."
            )

