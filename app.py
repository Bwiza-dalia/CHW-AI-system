from nltk.tokenize import word_tokenize, sent_tokenize
#
import streamlit as st
from grader import BilingualNLPGrader

# Initialize and train the grader (for demo, use the same sample data)
@st.cache_resource
def get_trained_grader():
    grader = BilingualNLPGrader()

    training_data = [
        {
            'reference_answer': "The main malaria prevention methods include using insecticide-treated bed nets, indoor residual spraying, eliminating stagnant water, using repellents, and seeking early treatment.",
            'student_answer': "To prevent malaria, we should use treated bed nets every night, spray homes with insecticide, remove standing water, use repellent creams, and go to health center when we have fever.",
            'score': 5,
            'language': 'en'
        },
        {
            'reference_answer': "The main malaria prevention methods include using insecticide-treated bed nets, indoor residual spraying, eliminating stagnant water, using repellents, and seeking early treatment.",
            'student_answer': "Use bed nets and remove standing water to prevent malaria.",
            'score': 3,
            'language': 'en'
        }
    ]
    grader.train(training_data)
    return grader

grader = get_trained_grader()

st.title("NLP Grading Dashboard")

reference_answer = st.text_area("Reference Answer", height=100)
student_answer = st.text_area("Student Answer", height=100)
language = st.selectbox("Language", options=["en", "kin"], format_func=lambda x: "English" if x == "en" else "Kinyarwanda")

if st.button("Grade Answer"):
    if reference_answer.strip() and student_answer.strip():
        try:
            result = grader.grade(reference_answer, student_answer, language=language)
            st.success(f"Score: {result['score']} (Confidence: {result['confidence']:.2f})")
            st.info(f"Feedback: {result['feedback']}")
            with st.expander("Show Features"):
                st.json(result['features'])
        except Exception as e:
            st.error(f"Error grading answer: {e}")
    else:
        st.warning("Please enter both a reference answer and a student answer.")