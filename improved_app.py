import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import json
import time
from improved_grader import ImprovedBilingualNLPGrader
from training_data_generator import TrainingDataGenerator

# Page configuration
st.set_page_config(
    page_title="CHW NLP Grading System",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .score-high { color: #28a745; }
    .score-medium { color: #ffc107; }
    .score-low { color: #dc3545; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_grader():
    """Load or initialize the grader"""
    model_path = "trained_model.pkl"
    if Path(model_path).exists():
        grader = ImprovedBilingualNLPGrader(model_path=model_path)
        st.success("✅ Loaded pre-trained model")
    else:
        grader = ImprovedBilingualNLPGrader()
        st.info("ℹ️ No pre-trained model found. Please train the model first.")
    return grader

@st.cache_data
def load_training_data():
    """Load training data if available"""
    training_path = "training_data.json"
    if Path(training_path).exists():
        with open(training_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None

def main():
    st.markdown('<h1 class="main-header">🏥 CHW NLP Grading System</h1>', unsafe_allow_html=True)
    
    # Initialize grader
    grader = load_grader()
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page",
        ["Grade Answers", "Train Model", "Batch Processing", "Analytics", "Settings"]
    )
    
    if page == "Grade Answers":
        show_grading_page(grader)
    elif page == "Train Model":
        show_training_page(grader)
    elif page == "Batch Processing":
        show_batch_page(grader)
    elif page == "Analytics":
        show_analytics_page(grader)
    elif page == "Settings":
        show_settings_page()

def show_grading_page(grader):
    """Main grading interface"""
    st.header("📝 Grade Student Answers")
    
    if not grader.is_trained:
        st.warning("⚠️ Model is not trained. Please train the model first in the 'Train Model' section.")
        return
    
    # Input section
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Reference Answer")
        reference_answer = st.text_area(
            "Enter the reference/correct answer:",
            height=150,
            placeholder="Enter the expected answer here..."
        )
        
        language = st.selectbox(
            "Language",
            options=["en", "kin"],
            format_func=lambda x: "English" if x == "en" else "Kinyarwanda"
        )
    
    with col2:
        st.subheader("Student Answer")
        student_answer = st.text_area(
            "Enter the student's answer:",
            height=150,
            placeholder="Enter the student's response here..."
        )
        
        # Quick examples
        st.subheader("Quick Examples")
        if st.button("Load Malaria Example"):
            if language == "en":
                st.session_state.reference = "The main malaria prevention methods include using insecticide-treated bed nets, indoor residual spraying, eliminating stagnant water, using repellents, and seeking early treatment."
                st.session_state.student = "To prevent malaria, we should use treated bed nets every night, spray homes with insecticide, remove standing water, use repellent creams, and go to health center when we have fever."
            else:
                st.session_state.reference = "Uburyo bwo kurinda malariya ni ukoresha ubunyangamugayo bwemewe, gusiga amabuye y'ubuvuzi mu nzu, gusiba amazi atemba, gukoresha ibikoresho byo kurinda inzige, no gufata ubuvuzi vuba."
                st.session_state.student = "Kurinda malariya dukwiye gukoresha ubunyangamugayo buri joro, gusiga amabuye y'ubuvuzi mu nzu, gusiba amazi atemba, gukoresha ibikoresho byo kurinda inzige, no kujya ku kigo nderabuzima iyo dufite umuriro."
    
    # Use session state for examples
    if hasattr(st.session_state, 'reference'):
        reference_answer = st.session_state.reference
    if hasattr(st.session_state, 'student'):
        student_answer = st.session_state.student
    
    # Grade button
    if st.button("🎯 Grade Answer", type="primary", use_container_width=True):
        if reference_answer.strip() and student_answer.strip():
            with st.spinner("Grading answer..."):
                try:
                    result = grader.grade(reference_answer, student_answer, language=language)
                    
                    # Display results
                    st.success("✅ Grading Complete!")
                    
                    # Score display
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        score_class = "score-high" if result['score'] >= 4 else "score-medium" if result['score'] >= 3 else "score-low"
                        st.markdown(f"""
                        <div class="metric-card">
                            <h3>Score</h3>
                            <h2 class="{score_class}">{result['score']}/5</h2>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col2:
                        confidence_color = "score-high" if result['confidence'] >= 0.7 else "score-medium" if result['confidence'] >= 0.4 else "score-low"
                        st.markdown(f"""
                        <div class="metric-card">
                            <h3>Confidence</h3>
                            <h2 class="{confidence_color}">{result['confidence']:.2f}</h2>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col3:
                        st.markdown(f"""
                        <div class="metric-card">
                            <h3>Raw Score</h3>
                            <h2>{result['raw_score']:.2f}</h2>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Feedback
                    st.subheader("📝 Feedback")
                    st.info(result['feedback'])
                    
                    # Detailed features
                    with st.expander("🔍 View Detailed Features"):
                        features_df = pd.DataFrame(list(result['features'].items()), columns=['Feature', 'Value'])
                        st.dataframe(features_df, use_container_width=True)
                        
                        # Feature visualization
                        fig = px.bar(
                            features_df, 
                            x='Feature', 
                            y='Value',
                            title="Feature Values"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                except Exception as e:
                    st.error(f"❌ Error grading answer: {e}")
        else:
            st.warning("⚠️ Please enter both a reference answer and a student answer.")

def show_training_page(grader):
    """Model training interface"""
    st.header("🤖 Train Model")
    
    # Training data options
    st.subheader("Training Data")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Option 1: Generate Training Data**")
        if st.button("Generate Sample Training Data"):
            with st.spinner("Generating training data..."):
                generator = TrainingDataGenerator()
                training_data = generator.generate_diverse_training_data(num_samples=200)
                generator.save_training_data(training_data, 'training_data.json')
                st.success(f"✅ Generated {len(training_data)} training samples")
                st.session_state.training_data = training_data
    
    with col2:
        st.write("**Option 2: Upload Training Data**")
        uploaded_file = st.file_uploader(
            "Upload JSON training data",
            type=['json'],
            help="Upload a JSON file with training data in the format: [{'reference_answer': '...', 'student_answer': '...', 'score': 5, 'language': 'en'}]"
        )
        
        if uploaded_file is not None:
            try:
                training_data = json.load(uploaded_file)
                st.success(f"✅ Loaded {len(training_data)} training samples")
                st.session_state.training_data = training_data
            except Exception as e:
                st.error(f"❌ Error loading file: {e}")
    
    # Load existing training data
    existing_data = load_training_data()
    if existing_data and 'training_data' not in st.session_state:
        st.session_state.training_data = existing_data
        st.info(f"ℹ️ Loaded {len(existing_data)} existing training samples")
    
    # Training section
    if 'training_data' in st.session_state:
        training_data = st.session_state.training_data
        
        st.subheader("Training Statistics")
        
        # Data statistics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Samples", len(training_data))
        
        with col2:
            en_count = sum(1 for item in training_data if item['language'] == 'en')
            st.metric("English Samples", en_count)
        
        with col3:
            kin_count = sum(1 for item in training_data if item['language'] == 'kin')
            st.metric("Kinyarwanda Samples", kin_count)
        
        # Score distribution
        score_counts = {}
        for item in training_data:
            score = item['score']
            score_counts[score] = score_counts.get(score, 0) + 1
        
        fig = px.bar(
            x=list(score_counts.keys()),
            y=list(score_counts.values()),
            title="Score Distribution",
            labels={'x': 'Score', 'y': 'Count'}
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Training options
        st.subheader("Training Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            validation_split = st.slider(
                "Validation Split",
                min_value=0.0,
                max_value=0.5,
                value=0.2,
                step=0.05,
                help="Percentage of data to use for validation"
            )
        
        with col2:
            if st.button("🚀 Train Model", type="primary"):
                with st.spinner("Training model..."):
                    try:
                        # Split data for training
                        train_size = int(len(training_data) * (1 - validation_split))
                        train_data = training_data[:train_size]
                        val_data = training_data[train_size:]
                        
                        # Train model
                        metrics = grader.train(train_data, validation_split=0.0)  # No additional split since we already split
                        
                        st.success("✅ Model training completed!")
                        
                        # Display training metrics
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.metric("Training R² Score", f"{metrics['train_score']:.3f}")
                        
                        with col2:
                            if val_data:
                                # Evaluate on validation set
                                val_metrics = grader.evaluate_model(val_data)
                                st.metric("Validation R² Score", f"{val_metrics['r2']:.3f}")
                        
                        # Save model
                        grader.save_model("trained_model.pkl")
                        st.success("💾 Model saved successfully!")
                        
                    except Exception as e:
                        st.error(f"❌ Training error: {e}")

def show_batch_page(grader):
    """Batch processing interface"""
    st.header("📊 Batch Processing")
    
    if not grader.is_trained:
        st.warning("⚠️ Model is not trained. Please train the model first.")
        return
    
    # Upload batch data
    st.subheader("Upload Batch Data")
    
    uploaded_file = st.file_uploader(
        "Upload CSV file with columns: reference_answer, student_answer, language",
        type=['csv'],
        help="CSV file should have columns: reference_answer, student_answer, language (en/kin)"
    )
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success(f"✅ Loaded {len(df)} rows")
            
            # Preview data
            st.subheader("Data Preview")
            st.dataframe(df.head(), use_container_width=True)
            
            # Process batch
            if st.button("🔍 Process Batch", type="primary"):
                with st.spinner("Processing batch..."):
                    results = []
                    
                    for idx, row in df.iterrows():
                        try:
                            result = grader.grade(
                                row['reference_answer'],
                                row['student_answer'],
                                language=row.get('language', 'en')
                            )
                            results.append({
                                'index': idx,
                                'score': result['score'],
                                'confidence': result['confidence'],
                                'raw_score': result['raw_score'],
                                'feedback': result['feedback']
                            })
                        except Exception as e:
                            results.append({
                                'index': idx,
                                'score': None,
                                'confidence': None,
                                'raw_score': None,
                                'feedback': f"Error: {e}"
                            })
                    
                    # Create results dataframe
                    results_df = pd.DataFrame(results)
                    combined_df = pd.concat([df, results_df.drop('index', axis=1)], axis=1)
                    
                    st.success("✅ Batch processing completed!")
                    
                    # Display results
                    st.subheader("Results")
                    st.dataframe(combined_df, use_container_width=True)
                    
                    # Download results
                    csv = combined_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Results",
                        data=csv,
                        file_name="batch_results.csv",
                        mime="text/csv"
                    )
                    
                    # Summary statistics
                    st.subheader("Summary Statistics")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        avg_score = results_df['score'].mean()
                        st.metric("Average Score", f"{avg_score:.2f}")
                    
                    with col2:
                        avg_confidence = results_df['confidence'].mean()
                        st.metric("Average Confidence", f"{avg_confidence:.2f}")
                    
                    with col3:
                        success_rate = (results_df['score'].notna().sum() / len(results_df)) * 100
                        st.metric("Success Rate", f"{success_rate:.1f}%")
                    
        except Exception as e:
            st.error(f"❌ Error processing file: {e}")

def show_analytics_page(grader):
    """Analytics and insights page"""
    st.header("📈 Analytics & Insights")
    
    # Load training data for analytics
    training_data = load_training_data()
    
    if training_data:
        st.subheader("Training Data Analytics")
        
        # Language distribution
        lang_counts = pd.DataFrame(training_data)['language'].value_counts()
        fig = px.pie(
            values=lang_counts.values,
            names=lang_counts.index,
            title="Language Distribution"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Score distribution by language
        df = pd.DataFrame(training_data)
        fig = px.histogram(
            df,
            x='score',
            color='language',
            title="Score Distribution by Language",
            barmode='group'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Topic analysis if available
        if 'topic' in df.columns:
            topic_counts = df['topic'].value_counts()
            fig = px.bar(
                x=topic_counts.index,
                y=topic_counts.values,
                title="Topic Distribution"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Model performance metrics
    if grader.is_trained:
        st.subheader("Model Information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Model Status", "✅ Trained")
            st.metric("Model Type", "Random Forest")
        
        with col2:
            st.metric("Features", "10+ NLP Features")
            st.metric("Languages", "English & Kinyarwanda")

def show_settings_page():
    """Settings and configuration page"""
    st.header("⚙️ Settings")
    
    st.subheader("System Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Model Configuration**")
        st.write("- TF-IDF Vectorizer: 10,000 features")
        st.write("- N-gram Range: 1-3")
        st.write("- Random Forest: 200 estimators")
        st.write("- Feature Scaling: StandardScaler")
    
    with col2:
        st.write("**Supported Languages**")
        st.write("- English (en)")
        st.write("- Kinyarwanda (kin)")
        st.write("- Custom stopwords for both")
        st.write("- Health domain keywords")
    
    st.subheader("Data Management")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🗑️ Clear Training Data"):
            if Path("training_data.json").exists():
                Path("training_data.json").unlink()
                st.success("✅ Training data cleared")
            else:
                st.info("ℹ️ No training data to clear")
    
    with col2:
        if st.button("🗑️ Clear Model"):
            if Path("trained_model.pkl").exists():
                Path("trained_model.pkl").unlink()
                st.success("✅ Model cleared")
            else:
                st.info("ℹ️ No model to clear")

if __name__ == "__main__":
    main()
