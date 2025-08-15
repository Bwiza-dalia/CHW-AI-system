# 🏥 CHW NLP Grading System

A bilingual (English/Kinyarwanda) automated grading system designed for Community Health Worker (CHW) training and assessment. This system uses advanced Natural Language Processing techniques to evaluate student answers against reference answers and provide detailed feedback.

## 🌟 Features

### Core Functionality
- **Bilingual Support**: Grade answers in both English and Kinyarwanda
- **Health Domain Focus**: Specialized for community health topics
- **Intelligent Scoring**: 1-5 scale with confidence metrics
- **Detailed Feedback**: Contextual feedback in both languages
- **Multiple NLP Features**: TF-IDF, Jaccard similarity, keyword density, and more

### Advanced Features
- **Model Training**: Train custom models with your data
- **Batch Processing**: Process multiple answers at once
- **Analytics Dashboard**: Visualize training data and model performance
- **Model Persistence**: Save and load trained models
- **Comprehensive UI**: Modern Streamlit interface with multiple pages

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <repository-url>
cd CHW

# Install dependencies
pip install -r requirements.txt

# Download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
```

### 2. Generate Training Data

```bash
# Generate sample training data
python training_data_generator.py
```

### 3. Run the Application

```bash
# Run the improved Streamlit app
streamlit run improved_app.py
```

## 📖 Usage Guide

### Basic Grading

1. **Navigate to "Grade Answers"** in the sidebar
2. **Enter Reference Answer**: The correct/expected answer
3. **Enter Student Answer**: The answer to be graded
4. **Select Language**: English or Kinyarwanda
5. **Click "Grade Answer"** to get results

### Model Training

1. **Navigate to "Train Model"** in the sidebar
2. **Generate Training Data**: Click to create sample data
3. **Configure Training**: Set validation split (default: 20%)
4. **Train Model**: Click "Train Model" to start training
5. **View Results**: Check training metrics and validation scores

### Batch Processing

1. **Navigate to "Batch Processing"** in the sidebar
2. **Upload CSV File**: Must contain columns: `reference_answer`, `student_answer`, `language`
3. **Process Batch**: Click to grade all answers
4. **Download Results**: Get CSV file with scores and feedback

### Analytics

1. **Navigate to "Analytics"** in the sidebar
2. **View Training Data**: Language distribution, score distribution
3. **Model Information**: Training metrics and performance

## 🏗️ Architecture

### Core Components

#### `ImprovedBilingualNLPGrader` (`improved_grader.py`)
- Main grading engine with enhanced features
- Advanced NLP preprocessing
- Multiple similarity metrics
- Model training and evaluation
- Model persistence (save/load)

#### `TrainingDataGenerator` (`training_data_generator.py`)
- Generate comprehensive training data
- Multiple health topics (malaria, pregnancy, hygiene, etc.)
- Bilingual support with variations
- Edge cases and challenging examples

#### `Improved Streamlit App` (`improved_app.py`)
- Multi-page interface
- Real-time grading
- Batch processing
- Analytics dashboard
- Model management

### NLP Features

The system extracts the following features:

1. **Length Ratios**: Text and word count ratios
2. **TF-IDF Similarity**: Cosine similarity using TF-IDF vectors
3. **Jaccard Similarity**: Set-based word overlap
4. **Term Coverage**: Percentage of reference terms in student answer
5. **Keyword Density**: Health domain terminology usage
6. **Semantic Density**: Vocabulary diversity
7. **Additional Metrics**: Word length, sentence complexity

### Model Architecture

- **Vectorizer**: TF-IDF with 10,000 features, 1-3 n-grams
- **Classifier**: Random Forest with 200 estimators
- **Preprocessing**: Language-specific stopwords and stemming
- **Scaling**: StandardScaler for feature normalization

## 📊 Training Data

### Health Topics Covered

1. **Malaria Prevention**
   - Bed nets, spraying, water management
   - Early treatment and repellents

2. **Pregnancy Nutrition**
   - Diverse foods, supplements
   - Folic acid and balanced diet

3. **Dehydration Signs**
   - Symptoms in children
   - Recognition and response

4. **Hand Hygiene**
   - Proper washing techniques
   - Timing and duration

5. **Vaccination Importance**
   - Disease prevention
   - Community health benefits

### Data Structure

```json
{
  "reference_answer": "The main malaria prevention methods...",
  "student_answer": "To prevent malaria, we should...",
  "score": 5,
  "language": "en",
  "topic": "malaria_prevention"
}
```

## 🔧 Configuration

### Model Parameters

- **TF-IDF Features**: 10,000 maximum features
- **N-gram Range**: 1-3 (unigrams, bigrams, trigrams)
- **Random Forest**: 200 estimators, max_depth=15
- **Validation Split**: Configurable (default: 20%)

### Language Support

#### English
- Standard NLTK stopwords
- Porter stemming
- Health domain keywords

#### Kinyarwanda
- Custom stopwords list
- No stemming (preserves morphology)
- Health domain keywords in Kinyarwanda

## 📈 Performance

### Model Evaluation

The system provides comprehensive evaluation metrics:

- **R² Score**: Model fit quality
- **Mean Absolute Error**: Average prediction error
- **Root Mean Square Error**: Standard deviation of errors
- **Cross-validation**: Robust performance estimation

### Expected Performance

With adequate training data:
- **R² Score**: 0.7-0.9
- **MAE**: 0.3-0.6 points
- **Confidence**: 0.6-0.9 for well-trained models

## 🛠️ Development

### Project Structure

```
CHW/
├── improved_grader.py          # Enhanced grading engine
├── training_data_generator.py  # Training data generation
├── improved_app.py            # Main Streamlit application
├── grader.py                  # Original grading engine
├── app.py                     # Original Streamlit app
├── kinyarwanda_corpus.py      # Kinyarwanda text processing
├── requirements.txt           # Python dependencies
├── README.md                  # This file
├── training_data.json         # Generated training data
├── test_data.json            # Test data for evaluation
└── trained_model.pkl         # Saved trained model
```

### Adding New Features

1. **New Health Topics**: Add to `TrainingDataGenerator.health_topics`
2. **New Languages**: Extend stopwords and keywords dictionaries
3. **New Features**: Add to `extract_features()` method
4. **New Models**: Implement in `ImprovedBilingualNLPGrader`

### Testing

```bash
# Test the grading system
python -c "
from improved_grader import ImprovedBilingualNLPGrader
from training_data_generator import TrainingDataGenerator

# Generate test data
generator = TrainingDataGenerator()
test_data = generator.create_test_data(10)

# Test grading
grader = ImprovedBilingualNLPGrader()
grader.train(test_data)

# Test single grading
result = grader.grade('Wash hands', 'Clean hands with soap', 'en')
print(f'Score: {result[\"score\"]}, Confidence: {result[\"confidence\"]}')
"
```

## 🙏 Acknowledgments

- **NLTK**: Natural language processing toolkit
- **Scikit-learn**: Machine learning library
- **Streamlit**: Web application framework
- **Community Health Workers**: For their valuable work and feedback
