#!/usr/bin/env python3
"""
Comprehensive testing script for the CHW NLP Grading System
"""

import sys
import traceback
from pathlib import Path

def test_imports():
    """Test if all required modules can be imported"""
    print("🔍 Testing imports...")
    
    try:
        import numpy as np
        import pandas as pd
        import streamlit as st
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.ensemble import RandomForestRegressor
        import nltk
        from nltk.tokenize import word_tokenize
        from nltk.corpus import stopwords
        print("✅ All imports successful")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_nltk_data():
    """Test if NLTK data is available"""
    print("\n🔍 Testing NLTK data...")
    
    try:
        import nltk
        from nltk.tokenize import word_tokenize
        from nltk.corpus import stopwords
        
        # Test tokenization
        tokens = word_tokenize("This is a test sentence.")
        assert len(tokens) > 0, "Tokenization failed"
        
        # Test stopwords
        stop_words = stopwords.words('english')
        assert len(stop_words) > 0, "Stopwords not loaded"
        
        print("✅ NLTK data available and working")
        return True
    except Exception as e:
        print(f"❌ NLTK data error: {e}")
        print("💡 Run: python -c \"import nltk; nltk.download('punkt'); nltk.download('stopwords')\"")
        return False

def test_grader():
    """Test the improved grader functionality"""
    print("\n🔍 Testing improved grader...")
    
    try:
        from improved_grader import ImprovedBilingualNLPGrader
        
        # Initialize grader
        grader = ImprovedBilingualNLPGrader()
        assert grader is not None, "Grader initialization failed"
        
        # Test preprocessing
        processed = grader.preprocess_text("This is a test sentence.", "en")
        assert isinstance(processed, str), "Preprocessing failed"
        
        # Test feature extraction
        features = grader.extract_features(
            "The main malaria prevention methods include using bed nets.",
            "Use bed nets to prevent malaria.",
            "en"
        )
        assert isinstance(features, dict), "Feature extraction failed"
        assert len(features) > 0, "No features extracted"
        
        print("✅ Improved grader functionality working")
        return True
    except Exception as e:
        print(f"❌ Grader error: {e}")
        traceback.print_exc()
        return False

def test_training_data_generator():
    """Test the training data generator"""
    print("\n🔍 Testing training data generator...")
    
    try:
        from training_data_generator import TrainingDataGenerator
        
        # Initialize generator
        generator = TrainingDataGenerator()
        assert generator is not None, "Generator initialization failed"
        
        # Test data generation
        training_data = generator.generate_training_data(num_samples_per_topic=2)
        assert isinstance(training_data, list), "Training data generation failed"
        assert len(training_data) > 0, "No training data generated"
        
        # Test data structure
        sample = training_data[0]
        required_keys = ['reference_answer', 'student_answer', 'score', 'language']
        for key in required_keys:
            assert key in sample, f"Missing key: {key}"
        
        print("✅ Training data generator working")
        return True
    except Exception as e:
        print(f"❌ Training data generator error: {e}")
        traceback.print_exc()
        return False

def test_model_training():
    """Test model training functionality"""
    print("\n🔍 Testing model training...")
    
    try:
        from improved_grader import ImprovedBilingualNLPGrader
        from training_data_generator import TrainingDataGenerator
        
        # Generate small training dataset
        generator = TrainingDataGenerator()
        training_data = generator.generate_training_data(num_samples_per_topic=3)
        
        # Initialize and train grader
        grader = ImprovedBilingualNLPGrader()
        metrics = grader.train(training_data, validation_split=0.2)
        
        assert grader.is_trained, "Model training failed"
        assert 'train_score' in metrics, "Training metrics missing"
        
        # Test grading
        result = grader.grade(
            "Wash hands with soap.",
            "Clean hands with soap and water.",
            "en"
        )
        
        assert 'score' in result, "Grading result missing score"
        assert 'confidence' in result, "Grading result missing confidence"
        assert 'feedback' in result, "Grading result missing feedback"
        
        print("✅ Model training working")
        return True
    except Exception as e:
        print(f"❌ Model training error: {e}")
        traceback.print_exc()
        return False

def run_all_tests():
    """Run all tests and provide summary"""
    print("🚀 Starting CHW NLP Grading System Tests\n")
    print("=" * 50)
    
    tests = [
        ("Imports", test_imports),
        ("NLTK Data", test_nltk_data),
        ("Improved Grader", test_grader),
        ("Training Data Generator", test_training_data_generator),
        ("Model Training", test_model_training)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 TEST SUMMARY")
    print("=" * 50)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:25} {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! System is ready to use.")
        return True
    else:
        print("⚠️ Some tests failed. Please check the errors above.")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
