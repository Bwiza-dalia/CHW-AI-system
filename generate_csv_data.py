#!/usr/bin/env python3
"""
Generate CSV data for CHW grading system
Format: reference_answer, student_answer, language
"""

import csv
import json
from training_data_generator import TrainingDataGenerator

def generate_csv_data(output_file="grading_data.csv", num_samples=100):
    """
    Generate CSV data in the format: reference_answer, student_answer, language
    
    Args:
        output_file (str): Output CSV file path
        num_samples (int): Number of samples to generate
    """
    
    # Initialize the training data generator
    generator = TrainingDataGenerator()
    
    # Generate training data
    print(f"Generating {num_samples} samples...")
    training_data = generator.generate_diverse_training_data(num_samples=num_samples)
    
    # Write to CSV
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['reference_answer', 'student_answer', 'language']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        # Write header
        writer.writeheader()
        
        # Write data
        for item in training_data:
            writer.writerow({
                'reference_answer': item['reference_answer'],
                'student_answer': item['student_answer'],
                'language': item['language']
            })
    
    print(f"✅ CSV data saved to {output_file}")
    
    # Print statistics
    en_count = sum(1 for item in training_data if item['language'] == 'en')
    kin_count = sum(1 for item in training_data if item['language'] == 'kin')
    
    print(f"\n📊 Generated {len(training_data)} samples:")
    print(f"   English: {en_count} samples")
    print(f"   Kinyarwanda: {kin_count} samples")
    
    # Show sample data
    print(f"\n📝 Sample data:")
    for i, item in enumerate(training_data[:3]):
        print(f"\nSample {i+1} ({item['language']}):")
        print(f"  Reference: {item['reference_answer'][:100]}...")
        print(f"  Student: {item['student_answer'][:100]}...")

def generate_topic_specific_csv(output_file="topic_data.csv", topic="malaria_prevention", num_samples=20):
    """
    Generate CSV data for a specific health topic
    
    Args:
        output_file (str): Output CSV file path
        topic (str): Health topic (malaria_prevention, pregnancy_nutrition, etc.)
        num_samples (int): Number of samples per language
    """
    
    generator = TrainingDataGenerator()
    
    # Get topic data
    if topic not in generator.health_topics:
        print(f"❌ Topic '{topic}' not found. Available topics:")
        for t in generator.health_topics.keys():
            print(f"   - {t}")
        return
    
    topic_data = generator.health_topics[topic]
    csv_data = []
    
    # Generate samples for each language
    for language in ['en', 'kin']:
        if language in topic_data:
            lang_data = topic_data[language]
            reference = lang_data['reference']
            variations = lang_data['variations']
            scores = lang_data['scores']
            
            # Generate samples
            for _ in range(num_samples):
                import random
                idx = random.randint(0, len(variations) - 1)
                student_answer = variations[idx]
                
                csv_data.append({
                    'reference_answer': reference,
                    'student_answer': student_answer,
                    'language': language
                })
    
    # Write to CSV
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['reference_answer', 'student_answer', 'language']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for item in csv_data:
            writer.writerow(item)
    
    print(f"✅ Topic-specific CSV data saved to {output_file}")
    print(f"📊 Generated {len(csv_data)} samples for topic: {topic}")

def generate_custom_csv(output_file="custom_data.csv"):
    """
    Generate custom CSV data with specific examples
    """
    
    custom_data = [
        # English examples
        {
            'reference_answer': "The main malaria prevention methods include using insecticide-treated bed nets, indoor residual spraying, eliminating stagnant water, using repellents, and seeking early treatment.",
            'student_answer': "To prevent malaria, we should use treated bed nets every night, spray homes with insecticide, remove standing water, use repellent creams, and go to health center when we have fever.",
            'language': 'en'
        },
        {
            'reference_answer': "Pregnant women should eat diverse foods including fruits, vegetables, proteins like beans and meat, dairy products, and take supplements like folic acid.",
            'student_answer': "Pregnant women need to eat fruits, vegetables, beans, meat, milk products and take folic acid for the baby's health.",
            'language': 'en'
        },
        {
            'reference_answer': "Signs of dehydration in children include dry mouth, sunken eyes, reduced urination, and lethargy.",
            'student_answer': "Children show dehydration through dry mouth, eyes that look sunken, less urine, and being very tired.",
            'language': 'en'
        },
        {
            'reference_answer': "Proper hand hygiene involves washing hands with soap and clean water for at least 20 seconds, especially before eating and after using the toilet.",
            'student_answer': "Good hand washing means using soap and clean water for 20 seconds, before meals and after bathroom use.",
            'language': 'en'
        },
        {
            'reference_answer': "Vaccination is important because it protects children from serious diseases, prevents outbreaks, and saves lives by building immunity.",
            'student_answer': "Vaccines are crucial as they protect children from dangerous diseases, stop disease spread, and save lives through immunity.",
            'language': 'en'
        },
        
        # Kinyarwanda examples
        {
            'reference_answer': "Uburyo bwo kurinda malariya ni ukoresha ubunyangamugayo bwemewe, gusiga amabuye y'ubuvuzi mu nzu, gusiba amazi atemba, gukoresha ibikoresho byo kurinda inzige, no gufata ubuvuzi vuba.",
            'student_answer': "Kurinda malariya dukwiye gukoresha ubunyangamugayo buri joro, gusiga amabuye y'ubuvuzi mu nzu, gusiba amazi atemba, gukoresha ibikoresho byo kurinda inzige, no kujya ku kigo nderabuzima iyo dufite umuriro.",
            'language': 'kin'
        },
        {
            'reference_answer': "Abagore batwite bagomba kurya ibiryo bitandukanye birimo imbuto n'imboga, ibiryo by'ubwoko nka ibishyimbo n'inyama, ibiryo by'ubworozi, no gufata ibikoresho nka folic acid.",
            'student_answer': "Abagore bafite inda bakwiye kurya imbuto n'imboga, ibishyimbo, inyama, ibiryo by'ubworozi no gufata folic acid kugira ngo umwana abe afite ubuzima.",
            'language': 'kin'
        },
        {
            'reference_answer': "Ibimenyetso byo kutagira amazi mu mubiri by'abana ni umunwa utose, amaso yatse, kudashobora kw'ijoro, no kuba banyoroshye.",
            'student_answer': "Abana bwerekana ko batagira amazi mu mubiri binyuze mu munwa utose, amaso yatse, kudashobora kw'ijoro, no kuba banyoroshye cyane.",
            'language': 'kin'
        },
        {
            'reference_answer': "Gukaraba amaboko neza ni ukubyongera amaboko doreye n'isabuni n'amazi meza amasegonda 20, cyane cyane mbere yo kurya n'inyuma yo gukoresha ubwiherero.",
            'student_answer': "Gukaraba amaboko neza ni ukubyongera isabuni n'amazi meza amasegonda 20, mbere yo kurya n'inyuma yo gukoresha ubwiherero.",
            'language': 'kin'
        },
        {
            'reference_answer': "Ubwishingizi ni ngombwa kuko burinda abana indwara zikomeye, burinda indwara kubyongera, kandi buraba ubuzima dushyiraho ubwishingizi.",
            'student_answer': "Ubwishingizi ni ngombwa kuko burinda abana indwara zikomeye, burinda indwara kubyongera, kandi buraba ubuzima dushyiraho ubwishingizi.",
            'language': 'kin'
        }
    ]
    
    # Write to CSV
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['reference_answer', 'student_answer', 'language']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for item in custom_data:
            writer.writerow(item)
    
    print(f"✅ Custom CSV data saved to {output_file}")
    print(f"📊 Generated {len(custom_data)} samples")

def main():
    """Main function to generate different types of CSV data"""
    
    print("🏥 CHW CSV Data Generator")
    print("=" * 40)
    
    # Option 1: Generate comprehensive data
    print("\n1️⃣ Generating comprehensive CSV data...")
    generate_csv_data("comprehensive_data.csv", num_samples=100)
    
    # Option 2: Generate topic-specific data
    print("\n2️⃣ Generating malaria prevention data...")
    generate_topic_specific_csv("malaria_data.csv", "malaria_prevention", num_samples=15)
    
    # Option 3: Generate custom curated data
    print("\n3️⃣ Generating custom curated data...")
    generate_custom_csv("custom_data.csv")
    
    print("\n🎉 All CSV files generated successfully!")
    print("\n📁 Generated files:")
    print("   - comprehensive_data.csv (100 samples)")
    print("   - malaria_data.csv (30 samples)")
    print("   - custom_data.csv (10 samples)")
    
    print("\n💡 You can now use these CSV files for:")
    print("   - Batch processing in the Streamlit app")
    print("   - Training data for the model")
    print("   - Testing the grading system")

if __name__ == "__main__":
    main()
