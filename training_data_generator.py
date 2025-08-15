import json
import random
from typing import List, Dict, Any

class TrainingDataGenerator:
    """Generate comprehensive training data for the CHW grading system"""
    
    def __init__(self):
        self.health_topics = {
            'malaria_prevention': {
                'en': {
                    'reference': "The main malaria prevention methods include using insecticide-treated bed nets, indoor residual spraying, eliminating stagnant water, using repellents, and seeking early treatment.",
                    'variations': [
                        "To prevent malaria, we should use treated bed nets every night, spray homes with insecticide, remove standing water, use repellent creams, and go to health center when we have fever.",
                        "Malaria prevention involves sleeping under treated nets, spraying walls with insecticide, removing water containers, applying mosquito repellent, and getting treatment early.",
                        "Use bed nets and remove standing water to prevent malaria.",
                        "Sleep under nets and spray house to prevent malaria.",
                        "I don't know how to prevent malaria."
                    ],
                    'scores': [5, 5, 3, 2, 1]
                },
                'kin': {
                    'reference': "Uburyo bwo kurinda malariya ni ukoresha ubunyangamugayo bwemewe, gusiga amabuye y'ubuvuzi mu nzu, gusiba amazi atemba, gukoresha ibikoresho byo kurinda inzige, no gufata ubuvuzi vuba.",
                    'variations': [
                        "Kurinda malariya dukwiye gukoresha ubunyangamugayo buri joro, gusiga amabuye y'ubuvuzi mu nzu, gusiba amazi atemba, gukoresha ibikoresho byo kurinda inzige, no kujya ku kigo nderabuzima iyo dufite umuriro.",
                        "Kurinda malariya ni ukuryama munsi y'ubunyangamugayo, gusiga amabuye y'ubuvuzi ku maso, gusiba ibikoresho by'amazi, gukoresha ibikoresho byo kurinda inzige, no gufata ubuvuzi vuba.",
                        "Koresha ubunyangamugayo no gusiba amazi atemba kurinda malariya.",
                        "Rya munsi y'ubunyangamugayo no gusiga amabuye mu nzu kurinda malariya.",
                        "Sinzi uko kurinda malariya."
                    ],
                    'scores': [5, 5, 3, 2, 1]
                }
            },
            'pregnancy_nutrition': {
                'en': {
                    'reference': "Pregnant women should eat diverse foods including fruits, vegetables, proteins like beans and meat, dairy products, and take supplements like folic acid.",
                    'variations': [
                        "Pregnant women need to eat fruits, vegetables, beans, meat, milk products and take folic acid for the baby's health.",
                        "Women who are pregnant should consume a variety of foods: fruits, vegetables, protein sources, dairy, and folic acid supplements.",
                        "Eat fruits and vegetables when pregnant.",
                        "Pregnant women should eat food.",
                        "I don't know what pregnant women should eat."
                    ],
                    'scores': [5, 5, 3, 2, 1]
                },
                'kin': {
                    'reference': "Abagore batwite bagomba kurya ibiryo bitandukanye birimo imbuto n'imboga, ibiryo by'ubwoko nka ibishyimbo n'inyama, ibiryo by'ubworozi, no gufata ibikoresho nka folic acid.",
                    'variations': [
                        "Abagore bafite inda bakwiye kurya imbuto n'imboga, ibishyimbo, inyama, ibiryo by'ubworozi no gufata folic acid kugira ngo umwana abe afite ubuzima.",
                        "Abagore batwite bagomba kurya ibiryo bitandukanye: imbuto, imboga, ibiryo by'ubwoko, ibiryo by'ubworozi, n'ibikoresho bya folic acid.",
                        "Rya imbuto n'imboga iyo ufite inda.",
                        "Abagore batwite bagomba kurya ibiryo.",
                        "Sinzi ibyo abagore batwite bagomba kurya."
                    ],
                    'scores': [5, 5, 3, 2, 1]
                }
            },
            'dehydration_signs': {
                'en': {
                    'reference': "Signs of dehydration in children include dry mouth, sunken eyes, reduced urination, and lethargy.",
                    'variations': [
                        "Children show dehydration through dry mouth, eyes that look sunken, less urine, and being very tired.",
                        "Dehydration symptoms in children are dry mouth, sunken eyes, decreased urination, and extreme tiredness.",
                        "Children are thirsty and tired when dehydrated.",
                        "Dehydrated children look sick.",
                        "I don't know the signs of dehydration."
                    ],
                    'scores': [5, 5, 3, 2, 1]
                },
                'kin': {
                    'reference': "Ibimenyetso byo kutagira amazi mu mubiri by'abana ni umunwa utose, amaso yatse, kudashobora kw'ijoro, no kuba banyoroshye.",
                    'variations': [
                        "Abana bwerekana ko batagira amazi mu mubiri binyuze mu munwa utose, amaso yatse, kudashobora kw'ijoro, no kuba banyoroshye cyane.",
                        "Ibimenyetso byo kutagira amazi mu mubiri by'abana ni umunwa utose, amaso yatse, kudashobora kw'ijoro, no kuba banyoroshye cyane.",
                        "Abana banyota kandi banyoroshye iyo batagira amazi mu mubiri.",
                        "Abana batagira amazi mu mubiri bararwaye.",
                        "Sinzi ibimenyetso byo kutagira amazi mu mubiri."
                    ],
                    'scores': [5, 5, 3, 2, 1]
                }
            },
            'hand_hygiene': {
                'en': {
                    'reference': "Proper hand hygiene involves washing hands with soap and clean water for at least 20 seconds, especially before eating and after using the toilet.",
                    'variations': [
                        "Good hand washing means using soap and clean water for 20 seconds, before meals and after bathroom use.",
                        "Hand hygiene requires washing with soap and water for 20+ seconds, particularly before eating and after toilet use.",
                        "Wash hands with soap before eating.",
                        "Clean hands with water.",
                        "I don't know how to wash hands properly."
                    ],
                    'scores': [5, 5, 3, 2, 1]
                },
                'kin': {
                    'reference': "Gukaraba amaboko neza ni ukubyongera amaboko doreye n'isabuni n'amazi meza amasegonda 20, cyane cyane mbere yo kurya n'inyuma yo gukoresha ubwiherero.",
                    'variations': [
                        "Gukaraba amaboko neza ni ukubyongera isabuni n'amazi meza amasegonda 20, mbere yo kurya n'inyuma yo gukoresha ubwiherero.",
                        "Gukaraba amaboko neza ni ukubyongera isabuni n'amazi amasegonda 20+, cyane cyane mbere yo kurya n'inyuma yo gukoresha ubwiherero.",
                        "Karaba amaboko doreye n'isabuni mbere yo kurya.",
                        "Sukura amaboko doreye n'amazi.",
                        "Sinzi uko karaba amaboko neza."
                    ],
                    'scores': [5, 5, 3, 2, 1]
                }
            },
            'vaccination_importance': {
                'en': {
                    'reference': "Vaccination is important because it protects children from serious diseases, prevents outbreaks, and saves lives by building immunity.",
                    'variations': [
                        "Vaccines are crucial as they protect children from dangerous diseases, stop disease spread, and save lives through immunity.",
                        "Getting vaccinated is important to protect kids from serious illnesses, prevent epidemics, and save lives by creating immunity.",
                        "Vaccines protect children from diseases.",
                        "Vaccination helps children.",
                        "I don't know why vaccination is important."
                    ],
                    'scores': [5, 5, 3, 2, 1]
                },
                'kin': {
                    'reference': "Ubwishingizi ni ngombwa kuko burinda abana indwara zikomeye, burinda indwara kubyongera, kandi buraba ubuzima dushyiraho ubwishingizi.",
                    'variations': [
                        "Ubwishingizi ni ngombwa kuko burinda abana indwara zikomeye, burinda indwara kubyongera, kandi buraba ubuzima dushyiraho ubwishingizi.",
                        "Gufata ubwishingizi ni ngombwa kurinda abana indwara zikomeye, kurinda indwara kubyongera, no kuraba ubuzima dushyiraho ubwishingizi.",
                        "Ubwishingizi burinda abana indwara.",
                        "Ubwishingizi bufasha abana.",
                        "Sinzi kuki ubwishingizi ari ngombwa."
                    ],
                    'scores': [5, 5, 3, 2, 1]
                }
            }
        }

    def generate_training_data(self, num_samples_per_topic: int = 10) -> List[Dict[str, Any]]:
        """Generate comprehensive training data"""
        training_data = []
        
        for topic_name, topic_data in self.health_topics.items():
            for language in ['en', 'kin']:
                if language in topic_data:
                    lang_data = topic_data[language]
                    reference = lang_data['reference']
                    variations = lang_data['variations']
                    scores = lang_data['scores']
                    
                    # Generate samples for this topic and language
                    for _ in range(num_samples_per_topic):
                        # Randomly select a variation and its corresponding score
                        idx = random.randint(0, len(variations) - 1)
                        student_answer = variations[idx]
                        score = scores[idx]
                        
                        training_data.append({
                            'reference_answer': reference,
                            'student_answer': student_answer,
                            'score': score,
                            'language': language,
                            'topic': topic_name
                        })
        
        return training_data

    def generate_diverse_training_data(self, num_samples: int = 100) -> List[Dict[str, Any]]:
        """Generate more diverse training data with variations"""
        training_data = []
        
        # Base training data
        base_data = self.generate_training_data(num_samples_per_topic=5)
        training_data.extend(base_data)
        
        # Generate additional variations
        for topic_name, topic_data in self.health_topics.items():
            for language in ['en', 'kin']:
                if language in topic_data:
                    lang_data = topic_data[language]
                    reference = lang_data['reference']
                    
                    # Create partial answers (score 2-3)
                    words = reference.split()
                    for i in range(3):
                        partial_length = random.randint(len(words) // 3, len(words) // 2)
                        partial_answer = ' '.join(words[:partial_length])
                        training_data.append({
                            'reference_answer': reference,
                            'student_answer': partial_answer,
                            'score': random.randint(2, 3),
                            'language': language,
                            'topic': topic_name
                        })
                    
                    # Create off-topic answers (score 1)
                    off_topic_responses = [
                        "I don't know the answer to this question.",
                        "This is not related to what I learned.",
                        "I need to study more about this topic."
                    ] if language == 'en' else [
                        "Sinzi igisubizo cy'iki kibazo.",
                        "Ibi ntibihuye n'ibyo nizeye.",
                        "Nkeneye kwiga kurushaho kuri iki kigize."
                    ]
                    
                    for off_topic in off_topic_responses:
                        training_data.append({
                            'reference_answer': reference,
                            'student_answer': off_topic,
                            'score': 1,
                            'language': language,
                            'topic': topic_name
                        })
        
        # Shuffle the data
        random.shuffle(training_data)
        
        # Limit to requested number of samples
        return training_data[:num_samples]

    def save_training_data(self, data: List[Dict[str, Any]], filepath: str):
        """Save training data to JSON file"""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"Training data saved to {filepath}")

    def load_training_data(self, filepath: str) -> List[Dict[str, Any]]:
        """Load training data from JSON file"""
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)

    def create_test_data(self, num_samples: int = 20) -> List[Dict[str, Any]]:
        """Create test data for model evaluation"""
        test_data = []
        
        # Create some edge cases and challenging examples
        edge_cases = [
            {
                'reference': "Malaria is transmitted by mosquitoes.",
                'student': "Malaria comes from mosquitoes.",
                'language': 'en',
                'expected_score': 5
            },
            {
                'reference': "Wash hands before eating.",
                'student': "Clean hands before food.",
                'language': 'en',
                'expected_score': 4
            },
            {
                'reference': "Vaccines prevent diseases.",
                'student': "I don't know about vaccines.",
                'language': 'en',
                'expected_score': 1
            },
            {
                'reference': "Malariya iturwa n'inzige.",
                'student': "Malariya iturwa n'inzige.",
                'language': 'kin',
                'expected_score': 5
            }
        ]
        
        for case in edge_cases:
            test_data.append({
                'reference_answer': case['reference'],
                'student_answer': case['student'],
                'score': case['expected_score'],
                'language': case['language']
            })
        
        # Add some random samples from training data
        training_samples = self.generate_training_data(num_samples_per_topic=2)
        test_data.extend(training_samples[:num_samples - len(edge_cases)])
        
        return test_data

def main():
    """Generate and save training data"""
    generator = TrainingDataGenerator()
    
    # Generate comprehensive training data
    print("Generating training data...")
    training_data = generator.generate_diverse_training_data(num_samples=200)
    
    # Save training data
    generator.save_training_data(training_data, 'training_data.json')
    
    # Generate test data
    print("Generating test data...")
    test_data = generator.create_test_data(num_samples=30)
    generator.save_training_data(test_data, 'test_data.json')
    
    # Print statistics
    print(f"\nGenerated {len(training_data)} training samples")
    print(f"Generated {len(test_data)} test samples")
    
    # Language distribution
    en_count = sum(1 for item in training_data if item['language'] == 'en')
    kin_count = sum(1 for item in training_data if item['language'] == 'kin')
    print(f"English samples: {en_count}")
    print(f"Kinyarwanda samples: {kin_count}")
    
    # Score distribution
    score_counts = {}
    for item in training_data:
        score = item['score']
        score_counts[score] = score_counts.get(score, 0) + 1
    
    print("\nScore distribution:")
    for score in sorted(score_counts.keys()):
        print(f"Score {score}: {score_counts[score]} samples")

if __name__ == "__main__":
    main()
