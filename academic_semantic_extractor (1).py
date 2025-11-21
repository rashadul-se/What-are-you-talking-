import streamlit as st
import pandas as pd
import json
from typing import List, Dict
from collections import defaultdict
import re

try:
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


class TextProcessor:
    """Base class for text processing"""
    
    def process(self, text: str) -> Dict:
        """Process text and return results"""
        raise NotImplementedError


class NERProcessor(TextProcessor):
    """Named Entity Recognition processor"""
    
    def __init__(self):
        try:
            self.nlp_ner = pipeline("ner", model="dslim/bert-base-NER")
        except Exception as e:
            st.error(f"Error loading NER model: {e}")
            self.nlp_ner = None
    
    def process(self, text: str) -> Dict:
        """Extract named entities from text"""
        if not self.nlp_ner:
            return {"error": "NER model not loaded"}
        
        try:
            entities = self.nlp_ner(text)
            aggregated = defaultdict(list)
            current_entity = ""
            current_label = ""
            current_score = 0
            
            for entity in entities:
                token = entity['word'].replace('##', '')
                label = entity['entity'].replace('B-', '').replace('I-', '')
                score = entity['score']
                
                if score > 0.80:
                    if entity['entity'].startswith('B-'):
                        if current_entity:
                            aggregated[current_label].append({
                                'text': current_entity,
                                'score': current_score
                            })
                        current_entity = token
                        current_label = label
                        current_score = score
                    elif entity['entity'].startswith('I-'):
                        current_entity += " " + token
                        current_score = (current_score + score) / 2
            
            if current_entity:
                aggregated[current_label].append({
                    'text': current_entity,
                    'score': current_score
                })
            
            return dict(aggregated)
        except Exception as e:
            return {"error": str(e)}


class ZeroShotClassifier(TextProcessor):
    """Zero-shot classification processor"""
    
    def __init__(self):
        try:
            self.classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
        except Exception as e:
            st.error(f"Error loading zero-shot model: {e}")
            self.classifier = None
    
    def process(self, text: str, candidates: List[str], 
                hypothesis_template: str = "This text mentions {}.") -> Dict:
        """Classify text against candidates"""
        if not self.classifier:
            return {"error": "Classifier not loaded"}
        
        try:
            result = self.classifier(text, candidates, hypothesis_template=hypothesis_template)
            return {
                'labels': result['labels'],
                'scores': result['scores'],
            }
        except Exception as e:
            return {"error": str(e)}


class FacultyExtractor:
    """Extract faculty information"""
    
    def __init__(self, ner_processor: NERProcessor, classifier: ZeroShotClassifier):
        self.ner_processor = ner_processor
        self.classifier = classifier
        self.faculty_roles = [
            "Professor", "Associate Professor", "Assistant Professor", "Lecturer",
            "Instructor", "Faculty Member", "Academic", "Researcher", "Educator"
        ]
    
    def extract(self, text: str) -> List[Dict]:
        """Extract faculty information"""
        ner_results = self.ner_processor.process(text)
        faculty_list = []
        
        if 'error' in ner_results or 'PER' not in ner_results:
            return faculty_list
        
        for person in ner_results['PER']:
            sentences = re.split(r'[.!?]+', text)
            person_context = None
            
            for sentence in sentences:
                if person['text'].lower() in sentence.lower():
                    person_context = sentence.strip()
                    break
            
            if person_context:
                role_result = self.classifier.process(
                    person_context,
                    self.faculty_roles,
                    hypothesis_template="This person is a {}."
                )
                
                faculty_dict = {
                    'name': person['text'],
                    'ner_confidence': float(person['score']),
                    'role': role_result['labels'][0] if 'error' not in role_result else 'Faculty',
                    'role_confidence': float(role_result['scores'][0]) if 'error' not in role_result else 0.0,
                    'context': person_context[:150]
                }
                faculty_list.append(faculty_dict)
        
        return faculty_list


class DepartmentExtractor:
    """Extract department information"""
    
    def __init__(self, classifier: ZeroShotClassifier):
        self.classifier = classifier
        self.department_candidates = [
            "Business", "Engineering", "Science", "Arts", "Medicine", "Law",
            "Education", "Computing", "Mathematics", "Physics", "Chemistry",
            "Literature", "History", "Economics", "Psychology", "Sociology",
            "Management", "Organizational Studies", "Strategy", "Leadership",
            "Finance", "Accounting", "Marketing", "Technology"
        ]
    
    def extract(self, text: str) -> List[Dict]:
        """Extract departments from text"""
        departments = []
        sentences = re.split(r'[.!?]+', text)
        
        for sentence in sentences:
            if len(sentence.strip()) < 10:
                continue
            
            result = self.classifier.process(
                sentence.strip(),
                self.department_candidates,
                hypothesis_template="This sentence discusses the {} department or faculty area."
            )
            
            if 'error' not in result and result['scores'][0] > 0.35:
                departments.append({
                    'name': result['labels'][0],
                    'confidence': float(result['scores'][0]),
                    'source_text': sentence.strip()[:150]
                })
        
        # Remove duplicates, keep highest confidence
        unique_depts = {}
        for dept in departments:
            if dept['name'] not in unique_depts:
                unique_depts[dept['name']] = dept
            elif dept['confidence'] > unique_depts[dept['name']]['confidence']:
                unique_depts[dept['name']] = dept
        
        return list(unique_depts.values())


class SubjectExtractor:
    """Extract subject and course information"""
    
    def __init__(self, classifier: ZeroShotClassifier):
        self.classifier = classifier
        self.subject_candidates = [
            "Strategic Management", "Change Management", "Organizational Theory",
            "Leadership", "Organizational Behavior", "Organizational Design",
            "Organizational Transformation", "Decision Making", "Organizational Strategy",
            "Systems Thinking", "Complex Systems", "Organizational Patterns",
            "Finance", "Marketing", "Data Science", "Machine Learning", "Accounting"
        ]
    
    def extract(self, text: str) -> List[Dict]:
        """Extract subjects from text"""
        subjects = []
        sentences = re.split(r'[.!?]+', text)
        
        # Extract bullet points and special topics
        bullet_points = re.findall(r'[•*\-]\s+(.+?)(?=\n|$)', text)
        model_patterns = re.findall(r'(?:Model|Framework|Approach|Concept|Course|Subject)[\s:]+([^.\n]+)', text, re.IGNORECASE)
        
        all_sentences = sentences + bullet_points + model_patterns
        
        for sentence in all_sentences:
            if len(sentence.strip()) < 8:
                continue
            
            result = self.classifier.process(
                sentence.strip(),
                self.subject_candidates,
                hypothesis_template="This is a topic in {}."
            )
            
            if 'error' not in result and result['scores'][0] > 0.30:
                subjects.append({
                    'topic': sentence.strip()[:150],
                    'category': result['labels'][0],
                    'confidence': float(result['scores'][0])
                })
        
        # Remove duplicates
        unique_subjects = {}
        for subj in subjects:
            key = subj['topic'].lower()
            if key not in unique_subjects:
                unique_subjects[key] = subj
            elif subj['confidence'] > unique_subjects[key]['confidence']:
                unique_subjects[key] = subj
        
        return list(unique_subjects.values())


class AcademicExtractor:
    """Main extractor orchestrator using composition"""
    
    def __init__(self):
        self.ner_processor = NERProcessor()
        self.classifier = ZeroShotClassifier()
        self.faculty_extractor = FacultyExtractor(self.ner_processor, self.classifier)
        self.department_extractor = DepartmentExtractor(self.classifier)
        self.subject_extractor = SubjectExtractor(self.classifier)
    
    def extract_all(self, text: str) -> Dict:
        """Extract all information from text"""
        return {
            'faculty': self.faculty_extractor.extract(text),
            'departments': self.department_extractor.extract(text),
            'subjects': self.subject_extractor.extract(text),
            'text_length': len(text)
        }


class ResultsFormatter:
    """Format extraction results for display"""
    
    @staticmethod
    def format_faculty_table(faculty: List[Dict]) -> pd.DataFrame:
        """Format faculty as table"""
        if not faculty:
            return pd.DataFrame()
        
        df = pd.DataFrame([
            {
                'Name': f['name'],
                'Role': f['role'],
                'NER Confidence': f"{f['ner_confidence']:.2%}",
                'Role Confidence': f"{f['role_confidence']:.2%}",
                'Context': f['context'][:60]
            }
            for f in faculty
        ])
        return df
    
    @staticmethod
    def format_department_table(departments: List[Dict]) -> pd.DataFrame:
        """Format departments as table"""
        if not departments:
            return pd.DataFrame()
        
        df = pd.DataFrame([
            {
                'Department': d['name'],
                'Confidence': f"{d['confidence']:.2%}",
                'Source': d['source_text'][:70]
            }
            for d in departments
        ])
        return df
    
    @staticmethod
    def format_subject_table(subjects: List[Dict]) -> pd.DataFrame:
        """Format subjects as table"""
        if not subjects:
            return pd.DataFrame()
        
        # Sort by confidence
        sorted_subjects = sorted(subjects, key=lambda x: x['confidence'], reverse=True)
        
        df = pd.DataFrame([
            {
                'Topic': s['topic'][:60],
                'Category': s['category'],
                'Confidence': f"{s['confidence']:.2%}"
            }
            for s in sorted_subjects
        ])
        return df


# Streamlit App
def main():
    st.set_page_config(
        page_title="Academic Information Extractor",
        page_icon="📚",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("📚 Academic Information Extractor")
    st.markdown("---")
    
    if not TRANSFORMERS_AVAILABLE:
        st.error("⚠️ Transformers library not installed. Install with: `pip install transformers torch sentencepiece`")
        return
    
    # Initialize session state
    if 'extractor' not in st.session_state:
        with st.spinner("Loading models..."):
            st.session_state.extractor = AcademicExtractor()
            st.session_state.formatter = ResultsFormatter()
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        extraction_mode = st.radio(
            "Choose Input Method:",
            ["Paste Article Text", "Upload Text File", "Sample Article"]
        )
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("📝 Input Text")
        
        if extraction_mode == "Paste Article Text":
            article_text = st.text_area(
                "Paste your article or academic text here:",
                height=250,
                placeholder="Enter academic text containing faculty, departments, and subjects..."
            )
        
        elif extraction_mode == "Upload Text File":
            uploaded_file = st.file_uploader("Upload a text file", type=['txt'])
            if uploaded_file:
                article_text = uploaded_file.read().decode('utf-8')
                st.text_area("File content:", value=article_text, height=250, disabled=True)
            else:
                article_text = ""
        
        else:  # Sample Article
            sample_articles = {
                "Sample 1 - Business Management": """Dr. James Smith and Professor Maria Garcia are faculty members in the Faculty of Business and Management Studies. They specialize in Strategic Management and Organizational Transformation. Their research covers the Octopus Organization Model, Change Management, and Distributed Decision-Making frameworks. They teach advanced courses on organizational design, sensing change in real time, recognizing complex systems, and identifying organizational antipatterns. The department also offers MBA programs in Finance and Accounting.""",
                
                "Sample 2 - Organizational Studies": """In the Department of Organizational Studies and Psychology, Professor Elena Rodriguez leads research on organizational theory and leadership development. The curriculum emphasizes Strategic Management & Adaptation, organizational behavior analysis, and frameworks for understanding complex systems. Dr. Thomas Wright and Dr. Sarah Chen explore change management strategies and scaling organizational changes. Students engage with real-world case studies in strategy and finance.""",
                
                "Sample 3 - Management & Technology": """The Faculty of Management and Technology offers programs integrating organizational transformation with data science. Professor Michael Chen and Dr. Lisa Anderson teach courses combining organizational design with machine learning applications. Topics include the Octopus Organization Model, distributed decision-making, and data science applications. The program includes specializations in Finance, Marketing, and Strategic Management."""
            }
            
            selected_sample = st.selectbox("Select a sample:", list(sample_articles.keys()))
            article_text = sample_articles[selected_sample]
            st.text_area("Sample article:", value=article_text, height=250, disabled=True)
    
    with col2:
        st.header("📊 Quick Stats")
        if article_text:
            st.metric("Text Length", f"{len(article_text)} chars")
            st.metric("Words", len(article_text.split()))
            st.metric("Sentences", len(re.split(r'[.!?]+', article_text)))
    
    # Extract button
    st.markdown("---")
    
    if st.button("🔍 Extract Information", use_container_width=True, type="primary"):
        if not article_text.strip():
            st.warning("Please enter some text to analyze")
            return
        
        with st.spinner("Extracting information... This may take a moment"):
            results = st.session_state.extractor.extract_all(article_text)
        
        # Display Results
        st.markdown("---")
        st.header("📋 Extraction Results")
        
        # Tabs for different result types
        tab1, tab2, tab3, tab4 = st.tabs(["👥 Faculty", "🏢 Departments", "📖 Subjects", "📊 JSON Export"])
        
        with tab1:
            st.subheader("Faculty Members")
            faculty_df = st.session_state.formatter.format_faculty_table(results['faculty'])
            if not faculty_df.empty:
                st.dataframe(faculty_df, use_container_width=True)
            else:
                st.info("No faculty members extracted")
        
        with tab2:
            st.subheader("Departments")
            dept_df = st.session_state.formatter.format_department_table(results['departments'])
            if not dept_df.empty:
                st.dataframe(dept_df, use_container_width=True)
            else:
                st.info("No departments extracted")
        
        with tab3:
            st.subheader("Subjects & Topics")
            subject_df = st.session_state.formatter.format_subject_table(results['subjects'])
            if not subject_df.empty:
                st.dataframe(subject_df, use_container_width=True)
            else:
                st.info("No subjects extracted")
        
        with tab4:
            st.subheader("JSON Export")
            json_output = json.dumps(results, indent=2, default=str)
            st.json(results)
            st.download_button(
                label="📥 Download JSON",
                data=json_output,
                file_name="extraction_results.json",
                mime="application/json"
            )
        
        # Summary statistics
        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Faculty Found", len(results['faculty']))
        with col2:
            st.metric("Departments Found", len(results['departments']))
        with col3:
            st.metric("Subjects Found", len(results['subjects']))


if __name__ == "__main__":
    main()