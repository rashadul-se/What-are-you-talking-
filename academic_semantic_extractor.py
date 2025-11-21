import streamlit as st
import pandas as pd
import json
from typing import List, Dict
from collections import defaultdict
import re
import os

# Suppress PyTorch warnings
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

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
        self.nlp_ner = None
        try:
            # Use smaller, more stable model
            self.nlp_ner = pipeline(
                "ner",
                model="dslim/bert-base-NER",
                device=-1  # Use CPU to avoid GPU issues
            )
        except Exception as e:
            st.warning(f"NER model initialization: {str(e)[:100]}")
            self.nlp_ner = None
    
    def process(self, text: str) -> Dict:
        """Extract named entities from text"""
        if not self.nlp_ner:
            return {"error": "NER model not loaded"}
        
        try:
            # Truncate text to avoid memory issues
            text_truncated = text[:2000]
            entities = self.nlp_ner(text_truncated)
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
            return {"error": f"NER processing failed: {str(e)[:50]}"}


class ZeroShotClassifier(TextProcessor):
    """Zero-shot classification processor"""
    
    def __init__(self):
        self.classifier = None
        try:
            # Use smaller model for better compatibility
            self.classifier = pipeline(
                "zero-shot-classification",
                model="facebook/bart-large-mnli",
                device=-1  # Use CPU
            )
        except Exception as e:
            st.warning(f"Classifier initialization: {str(e)[:100]}")
            self.classifier = None
    
    def process(self, text: str, candidates: List[str], 
                hypothesis_template: str = "This text mentions {}.") -> Dict:
        """Classify text against candidates"""
        if not self.classifier:
            return {"error": "Classifier not loaded"}
        
        try:
            # Truncate and clean text
            text_clean = text.strip()[:500]
            result = self.classifier(
                text_clean,
                candidates[:10],  # Limit candidates
                hypothesis_template=hypothesis_template,
                multi_class=True
            )
            return {
                'labels': result['labels'],
                'scores': result['scores'],
            }
        except Exception as e:
            return {"error": f"Classification failed: {str(e)[:50]}"}


class FacultyExtractor:
    """Extract faculty information with improved detection"""
    
    def __init__(self, ner_processor: NERProcessor, classifier: ZeroShotClassifier):
        self.ner_processor = ner_processor
        self.classifier = classifier
        self.faculty_roles = [
            "Professor", "Associate Professor", "Assistant Professor", "Lecturer",
            "Instructor", "Faculty Member", "Academic", "Researcher", "Educator",
            "Enterprise Strategist", "Author", "Expert", "Specialist", "Director"
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
                
                if 'error' not in role_result:
                    top_roles = list(zip(
                        role_result['labels'], 
                        role_result['scores']
                    ))[:3]
                    
                    faculty_dict = {
                        'name': person['text'],
                        'ner_confidence': float(person['score']),
                        'primary_role': role_result['labels'][0],
                        'primary_role_confidence': float(role_result['scores'][0]),
                        'alternative_roles': [
                            {'role': role, 'confidence': float(score)} 
                            for role, score in top_roles[1:]
                        ],
                        'context': person_context[:150]
                    }
                    faculty_list.append(faculty_dict)
        
        return faculty_list


class DepartmentExtractor:
    """Extract department and organization information"""
    
    def __init__(self, classifier: ZeroShotClassifier):
        self.classifier = classifier
        self.department_candidates = [
            "Business", "Engineering", "Science", "Arts", "Medicine", "Law",
            "Education", "Computing", "Mathematics", "Physics", "Chemistry",
            "Literature", "History", "Economics", "Psychology", "Sociology",
            "Management", "Strategy", "Leadership", "Finance", "Accounting",
            "Marketing", "Technology", "Information Technology",
            "Enterprise Strategy", "Digital Transformation", "Organizational Studies"
        ]
    
    def extract(self, text: str) -> List[Dict]:
        """Extract departments from text"""
        departments = []
        sentences = re.split(r'[.!?]+', text)
        
        # Extract company/organization names
        org_pattern = r'\b(?:Amazon|Google|Microsoft|Apple|IBM|Facebook|Meta|AWS|Inc\.|Corp\.|Ltd\.|LLC)\b'
        orgs = re.findall(org_pattern, text, re.IGNORECASE)
        for org in orgs:
            departments.append({
                'name': org,
                'type': 'Organization',
                'confidence': 1.0,
                'source_text': text[:150]
            })
        
        for sentence in sentences:
            if len(sentence.strip()) < 10:
                continue
            
            result = self.classifier.process(
                sentence.strip(),
                self.department_candidates,
                hypothesis_template="This sentence discusses the {} department."
            )
            
            if 'error' not in result and result['scores'][0] > 0.25:
                departments.append({
                    'name': result['labels'][0],
                    'type': 'Department/Faculty',
                    'confidence': float(result['scores'][0]),
                    'source_text': sentence.strip()[:150]
                })
        
        # Remove duplicates, keep highest confidence
        unique_depts = {}
        for dept in departments:
            key = dept['name'].lower()
            if key not in unique_depts:
                unique_depts[key] = dept
            elif dept['confidence'] > unique_depts[key]['confidence']:
                unique_depts[key] = dept
        
        return list(unique_depts.values())


class SubjectExtractor:
    """Extract subject, course, and concept information"""
    
    def __init__(self, classifier: ZeroShotClassifier):
        self.classifier = classifier
        self.subject_candidates = [
            "Strategic Management", "Change Management", "Organizational Theory",
            "Leadership", "Organizational Behavior", "Organizational Design",
            "Organizational Transformation", "Decision Making", "Systems Thinking",
            "Complex Systems", "Innovation", "Digital Transformation",
            "Finance", "Marketing", "Data Science", "Machine Learning", "Accounting",
            "Distributed Decision-Making", "Business Transformation", "Adaptability",
            "Enterprise Strategy", "Organizational Agility"
        ]
    
    def extract(self, text: str) -> List[Dict]:
        """Extract subjects from text"""
        subjects = []
        sentences = re.split(r'[.!?]+', text)
        
        # Extract patterns
        bullet_points = re.findall(r'[•*\-]\s+(.+?)(?=\n|$)', text)
        
        model_patterns = re.findall(
            r'(?:Model|Framework|Approach|Concept|Course|Subject|Organization)[\s:]+([^.\n]+)',
            text, re.IGNORECASE
        )
        
        all_sentences = sentences + bullet_points + model_patterns
        
        for sentence in all_sentences:
            if len(sentence.strip()) < 8:
                continue
            
            result = self.classifier.process(
                sentence.strip(),
                self.subject_candidates,
                hypothesis_template="This is about {}."
            )
            
            if 'error' not in result and result['scores'][0] > 0.20:
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
    """Main extractor orchestrator"""
    
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
                'Primary Role': f['primary_role'],
                'Role Confidence': f"{f['primary_role_confidence']:.1%}",
                'NER Confidence': f"{f['ner_confidence']:.1%}",
                'Context': f['context'][:50]
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
                'Name': d['name'],
                'Type': d.get('type', 'Department'),
                'Confidence': f"{d['confidence']:.1%}",
                'Source': d['source_text'][:60]
            }
            for d in departments
        ])
        return df
    
    @staticmethod
    def format_subject_table(subjects: List[Dict]) -> pd.DataFrame:
        """Format subjects as table"""
        if not subjects:
            return pd.DataFrame()
        
        sorted_subjects = sorted(subjects, key=lambda x: x['confidence'], reverse=True)
        
        df = pd.DataFrame([
            {
                'Topic': s['topic'][:60],
                'Category': s['category'],
                'Confidence': f"{s['confidence']:.1%}"
            }
            for s in sorted_subjects
        ])
        return df


def main():
    st.set_page_config(
        page_title="Academic Information Extractor",
        page_icon="📚",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("📚 Academic Information Extractor (Fixed)")
    st.markdown("---")
    
    if not TRANSFORMERS_AVAILABLE:
        st.error("⚠️ Transformers library not installed. Install with: `pip install transformers torch`")
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
        
        st.markdown("---")
        st.subheader("✅ Fixes Applied")
        st.markdown("""
        - ✓ CPU-only mode (no GPU errors)
        - ✓ Removed sentencepiece dependency
        - ✓ Python 3.12 compatible
        - ✓ Better error handling
        - ✓ Text truncation for stability
        - ✓ Lightweight models
        """)
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("📝 Input Text")
        
        if extraction_mode == "Paste Article Text":
            article_text = st.text_area(
                "Paste your article or academic text here:",
                height=250,
                placeholder="Enter academic text..."
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
                "Sample 1 - Octopus Organization": """As companies pour trillions into transformation efforts, few see lasting results. That's because most organizations approach change like machines—rigidly, predictably, and from the top down, argue Amazon Web Services enterprise strategists Jana Werner and Phil Le-Brun. The Octopus Organization model distributes decision-making, senses change in real time, and continually adapts. Key topics: Strategic Management, Change Management, Organizational Transformation, Digital Transformation.""",
                
                "Sample 2 - Business Management": """Dr. James Smith and Professor Maria Garcia are faculty members in the Faculty of Business. They specialize in Strategic Management and Organizational Theory. Their research covers change management frameworks. Departments: Business, Management, Finance, Accounting. Topics: Leadership, Organizational Design.""",
            }
            
            selected_sample = st.selectbox("Select a sample:", list(sample_articles.keys()))
            article_text = sample_articles[selected_sample]
            st.text_area("Sample article:", value=article_text, height=250, disabled=True)
    
    with col2:
        st.header("📊 Stats")
        if article_text:
            st.metric("Text Length", f"{len(article_text)} chars")
            st.metric("Words", len(article_text.split()))
    
    st.markdown("---")
    
    if st.button("🔍 Extract Information", use_container_width=True, type="primary"):
        if not article_text.strip():
            st.warning("Please enter some text")
            return
        
        with st.spinner("Processing..."):
            results = st.session_state.extractor.extract_all(article_text)
        
        st.markdown("---")
        st.header("📋 Results")
        
        tab1, tab2, tab3, tab4 = st.tabs(["👥 Faculty", "🏢 Departments", "📖 Subjects", "📊 JSON"])
        
        with tab1:
            faculty_df = st.session_state.formatter.format_faculty_table(results['faculty'])
            if not faculty_df.empty:
                st.dataframe(faculty_df, use_container_width=True)
            else:
                st.info("No faculty extracted")
        
        with tab2:
            dept_df = st.session_state.formatter.format_department_table(results['departments'])
            if not dept_df.empty:
                st.dataframe(dept_df, use_container_width=True)
            else:
                st.info("No departments extracted")
        
        with tab3:
            subject_df = st.session_state.formatter.format_subject_table(results['subjects'])
            if not subject_df.empty:
                st.dataframe(subject_df, use_container_width=True)
            else:
                st.info("No subjects extracted")
        
        with tab4:
            json_output = json.dumps(results, indent=2, default=str)
            st.json(results)
            st.download_button(
                label="📥 Download JSON",
                data=json_output,
                file_name="results.json",
                mime="application/json"
            )
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Faculty Found", len(results['faculty']))
        with col2:
            st.metric("Departments Found", len(results['departments']))
        with col3:
            st.metric("Subjects Found", len(results['subjects']))


if __name__ == "__main__":
    main()
