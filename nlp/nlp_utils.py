"""
NLP utility functions for document processing
"""
import re
import logging
from typing import List, Dict, Any, Optional, Tuple
from collections import Counter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def extract_keywords(text: str, top_k: int = 10) -> List[Tuple[str, int]]:
    """
    Extract keywords from text using simple frequency analysis
    
    Args:
        text (str): Input text
        top_k (int): Number of top keywords to return
        
    Returns:
        list: List of (keyword, count) tuples
    """
    if not text:
        return []
    
    # Convert to lowercase and extract words
    words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
    
    # Common stop words to filter out
    stop_words = {
        'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with',
        'by', 'from', 'up', 'about', 'into', 'through', 'during', 'before',
        'after', 'above', 'below', 'over', 'under', 'again', 'further', 'then',
        'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any',
        'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'only',
        'own', 'same', 'than', 'too', 'very', 'can', 'will', 'just', 'should',
        'now', 'has', 'have', 'had', 'been', 'being', 'was', 'were', 'are',
        'this', 'that', 'these', 'those', 'they', 'them', 'their', 'what',
        'which', 'who', 'whom', 'whose', 'would', 'could', 'should', 'may',
        'might', 'must', 'shall', 'will', 'can', 'said', 'says', 'get', 'got'
    }
    
    # Filter out stop words and count
    filtered_words = [word for word in words if word not in stop_words and len(word) > 3]
    word_counts = Counter(filtered_words)
    
    return word_counts.most_common(top_k)

def calculate_text_complexity(text: str) -> Dict[str, Any]:
    """
    Calculate various text complexity metrics
    
    Args:
        text (str): Input text
        
    Returns:
        dict: Complexity metrics
    """
    if not text:
        return {}
    
    # Basic counts
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    words = re.findall(r'\b\w+\b', text)
    characters = len(text.replace(' ', ''))
    
    # Calculate metrics
    avg_sentence_length = len(words) / len(sentences) if sentences else 0
    avg_word_length = sum(len(word) for word in words) / len(words) if words else 0
    
    # Syllable count (simplified)
    def count_syllables(word):
        vowels = 'aeiouy'
        count = 0
        prev_char_was_vowel = False
        for char in word.lower():
            if char in vowels:
                if not prev_char_was_vowel:
                    count += 1
                prev_char_was_vowel = True
            else:
                prev_char_was_vowel = False
        return max(count, 1)  # At least 1 syllable
    
    total_syllables = sum(count_syllables(word) for word in words)
    
    # Flesch Reading Ease (simplified)
    if len(sentences) > 0 and len(words) > 0:
        flesch_score = 206.835 - (1.015 * avg_sentence_length) - (84.6 * (total_syllables / len(words)))
    else:
        flesch_score = 0
    
    return {
        'sentence_count': len(sentences),
        'word_count': len(words),
        'character_count': characters,
        'avg_sentence_length': round(avg_sentence_length, 2),
        'avg_word_length': round(avg_word_length, 2),
        'flesch_reading_ease': round(flesch_score, 2),
        'complexity_level': 'easy' if flesch_score > 80 else 'moderate' if flesch_score > 50 else 'difficult'
    }

def extract_entities(text: str) -> Dict[str, List[str]]:
    """
    Extract basic entities from text using regex patterns
    
    Args:
        text (str): Input text
        
    Returns:
        dict: Extracted entities by category
    """
    entities = {
        'dates': [],
        'numbers': [],
        'percentages': [],
        'organizations': [],
        'locations': [],
        'technical_terms': []
    }
    
    if not text:
        return entities
    
    # Date patterns
    date_patterns = [
        r'\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b',
        r'\b\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{2,4}\b',
        r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{2,4}\b'
    ]
    
    for pattern in date_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        entities['dates'].extend(matches)
    
    # Numbers and percentages
    entities['numbers'] = re.findall(r'\b\d+(?:\.\d+)?\b', text)
    entities['percentages'] = re.findall(r'\b\d+(?:\.\d+)?%\b', text)
    
    # Organizations (basic patterns)
    org_patterns = [
        r'\b[A-Z][A-Z0-9&\s]{2,30}(?:Ltd|Inc|Corp|Company|Department|Ministry|Authority|Commission|Board|Agency|Organization)\b',
        r'\b(?:KMRL|Indian Railways|Metro|Corporation|Department|Ministry)\b'
    ]
    
    for pattern in org_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        entities['organizations'].extend(matches)
    
    # Technical terms (engineering, railway terms)
    tech_terms = [
        'maintenance', 'inspection', 'safety', 'protocol', 'procedure', 'manual',
        'technical', 'engineering', 'electrical', 'mechanical', 'civil', 'signal',
        'telecommunication', 'rolling stock', 'infrastructure', 'operations',
        'compliance', 'audit', 'assessment', 'specification', 'standard'
    ]
    
    for term in tech_terms:
        if term.lower() in text.lower():
            entities['technical_terms'].append(term)
    
    # Remove duplicates
    for key in entities:
        entities[key] = list(set(entities[key]))
    
    return entities

def segment_document(text: str) -> List[Dict[str, str]]:
    """
    Segment document into logical sections
    
    Args:
        text (str): Input text
        
    Returns:
        list: List of sections with type and content
    """
    if not text:
        return []
    
    sections = []
    lines = text.split('\n')
    current_section = {'type': 'content', 'title': '', 'content': []}
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # Detect section headers
        is_header = False
        header_type = 'content'
        
        # Check for various header patterns
        if re.match(r'^\d+\.\s+[A-Z]', line):  # Numbered sections
            is_header = True
            header_type = 'numbered_section'
        elif re.match(r'^[A-Z][A-Z\s]{5,50}$', line):  # All caps headers
            is_header = True
            header_type = 'major_section'
        elif line.endswith(':') and len(line) < 100:  # Lines ending with colon
            is_header = True
            header_type = 'subsection'
        elif re.match(r'^(?:INTRODUCTION|OVERVIEW|SUMMARY|CONCLUSION|APPENDIX|REFERENCES)', line, re.IGNORECASE):
            is_header = True
            header_type = 'document_section'
        
        if is_header and current_section['content']:
            # Save current section
            sections.append({
                'type': current_section['type'],
                'title': current_section['title'],
                'content': '\n'.join(current_section['content'])
            })
            
            # Start new section
            current_section = {
                'type': header_type,
                'title': line,
                'content': []
            }
        else:
            current_section['content'].append(line)
    
    # Add final section
    if current_section['content']:
        sections.append({
            'type': current_section['type'],
            'title': current_section['title'] or 'Content',
            'content': '\n'.join(current_section['content'])
        })
    
    return sections

def calculate_document_similarity(text1: str, text2: str) -> float:
    """
    Calculate similarity between two documents using simple word overlap
    
    Args:
        text1 (str): First document
        text2 (str): Second document
        
    Returns:
        float: Similarity score (0-1)
    """
    if not text1 or not text2:
        return 0.0
    
    # Extract words
    words1 = set(re.findall(r'\b\w{3,}\b', text1.lower()))
    words2 = set(re.findall(r'\b\w{3,}\b', text2.lower()))
    
    if not words1 or not words2:
        return 0.0
    
    # Calculate Jaccard similarity
    intersection = len(words1.intersection(words2))
    union = len(words1.union(words2))
    
    return intersection / union if union > 0 else 0.0