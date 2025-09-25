import re
import logging
from typing import List, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TextCleaner:
    """Clean and preprocess extracted text for better NLP processing"""
    
    def __init__(self):
        # Common patterns to remove
        self.noise_patterns = [
            r'\x0c',  # Form feed characters
            r'\n{3,}',  # Multiple newlines
            r'\s{3,}',  # Multiple spaces
            r'[^\x00-\x7F]+',  # Non-ASCII characters (optional)
        ]
        
        # Pattern for email addresses
        self.email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        
        # Pattern for URLs
        self.url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        
        # Pattern for phone numbers (basic)
        self.phone_pattern = r'(?:\+?1[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}'
    
    def clean_basic(self, text: str) -> str:
        """
        Basic text cleaning - remove noise, normalize whitespace
        
        Args:
            text (str): Raw text
            
        Returns:
            str: Cleaned text
        """
        if not text:
            return ""
        
        cleaned = text
        
        # Remove form feed and other control characters
        cleaned = re.sub(r'[\x0c\x0b\x0e-\x1f]', '', cleaned)
        
        # Replace multiple newlines with double newline
        cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
        
        # Replace multiple spaces with single space
        cleaned = re.sub(r'[ \t]{2,}', ' ', cleaned)
        
        # Remove trailing/leading whitespace from each line
        lines = cleaned.split('\n')
        lines = [line.strip() for line in lines]
        cleaned = '\n'.join(lines)
        
        # Remove empty lines at start and end
        cleaned = cleaned.strip()
        
        return cleaned
    
    def remove_headers_footers(self, text: str, header_keywords: Optional[List[str]] = None, 
                              footer_keywords: Optional[List[str]] = None) -> str:
        """
        Remove common headers and footers from document text
        
        Args:
            text (str): Input text
            header_keywords (list, optional): Keywords that indicate headers
            footer_keywords (list, optional): Keywords that indicate footers
            
        Returns:
            str: Text with headers/footers removed
        """
        if not text:
            return ""
        
        if header_keywords is None:
            header_keywords = ['page', 'document', 'confidential', 'proprietary', 'draft']
        
        if footer_keywords is None:
            footer_keywords = ['page', 'copyright', '©', 'all rights reserved', 'confidential']
        
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line_lower = line.lower().strip()
            
            # Skip if line contains header/footer keywords and is short
            if len(line_lower) < 100:  # Likely header/footer if short
                is_header_footer = False
                
                for keyword in header_keywords + footer_keywords:
                    if keyword.lower() in line_lower:
                        is_header_footer = True
                        break
                
                # Also skip if line contains page numbers
                if re.search(r'page\s+\d+', line_lower) or re.search(r'\d+\s+of\s+\d+', line_lower):
                    is_header_footer = True
                
                if is_header_footer:
                    continue
            
            cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)
    
    def normalize_text(self, text: str) -> str:
        """
        Normalize text for consistent processing
        
        Args:
            text (str): Input text
            
        Returns:
            str: Normalized text
        """
        if not text:
            return ""
        
        normalized = text
        
        # Convert to lowercase for processing (but preserve original case for output)
        # Fix common OCR errors
        ocr_corrections = {
            'rn': 'm',  # Common OCR error
            '|': 'l',   # Vertical bar to lowercase l
            '0': 'O',   # Zero to capital O in words (context-dependent)
        }
        
        # Apply corrections carefully (only in specific contexts)
        # This is a simplified approach - a more sophisticated method would use context
        
        # Fix spacing around punctuation
        normalized = re.sub(r'\s+([,.!?;:])', r'\1', normalized)
        normalized = re.sub(r'([,.!?;:])\s*', r'\1 ', normalized)
        
        # Fix quotation marks
        normalized = re.sub(r'"\s*', '"', normalized)
        normalized = re.sub(r'\s*"', '"', normalized)
        
        # Normalize dashes
        normalized = re.sub(r'[-–—]{2,}', '—', normalized)
        
        return normalized
    
    def extract_metadata(self, text: str) -> dict:
        """
        Extract metadata from text (emails, urls, phone numbers, etc.)
        
        Args:
            text (str): Input text
            
        Returns:
            dict: Extracted metadata
        """
        metadata = {
            'emails': [],
            'urls': [],
            'phone_numbers': [],
            'dates': []
        }
        
        if not text:
            return metadata
        
        # Extract emails
        emails = re.findall(self.email_pattern, text)
        metadata['emails'] = list(set(emails))  # Remove duplicates
        
        # Extract URLs
        urls = re.findall(self.url_pattern, text)
        metadata['urls'] = list(set(urls))
        
        # Extract phone numbers
        phones = re.findall(self.phone_pattern, text)
        metadata['phone_numbers'] = list(set(phones))
        
        # Extract dates (basic patterns)
        date_patterns = [
            r'\d{1,2}[-/]\d{1,2}[-/]\d{2,4}',  # MM/DD/YYYY or DD/MM/YYYY
            r'\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\w*\s+\d{2,4}',  # 15 Jan 2024
            r'(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\w*\s+\d{1,2},?\s+\d{2,4}'  # January 15, 2024
        ]
        
        dates = []
        for pattern in date_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            dates.extend(matches)
        
        metadata['dates'] = list(set(dates))
        
        return metadata
    
    def clean_for_nlp(self, text: str, remove_metadata: bool = False) -> str:
        """
        Complete cleaning pipeline for NLP processing
        
        Args:
            text (str): Raw text
            remove_metadata (bool): Whether to remove emails, URLs, etc.
            
        Returns:
            str: Cleaned text ready for NLP
        """
        if not text:
            return ""
        
        # Step 1: Basic cleaning
        cleaned = self.clean_basic(text)
        
        # Step 2: Remove headers/footers
        cleaned = self.remove_headers_footers(cleaned)
        
        # Step 3: Remove metadata if requested
        if remove_metadata:
            cleaned = re.sub(self.email_pattern, '[EMAIL]', cleaned)
            cleaned = re.sub(self.url_pattern, '[URL]', cleaned)
            cleaned = re.sub(self.phone_pattern, '[PHONE]', cleaned)
        
        # Step 4: Normalize text
        cleaned = self.normalize_text(cleaned)
        
        # Step 5: Final cleanup
        cleaned = self.clean_basic(cleaned)  # One more pass
        
        return cleaned
    
    def split_into_sections(self, text: str) -> List[dict]:
        """
        Split text into logical sections based on headings and structure
        
        Args:
            text (str): Input text
            
        Returns:
            list: List of sections with titles and content
        """
        if not text:
            return []
        
        sections = []
        lines = text.split('\n')
        current_section = {'title': 'Introduction', 'content': []}
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Simple heuristic for headings (all caps, short lines, numbered sections, etc.)
            is_heading = (
                len(line) < 100 and  # Short lines might be headings
                (line.isupper() or  # All caps
                 re.match(r'^\d+\.', line) or  # Numbered sections
                 re.match(r'^[A-Z][A-Z\s]+$', line) or  # Multiple caps with spaces
                 line.endswith(':'))  # Ends with colon
            )
            
            if is_heading and len(current_section['content']) > 0:
                # Start new section
                sections.append({
                    'title': current_section['title'],
                    'content': '\n'.join(current_section['content'])
                })
                current_section = {'title': line, 'content': []}
            else:
                current_section['content'].append(line)
        
        # Add final section
        if current_section['content']:
            sections.append({
                'title': current_section['title'],
                'content': '\n'.join(current_section['content'])
            })
        
        return sections

def clean_document_text(text: str, remove_metadata: bool = False) -> str:
    """
    Convenience function for text cleaning
    
    Args:
        text (str): Raw text
        remove_metadata (bool): Whether to remove personal data
        
    Returns:
        str: Cleaned text
    """
    cleaner = TextCleaner()
    return cleaner.clean_for_nlp(text, remove_metadata)

def extract_text_metadata(text: str) -> dict:
    """
    Convenience function for metadata extraction
    
    Args:
        text (str): Input text
        
    Returns:
        dict: Extracted metadata
    """
    cleaner = TextCleaner()
    return cleaner.extract_metadata(text)