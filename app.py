# Main Flask application for OCR-NLP document processing
import os
import sys
import uuid
import logging
from pathlib import Path
from flask import Flask, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from werkzeug.exceptions import BadRequest, RequestEntityTooLarge

# Add project root to Python path
ROOT_DIR = Path(__file__).parent.absolute()
sys.path.append(str(ROOT_DIR))

# Import our modules
from config import *
from utils.file_converter import FileConverter, download_file_from_url, validate_uploaded_file
from utils.text_cleaner import clean_document_text, extract_text_metadata
from ocr.ocr_utils import extract_text_from_file
from nlp.inference import process_document_text

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create Flask app
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

# Initialize components
file_converter = FileConverter()

class DocumentProcessor:
    """Main document processing pipeline"""
    
    def __init__(self):
        self.file_converter = FileConverter()
    
    def process_document(self, file_path: str, original_filename: str = "") -> dict:
        """
        Complete document processing pipeline: OCR -> Clean -> NLP
        
        Args:
            file_path (str): Path to document file
            original_filename (str): Original filename for display
            
        Returns:
            dict: Processing results
        """
        try:
            # Step 1: Extract text using OCR
            logger.info(f"🔍 Starting OCR for: {file_path}")
            ocr_result = extract_text_from_file(file_path)
            
            if not ocr_result['success'] or not ocr_result['text']:
                return {
                    'success': False,
                    'error': 'Failed to extract text from document'
                }
            
            extracted_text = ocr_result['text']
            logger.info(f"📝 Extracted {len(extracted_text)} characters")
            
            # Step 2: Clean text
            logger.info("🧹 Cleaning extracted text")
            cleaned_text = clean_document_text(extracted_text, remove_metadata=False)
            
            if len(cleaned_text.strip()) < 10:
                return {
                    'success': False,
                    'error': 'Document text is too short after cleaning'
                }
            
            # Step 3: NLP processing (classification + summarization)
            logger.info("🤖 Starting NLP processing")
            nlp_result = process_document_text(cleaned_text, original_filename or Path(file_path).name)
            
            if not nlp_result['success']:
                return {
                    'success': False,
                    'error': 'NLP processing failed'
                }
            
            # Step 4: Combine results
            result = {
                'success': True,
                'doc_type': nlp_result['doc_type'],
                'org_type': nlp_result['org_type'],
                'summary': nlp_result['summary'],
                'shortSummary': nlp_result['shortSummary'],
                'metadata': {
                    'text_length': nlp_result['text_length'],
                    'pages': ocr_result['pages'],
                    'confidence': nlp_result['confidence'],
                    'file_info': ocr_result['metadata']
                }
            }
            
            logger.info(f"✅ Complete pipeline finished for: {original_filename}")
            return result
            
        except Exception as e:
            logger.error(f"Document processing failed: {str(e)}")
            return {
                'success': False,
                'error': f"Processing failed: {str(e)}"
            }

# Initialize processor
processor = DocumentProcessor()

@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "ok",
        "service": "OCR-NLP Document Processor",
        "version": "1.0.0"
    })

@app.route("/process_url", methods=["POST"])
def process_url():
    """
    Process document from URL
    
    Expected JSON:
    {
        "url": "https://example.com/document.pdf"
    }
    
    Returns:
    {
        "results": [
            {
                "doc_name": "document.pdf",
                "doc_type": "TBL",
                "org_type": "ENG", 
                "summary": ["point1", "point2"],
                "shortSummary": "Brief summary text"
            }
        ]
    }
    """
    try:
        # Validate request
        if not request.is_json:
            return jsonify({"error": "Request must be JSON"}), 400
        
        data = request.get_json()
        if not data or 'url' not in data:
            return jsonify({"error": "URL is required"}), 400
        
        url = data['url']
        logger.info(f"Processing URL: {url}")
        
        # Download file
        download_result = download_file_from_url(url)
        
        if not download_result['success']:
            return jsonify({"error": download_result['error']}), 400
        
        file_path = download_result['file_path']
        original_filename = download_result['original_filename']
        
        # Validate downloaded file
        validation = validate_uploaded_file(file_path)
        if not validation['valid']:
            # Clean up downloaded file
            try:
                Path(file_path).unlink()
            except:
                pass
            return jsonify({"error": validation['error']}), 400
        
        # Process document
        result = processor.process_document(file_path, original_filename)
        
        # Clean up file after processing
        try:
            Path(file_path).unlink()
        except:
            logger.warning(f"Could not remove temporary file: {file_path}")
        
        if not result['success']:
            return jsonify({"error": result['error']}), 500
        
        # Format response
        response_data = {
            "doc_type": result['doc_type'],
            "org_type": result['org_type'], 
            "summary": result['summary'],
            "shortSummary": result['shortSummary']
        }
        
        return jsonify({"results": [response_data]})
        
    except Exception as e:
        logger.error(f"Error in process_url: {str(e)}")
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500

@app.route("/process", methods=["POST"])
def process_files():
    """
    Process uploaded files
    
    Accepts multipart/form-data with files
    """
    try:
        if 'files' not in request.files:
            return jsonify({"error": "No files uploaded"}), 400
        
        files = request.files.getlist('files')
        if not files or all(f.filename == '' for f in files):
            return jsonify({"error": "No files selected"}), 400
        
        results = []
        
        for file in files:
            if file.filename == '':
                continue
            
            # Secure filename
            original_filename = file.filename
            safe_filename = secure_filename(original_filename)
            unique_filename = f"{uuid.uuid4().hex[:8]}_{safe_filename}"
            
            # Save uploaded file
            file_path = UPLOADS_DIR / unique_filename
            file.save(str(file_path))
            
            try:
                # Validate file
                validation = validate_uploaded_file(str(file_path))
                if not validation['valid']:
                    results.append({
                        "error": validation['error']
                    })
                    continue
                
                # Process document
                result = processor.process_document(str(file_path), original_filename)
                
                if result['success']:
                    results.append({
                        "doc_type": result['doc_type'],
                        "org_type": result['org_type'],
                        "summary": result['summary'],
                        "shortSummary": result['shortSummary']
                    })
                else:
                    results.append({
                        "error": result['error']
                    })
            
            finally:
                # Clean up uploaded file
                try:
                    file_path.unlink()
                except:
                    logger.warning(f"Could not remove uploaded file: {file_path}")
        
        return jsonify({"results": results})
        
    except Exception as e:
        logger.error(f"Error in process_files: {str(e)}")
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500

@app.route("/test", methods=["GET"])
def test_system():
    """Test endpoint to check if all components are working"""
    status = {
        "ocr": "unknown",
        "nlp": "unknown", 
        "file_converter": "unknown",
        "directories": {}
    }
    
    try:
        # Test OCR
        test_text = "This is a test document for KMRL Engineering Department Technical Bulletin"
        ocr_result = process_document_text(test_text, "test.pdf")
        status["ocr"] = "working" if ocr_result['success'] else "error"
        
        # Test NLP  
        status["nlp"] = "working" if ocr_result.get('doc_type') and ocr_result.get('org_type') else "error"
        
        # Test file converter
        file_info = file_converter.get_file_info(__file__)
        status["file_converter"] = "working" if file_info['exists'] else "error"
        
        # Check directories
        for dir_name, dir_path in [("uploads", UPLOADS_DIR), ("data", DATA_DIR)]:
            status["directories"][dir_name] = {
                "exists": dir_path.exists(),
                "path": str(dir_path)
            }
        
        return jsonify({
            "status": "System test completed",
            "components": status,
            "test_classification": {
                "doc_type": ocr_result.get('doc_type'),
                "org_type": ocr_result.get('org_type'),
                "short_summary": ocr_result.get('shortSummary')
            }
        })
        
    except Exception as e:
        return jsonify({
            "status": "System test failed",
            "error": str(e),
            "components": status
        }), 500

@app.errorhandler(413)
def too_large(e):
    return jsonify({"error": "File too large (max 50MB)"}), 413

@app.errorhandler(400)
def bad_request(e):
    return jsonify({"error": "Bad request"}), 400

@app.errorhandler(500)
def internal_error(e):
    return jsonify({"error": "Internal server error"}), 500

if __name__ == "__main__":
    logger.info("Starting OCR-NLP Document Processing Server")
    logger.info(f"Upload directory: {UPLOADS_DIR}")
    
    # Clean up old files on startup
    try:
        removed = file_converter.cleanup_temp_files(older_than_hours=24)
        logger.info(f"Startup cleanup: removed {removed} old files")
    except Exception as e:
        logger.warning(f"Startup cleanup failed: {str(e)}")
    
    app.run(debug=True, host=FLASK_HOST, port=FLASK_PORT)