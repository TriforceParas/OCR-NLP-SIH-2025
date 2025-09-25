import os
import logging
import uuid
from pathlib import Path
from typing import Optional, Dict, Any
import shutil

try:
    from docx import Document
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

from config import UPLOADS_DIR, DOWNLOAD_TIMEOUT, ALLOWED_EXTENSIONS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FileConverter:
    """Handle file downloads, conversions, and management"""
    
    def __init__(self):
        self.uploads_dir = UPLOADS_DIR
        
        # Ensure directories exist
        self.uploads_dir.mkdir(parents=True, exist_ok=True)
    
    def download_file(self, url: str, filename: Optional[str] = None) -> Dict[str, Any]:
        """
        Download file from URL
        
        Args:
            url (str): URL to download from
            filename (str, optional): Filename to save as
            
        Returns:
            dict: Download result with file path and metadata
        """
        if not REQUESTS_AVAILABLE:
            raise ImportError("requests library not available for downloads")
        
        try:
            # Make request with timeout
            response = requests.get(url, stream=True, timeout=DOWNLOAD_TIMEOUT)
            response.raise_for_status()
            
            # Determine filename
            if not filename:
                filename = self._extract_filename_from_url(url, response)
            
            # Generate unique filename to avoid conflicts
            unique_filename = f"{uuid.uuid4().hex[:8]}_{filename}"
            file_path = self.uploads_dir / unique_filename
            
            # Check file size from headers
            content_length = response.headers.get('content-length')
            if content_length and int(content_length) > 50 * 1024 * 1024:  # 50MB
                raise ValueError("File too large (>50MB)")
            
            # Download with size checking
            total_size = 0
            with open(file_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        total_size += len(chunk)
                        if total_size > 50 * 1024 * 1024:  # 50MB limit
                            f.close()
                            file_path.unlink()  # Delete partial file
                            raise ValueError("File too large (>50MB)")
                        f.write(chunk)
            
            # Validate file extension
            file_ext = file_path.suffix.lower()[1:]  # Remove dot
            if file_ext not in ALLOWED_EXTENSIONS:
                file_path.unlink()  # Delete file
                raise ValueError(f"File type '{file_ext}' not allowed")
            
            logger.info(f"File downloaded successfully: {file_path}")
            
            return {
                'success': True,
                'file_path': str(file_path),
                'filename': unique_filename,
                'original_filename': filename,
                'size': total_size,
                'url': url
            }
            
        except requests.RequestException as e:
            logger.error(f"Download failed: {str(e)}")
            return {
                'success': False,
                'error': f"Download failed: {str(e)}",
                'url': url
            }
        except Exception as e:
            logger.error(f"Download error: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'url': url
            }
    
    def _extract_filename_from_url(self, url: str, response: requests.Response) -> str:
        """Extract filename from URL or response headers"""
        # Try to get filename from Content-Disposition header
        content_disposition = response.headers.get('content-disposition')
        if content_disposition:
            import re
            filename_match = re.search(r'filename\*?=([^;\n\r"]*)', content_disposition)
            if filename_match:
                filename = filename_match.group(1).strip('"\'')
                if filename:
                    return filename
        
        # Extract from URL
        from urllib.parse import urlparse, unquote
        parsed_url = urlparse(url)
        filename = os.path.basename(unquote(parsed_url.path))
        
        if not filename or '.' not in filename:
            # Default filename based on content type
            content_type = response.headers.get('content-type', '')
            if 'pdf' in content_type:
                filename = f"document_{uuid.uuid4().hex[:6]}.pdf"
            elif 'docx' in content_type or 'document' in content_type:
                filename = f"document_{uuid.uuid4().hex[:6]}.docx"
            else:
                filename = f"file_{uuid.uuid4().hex[:6]}.pdf"  # Default to PDF
        
        return filename
    
    def cleanup_temp_files(self, older_than_hours: int = 24):
        """
        Clean up old temporary files in uploads directory
        
        Args:
            older_than_hours (int): Remove files older than this many hours
        """
        import time
        
        current_time = time.time()
        cutoff_time = current_time - (older_than_hours * 3600)  # Convert to seconds
        
        removed_count = 0
        
        for file_path in self.uploads_dir.iterdir():
            if file_path.is_file():
                file_mtime = file_path.stat().st_mtime
                if file_mtime < cutoff_time:
                    try:
                        file_path.unlink()
                        removed_count += 1
                        logger.info(f"Removed old temp file: {file_path.name}")
                    except Exception as e:
                        logger.error(f"Error removing file {file_path}: {str(e)}")
        
        logger.info(f"Cleanup complete: removed {removed_count} files")
        return removed_count
    
    def validate_file(self, file_path: str) -> Dict[str, Any]:
        """
        Validate file type and basic properties
        
        Args:
            file_path (str): Path to file
            
        Returns:
            dict: Validation results
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            return {
                'valid': False,
                'error': 'File does not exist'
            }
        
        # Check file extension
        file_ext = file_path.suffix.lower()[1:]  # Remove dot
        if file_ext not in ALLOWED_EXTENSIONS:
            return {
                'valid': False,
                'error': f'File type "{file_ext}" not allowed'
            }
        
        # Check file size
        file_size = file_path.stat().st_size
        if file_size > 50 * 1024 * 1024:  # 50MB
            return {
                'valid': False,
                'error': 'File too large (>50MB)'
            }
        
        if file_size == 0:
            return {
                'valid': False,
                'error': 'File is empty'
            }
        
        # Basic file type validation by reading headers
        try:
            with open(file_path, 'rb') as f:
                header = f.read(4)
                
            # Check PDF header
            if file_ext == 'pdf':
                if not header.startswith(b'%PDF'):
                    return {
                        'valid': False,
                        'error': 'Invalid PDF file format'
                    }
            
            # Check ZIP-based formats (DOCX)
            elif file_ext in ['docx']:
                if not header.startswith(b'PK'):  # ZIP header
                    return {
                        'valid': False,
                        'error': f'Invalid {file_ext.upper()} file format'
                    }
        
        except Exception as e:
            logger.warning(f"Could not validate file header: {str(e)}")
        
        return {
            'valid': True,
            'file_size': file_size,
            'file_type': file_ext
        }
    
    def get_file_info(self, file_path: str) -> Dict[str, Any]:
        """
        Get comprehensive file information
        
        Args:
            file_path (str): Path to file
            
        Returns:
            dict: File information
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            return {'exists': False}
        
        stat = file_path.stat()
        
        return {
            'exists': True,
            'name': file_path.name,
            'size': stat.st_size,
            'size_mb': round(stat.st_size / (1024 * 1024), 2),
            'extension': file_path.suffix.lower(),
            'created': stat.st_ctime,
            'modified': stat.st_mtime,
            'path': str(file_path.absolute())
        }

def download_file_from_url(url: str, filename: Optional[str] = None) -> Dict[str, Any]:
    """
    Convenience function to download file
    
    Args:
        url (str): URL to download
        filename (str, optional): Filename to save as
        
    Returns:
        dict: Download result
    """
    converter = FileConverter()
    return converter.download_file(url, filename)

def validate_uploaded_file(file_path: str) -> Dict[str, Any]:
    """
    Convenience function to validate file
    
    Args:
        file_path (str): Path to file
        
    Returns:
        dict: Validation result
    """
    converter = FileConverter()
    return converter.validate_file(file_path)