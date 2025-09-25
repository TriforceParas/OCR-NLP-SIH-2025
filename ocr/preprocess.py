import logging
from pathlib import Path
from typing import Optional, Tuple, Any
import numpy as np

try:
    from PIL import Image, ImageEnhance, ImageFilter
    import cv2
    PIL_AVAILABLE = True
    OPENCV_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    OPENCV_AVAILABLE = False

from config import OCR_DPI

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ImagePreprocessor:
    """Image preprocessing for better OCR results"""
    
    def __init__(self):
        self.dpi = OCR_DPI
        
    def preprocess_for_ocr(self, image_path: str, output_path: Optional[str] = None) -> str:
        """
        Preprocess an image to improve OCR accuracy
        
        Args:
            image_path (str): Path to input image
            output_path (str, optional): Path to save processed image
            
        Returns:
            str: Path to processed image
        """
        if not PIL_AVAILABLE:
            logger.warning("PIL not available, returning original image path")
            return image_path
            
        try:
            image_path = Path(image_path)
            
            # Load image
            image = Image.open(image_path)
            
            # Convert to grayscale for better OCR
            if image.mode != 'L':
                image = image.convert('L')
            
            # Enhance contrast
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(1.5)
            
            # Enhance sharpness
            enhancer = ImageEnhance.Sharpness(image)
            image = enhancer.enhance(2.0)
            
            # Apply slight blur to reduce noise
            image = image.filter(ImageFilter.GaussianBlur(radius=0.5))
            
            # Resize if image is too small (OCR works better on larger images)
            width, height = image.size
            if width < 1000 or height < 1000:
                # Calculate new size maintaining aspect ratio
                scale_factor = max(1000/width, 1000/height)
                new_width = int(width * scale_factor)
                new_height = int(height * scale_factor)
                image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            # Save processed image
            if output_path is None:
                output_path = image_path.parent / f"{image_path.stem}_processed{image_path.suffix}"
            
            image.save(output_path, dpi=(self.dpi, self.dpi))
            
            logger.info(f"Image preprocessed and saved to: {output_path}")
            return str(output_path)
            
        except Exception as e:
            logger.error(f"Image preprocessing failed: {str(e)}")
            return str(image_path)  # Return original path on error
    
    def deskew_image(self, image_path: str) -> str:
        """
        Deskew an image (rotate to correct orientation)
        Requires OpenCV
        """
        if not OPENCV_AVAILABLE:
            logger.warning("OpenCV not available, skipping deskew")
            return image_path
            
        try:
            # Read image
            image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
            
            if image is None:
                logger.warning(f"Could not read image: {image_path}")
                return image_path
            
            # Find edges
            edges = cv2.Canny(image, 50, 150, apertureSize=3)
            
            # Find lines using Hough transform
            lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)
            
            if lines is not None:
                # Calculate average angle
                angles = []
                for line in lines:
                    rho, theta = line[0]
                    angle = theta * 180 / np.pi
                    # Convert to rotation angle
                    if angle > 90:
                        angle = angle - 180
                    angles.append(angle)
                
                # Get median angle to avoid outliers
                median_angle = np.median(angles)
                
                # Only rotate if angle is significant
                if abs(median_angle) > 0.5:
                    # Get image dimensions
                    height, width = image.shape
                    center = (width // 2, height // 2)
                    
                    # Create rotation matrix
                    rotation_matrix = cv2.getRotationMatrix2D(center, median_angle, 1.0)
                    
                    # Rotate image
                    rotated = cv2.warpAffine(image, rotation_matrix, (width, height),
                                           flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
                    
                    # Save rotated image
                    output_path = Path(image_path).parent / f"{Path(image_path).stem}_deskewed{Path(image_path).suffix}"
                    cv2.imwrite(str(output_path), rotated)
                    
                    logger.info(f"Image deskewed by {median_angle:.2f} degrees")
                    return str(output_path)
            
            return image_path
            
        except Exception as e:
            logger.error(f"Deskewing failed: {str(e)}")
            return image_path
    
    def enhance_for_ocr(self, image_path: str, output_dir: Optional[str] = None) -> str:
        """
        Complete image enhancement pipeline for OCR
        
        Args:
            image_path (str): Input image path
            output_dir (str, optional): Output directory
            
        Returns:
            str: Path to enhanced image
        """
        try:
            # Step 1: Deskew if needed
            deskewed_path = self.deskew_image(image_path)
            
            # Step 2: General preprocessing
            if output_dir:
                output_path = Path(output_dir) / f"{Path(image_path).stem}_enhanced{Path(image_path).suffix}"
            else:
                output_path = None
                
            enhanced_path = self.preprocess_for_ocr(deskewed_path, str(output_path) if output_path else None)
            
            # Clean up intermediate files if different from original
            if deskewed_path != image_path and deskewed_path != enhanced_path:
                try:
                    Path(deskewed_path).unlink()
                except:
                    pass
            
            return enhanced_path
            
        except Exception as e:
            logger.error(f"Complete enhancement failed: {str(e)}")
            return image_path

def preprocess_image_for_ocr(image_path: str, output_dir: Optional[str] = None) -> str:
    """
    Convenience function for image preprocessing
    
    Args:
        image_path (str): Path to image file
        output_dir (str, optional): Output directory for processed image
        
    Returns:
        str: Path to processed image
    """
    preprocessor = ImagePreprocessor()
    return preprocessor.enhance_for_ocr(image_path, output_dir)