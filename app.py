# import cv2
# import numpy as np
# from PIL import Image, ImageDraw, ImageFont
# import io
# from flask import Flask, request, render_template, send_file, jsonify, send_from_directory
# import base64
# import os
# from datetime import datetime
# import json
# import logging

# # Configure logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# app = Flask(__name__)
# app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size


# BASE_DIR = os.path.abspath(os.path.dirname(__file__))

# # Create necessary directories
# UPLOAD_FOLDER = os.path.join(BASE_DIR, 'static', 'uploads')
# ORIGINAL_FOLDER = os.path.join(UPLOAD_FOLDER, 'original')
# PROCESSED_FOLDER = os.path.join(UPLOAD_FOLDER, 'processed')
# RECENT_IMAGES_FILE = os.path.join(BASE_DIR, 'static', 'recent_images.json')


# # Create directories if they don't exist
# for folder in [UPLOAD_FOLDER, ORIGINAL_FOLDER, PROCESSED_FOLDER]:
#     os.makedirs(folder, exist_ok=True)

# def load_recent_images():
#     if os.path.exists(RECENT_IMAGES_FILE):
#         with open(RECENT_IMAGES_FILE, 'r') as f:
#             images = json.load(f)
#             return images[:10]  # Always return only the 10 most recent images
#     return []

# def save_recent_image(processed_path, user_name):
#     recent_images = load_recent_images()
#     timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
#     recent_images.insert(0, {
#         'path': processed_path,
#         'name': user_name,
#         'timestamp': timestamp
#     })
#     # Keep only the last 10 images
#     recent_images = recent_images[:10]
#     with open(RECENT_IMAGES_FILE, 'w') as f:
#         json.dump(recent_images, f)

# def get_selfie_count():
#     recent_images = load_recent_images()
#     return len(recent_images)



# def add_selfie_to_template(user_image, template_image, user_name):
#     """Places the user's selfie inside the circular frame and adds the name."""
    
#     # Resize the user image before processing to reduce computation time
#     max_size = 800
#     if user_image.size[0] > max_size or user_image.size[1] > max_size:
#         ratio = min(max_size / user_image.size[0], max_size / user_image.size[1])
#         new_size = (int(user_image.size[0] * ratio), int(user_image.size[1] * ratio))
#         user_image = user_image.resize(new_size, Image.Resampling.LANCZOS)
    
#     # Convert images to OpenCV format
#     template_cv = cv2.cvtColor(np.array(template_image), cv2.COLOR_RGB2BGRA)
#     user_cv = cv2.cvtColor(np.array(user_image), cv2.COLOR_RGB2BGRA)
    
#     # Get template dimensions
#     template_height, template_width = template_cv.shape[:2]
#     user_height, user_width = user_cv.shape[:2]
    
#     # Calculate the center position for the circle
#     circle_x = template_width // 2  # Center horizontally
#     circle_y = int(template_height * 0.77)  # Position at 77% from top
#     circle_r = int(template_width * 0.17)  # Slightly smaller radius
    
#     # Calculate aspect ratio preserving dimensions for user image
#     target_size = circle_r * 2
#     if user_width > user_height:
#         new_width = int(user_width * target_size / user_height)
#         new_height = target_size
#         x_offset = (new_width - target_size) // 2
#         y_offset = 0
#     else:
#         new_width = target_size
#         new_height = int(user_height * target_size / user_width)
#         x_offset = 0
#         y_offset = (new_height - target_size) // 2
    
#     # Use INTER_AREA for downscaling and INTER_LINEAR for upscaling
#     interpolation = cv2.INTER_AREA if target_size < user_width else cv2.INTER_LINEAR
#     user_resized = cv2.resize(user_cv, (new_width, new_height), interpolation=interpolation)
    
#     # Crop to square from center
#     x_start = x_offset
#     x_end = x_offset + target_size
#     y_start = y_offset
#     y_end = y_offset + target_size
#     user_square = user_resized[y_start:y_end, x_start:x_end]
    
#     # Create a circular mask with anti-aliasing
#     mask = np.zeros((target_size, target_size, 4), dtype=np.uint8)
#     center = (target_size // 2, target_size // 2)
#     cv2.circle(mask, center, circle_r - 2, (255, 255, 255, 255), -1)
    
#     # Optimize feathering
#     feather_amount = 1
#     mask = cv2.GaussianBlur(mask, (feather_amount*2+1, feather_amount*2+1), 0)
    
#     # Apply mask to create circular image
#     alpha_mask = mask[:, :, 3] / 255.0
#     alpha_mask = np.stack([alpha_mask] * 4, axis=-1)
#     user_circular = user_square * alpha_mask
    
#     # Calculate position to place the circular image
#     y1 = circle_y - circle_r
#     y2 = circle_y + circle_r
#     x1 = circle_x - circle_r
#     x2 = circle_x + circle_r
    
#     # Create a region of interest and blend
#     roi = template_cv[y1:y2, x1:x2].copy()
#     template_cv[y1:y2, x1:x2] = user_circular + roi * (1 - alpha_mask)
    
#     # Convert back to PIL format
#     final_image = Image.fromarray(cv2.cvtColor(template_cv, cv2.COLOR_BGRA2RGBA))
    
#     # ==== Add Centered Username ====
#     draw = ImageDraw.Draw(final_image)
    
#     try:
#         # Try loading bold font (add fallback for different OS font paths)
#         font = ImageFont.truetype("arialbd.ttf", size=60)
#     except IOError:
#         try:
#             font = ImageFont.truetype("arial.ttf", size=60)
#         except IOError:
#             font = ImageFont.load_default()

#     # Clean and prepare text
#     text = user_name.strip()
    
#     # Calculate text dimensions using bbox method
#     bbox = draw.textbbox((0, 0), text, font=font)
#     text_width = bbox[2] - bbox[0]
#     text_height = bbox[3] - bbox[1]
    
#     # Calculate centered position
#     text_x = (template_width - text_width) // 2  # Center horizontally relative to template
#     text_y = y2 + 30  # Fixed vertical offset from circle bottom
    
#     # Add text with white color
#     draw.text((text_x, text_y), text, font=font, fill="white", stroke_width=2, stroke_fill="black")

#     return final_image

# @app.route('/')
# def index():
#     selfie_count = get_selfie_count()
#     recent_images = load_recent_images()
#     return render_template('index.html', selfie_count=selfie_count, recent_images=recent_images)

# @app.route('/process', methods=['POST'])
# def process_image():
#     try:
#         # Get the image data and name from the request
#         data = request.get_json()
#         if not data or 'image' not in data or 'name' not in data:
#             return jsonify({
#                 'success': False,
#                 'error': 'Missing required data (image or name)'
#             })

#         image_data = data['image'].split(',')[1]  # Remove the data URL prefix
#         user_name = data['name'].strip()
        
#         if not user_name:
#             return jsonify({
#                 'success': False,
#                 'error': 'Name is required'
#             })
        
#         try:
#             # Convert base64 to image
#             image_bytes = base64.b64decode(image_data)
#             user_image = Image.open(io.BytesIO(image_bytes)).convert("RGBA")
#         except Exception as e:
#             return jsonify({
#                 'success': False,
#                 'error': 'Invalid image data'
#             })
        
#         # Load template image
#         template_path = os.path.join(BASE_DIR, "static", "images", "template.png")
#         if not os.path.exists(template_path):
#             return jsonify({
#                 'success': False,
#                 'error': 'Template image not found'
#             })
            
#         template_image = Image.open(template_path).convert("RGBA")
        
#         # Generate unique filenames
#         timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
#         original_filename = f'original_{timestamp}.png'
#         processed_filename = f'processed_{timestamp}.png'
        
#         # Save paths
#         original_filepath = os.path.join(ORIGINAL_FOLDER, original_filename)
#         processed_filepath = os.path.join(PROCESSED_FOLDER, processed_filename)
        
#         # Save the original image
#         user_image.save(original_filepath, 'PNG')
        
#         try:
#             # Process the image
#             result = add_selfie_to_template(user_image, template_image, user_name)
            
#             # Save the processed image
#             result.save(processed_filepath, 'PNG')
#         except Exception as e:
#             # Clean up original image if processing fails
#             if os.path.exists(original_filepath):
#                 os.remove(original_filepath)
#             return jsonify({
#                 'success': False,
#                 'error': 'Error processing image'
#             })
        
#         # Save to recent images with processed path only
#         save_recent_image(f'uploads/processed/{processed_filename}', user_name)
        
#         # Convert to base64 for sending back to client
#         img_io = io.BytesIO()
#         result.save(img_io, 'PNG')
#         img_io.seek(0)
#         img_str = base64.b64encode(img_io.getvalue()).decode('utf-8')
        
#         return jsonify({
#             'success': True,
#             'image': f'data:image/png;base64,{img_str}',
#             'selfie_count': get_selfie_count()
#         })
#     except Exception as e:
#         return jsonify({
#             'success': False,
#             'error': str(e)
#         })

# @app.route('/recent-images')
# def get_recent_images():
#     images = load_recent_images()  # This will now always return max 10 images
#     return jsonify({'images': images})  # Return images in the expected format

# if __name__ == '__main__':
#     logger.info("Selfie Generator startup")
#     if os.environ.get('PRODUCTION'):
#         # Production settings
#         app.config['DEBUG'] = False
#         app.config['ENV'] = 'production'
#         port = int(os.environ.get('PORT', 8080))
#         app.run(host='0.0.0.0', port=port)
#     else:
#         # Development settings
#         app.config['DEBUG'] = True
#         app.run(host='127.0.0.1', port=5000)

# # WSGI entry point
# application = app

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import io
from flask import Flask, request, render_template, send_file, jsonify, send_from_directory
import base64
import os
from datetime import datetime
import json
import logging
import logging.handlers
import traceback
import uuid
import socket

# Configure enhanced logging
def setup_logging():
    log_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'logs')
    os.makedirs(log_dir, exist_ok=True)
    
    # Configure root logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # Clear any existing handlers
    if logger.handlers:
        logger.handlers.clear()
    
    # Log format with timestamp, log level, file, line number, and message
    log_format = logging.Formatter(
        '%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_format)
    logger.addHandler(console_handler)
    
    # File handlers
    file_handler = logging.handlers.TimedRotatingFileHandler(
        os.path.join(log_dir, 'app.log'),
        when='midnight',
        backupCount=7  # Keep logs for a week
    )
    file_handler.setFormatter(log_format)
    logger.addHandler(file_handler)
    
    # Error file handler - separate file for errors
    error_file_handler = logging.handlers.RotatingFileHandler(
        os.path.join(log_dir, 'error.log'),
        maxBytes=10485760,  # 10MB
        backupCount=5
    )
    error_file_handler.setLevel(logging.ERROR)
    error_file_handler.setFormatter(log_format)
    logger.addHandler(error_file_handler)
    
    return logger

# Initialize logging
logger = setup_logging()

# Request ID middleware
class RequestIDMiddleware:
    def __init__(self, app):
        self.app = app
        self.hostname = socket.gethostname()
    
    def __call__(self, environ, start_response):
        request_id = str(uuid.uuid4())
        environ['REQUEST_ID'] = request_id
        
        def _start_response(status, headers, exc_info=None):
            headers.append(('X-Request-ID', request_id))
            return start_response(status, headers, exc_info)
        
        return self.app(environ, _start_response)

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.wsgi_app = RequestIDMiddleware(app.wsgi_app)

# Log each request
# @app.before_request
# def log_request_info():
#     logger.info(f"Request: {request.method} {request.path} - IP: {request.remote_addr}")
#     if request.get_data():
#         content_length = request.headers.get('Content-Length', 0)
#         logger.debug(f"Request data size: {content_length} bytes")

# Log each response
# @app.after_request
# def log_response_info(response):
#     logger.info(f"Response: {response.status_code}")
#     return response

BASE_DIR = os.path.abspath(os.path.dirname(__file__))

# Create necessary directories
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'static', 'uploads')
ORIGINAL_FOLDER = os.path.join(UPLOAD_FOLDER, 'original')
PROCESSED_FOLDER = os.path.join(UPLOAD_FOLDER, 'processed')
RECENT_IMAGES_FILE = os.path.join(BASE_DIR, 'static', 'recent_images.json')

# Create directories if they don't exist
for folder in [UPLOAD_FOLDER, ORIGINAL_FOLDER, PROCESSED_FOLDER]:
    os.makedirs(folder, exist_ok=True)
    logger.info(f"Directory ensured: {folder}")

def load_recent_images():
    try:
        if os.path.exists(RECENT_IMAGES_FILE):
            with open(RECENT_IMAGES_FILE, 'r') as f:
                images = json.load(f)
                logger.debug(f"Loaded {len(images)} recent images")
                return images[:10]  # Always return only the 10 most recent images
    except Exception as e:
        logger.error(f"Failed to load recent images: {str(e)}")
        logger.debug(traceback.format_exc())
    return []

def save_recent_image(processed_path, user_name):
    try:
        recent_images = load_recent_images()
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        recent_images.insert(0, {
            'path': processed_path,
            'name': user_name,
            'timestamp': timestamp
        })
        # Keep only the last 10 images
        recent_images = recent_images[:10]
        with open(RECENT_IMAGES_FILE, 'w') as f:
            json.dump(recent_images, f)
        logger.info(f"Saved new image to recent list for user: {user_name}")
    except Exception as e:
        logger.error(f"Failed to save recent image: {str(e)}")
        logger.debug(traceback.format_exc())

def get_selfie_count():
    recent_images = load_recent_images()
    count = len(recent_images)
    logger.debug(f"Current selfie count: {count}")
    return count

def add_selfie_to_template(user_image, template_image, user_name):
    """Places the user's selfie inside the circular frame and adds the name."""
    logger.info(f"Processing selfie for user: {user_name}")
    
    try:
        # Resize the user image before processing to reduce computation time
        max_size = 800
        original_size = user_image.size
        if user_image.size[0] > max_size or user_image.size[1] > max_size:
            ratio = min(max_size / user_image.size[0], max_size / user_image.size[1])
            new_size = (int(user_image.size[0] * ratio), int(user_image.size[1] * ratio))
            user_image = user_image.resize(new_size, Image.Resampling.LANCZOS)
            logger.debug(f"Resized image from {original_size} to {new_size}")
        
        # Convert images to OpenCV format
        template_cv = cv2.cvtColor(np.array(template_image), cv2.COLOR_RGB2BGRA)
        user_cv = cv2.cvtColor(np.array(user_image), cv2.COLOR_RGB2BGRA)
        
        # Get template dimensions
        template_height, template_width = template_cv.shape[:2]
        user_height, user_width = user_cv.shape[:2]
        logger.debug(f"Template dimensions: {template_width}x{template_height}, User dimensions: {user_width}x{user_height}")
        
        # Calculate the center position for the circle
        circle_x = template_width // 2  # Center horizontally
        circle_y = int(template_height * 0.77)  # Position at 77% from top
        circle_r = int(template_width * 0.17)  # Slightly smaller radius
        logger.debug(f"Circle parameters: center=({circle_x}, {circle_y}), radius={circle_r}")
        
        # Calculate aspect ratio preserving dimensions for user image
        target_size = circle_r * 2
        if user_width > user_height:
            new_width = int(user_width * target_size / user_height)
            new_height = target_size
            x_offset = (new_width - target_size) // 2
            y_offset = 0
        else:
            new_width = target_size
            new_height = int(user_height * target_size / user_width)
            x_offset = 0
            y_offset = (new_height - target_size) // 2
        
        logger.debug(f"Resizing to: {new_width}x{new_height}, offsets: ({x_offset}, {y_offset})")
        
        # Use INTER_AREA for downscaling and INTER_LINEAR for upscaling
        interpolation = cv2.INTER_AREA if target_size < user_width else cv2.INTER_LINEAR
        user_resized = cv2.resize(user_cv, (new_width, new_height), interpolation=interpolation)
        
        # Crop to square from center
        x_start = x_offset
        x_end = x_offset + target_size
        y_start = y_offset
        y_end = y_offset + target_size
        user_square = user_resized[y_start:y_end, x_start:x_end]
        
        # Create a circular mask with anti-aliasing
        mask = np.zeros((target_size, target_size, 4), dtype=np.uint8)
        center = (target_size // 2, target_size // 2)
        cv2.circle(mask, center, circle_r - 2, (255, 255, 255, 255), -1)
        
        # Optimize feathering
        feather_amount = 1
        mask = cv2.GaussianBlur(mask, (feather_amount*2+1, feather_amount*2+1), 0)
        
        # Apply mask to create circular image
        alpha_mask = mask[:, :, 3] / 255.0
        alpha_mask = np.stack([alpha_mask] * 4, axis=-1)
        user_circular = user_square * alpha_mask
        
        # Calculate position to place the circular image
        y1 = circle_y - circle_r
        y2 = circle_y + circle_r
        x1 = circle_x - circle_r
        x2 = circle_x + circle_r
        
        # Create a region of interest and blend
        roi = template_cv[y1:y2, x1:x2].copy()
        template_cv[y1:y2, x1:x2] = user_circular + roi * (1 - alpha_mask)
        
        # Convert back to PIL format
        final_image = Image.fromarray(cv2.cvtColor(template_cv, cv2.COLOR_BGRA2RGBA))
        
        # ==== Add Centered Username ====
        # Add user name with optimized font loading
        draw = ImageDraw.Draw(final_image)
        
        # Calculate font size based on template width
        font_size = int(template_width * 0.06)  # 6% of template width
        min_font_size = 36
        max_font_size = 72
        font_size = max(min_font_size, min(font_size, max_font_size))
        
        # Try multiple font paths
        font_paths = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",  # Linux
            "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",  # Linux alternative
            "arial.ttf",  # Windows
            "/System/Library/Fonts/Helvetica.ttc",  # macOS
            os.path.join(BASE_DIR, "static", "fonts", "Arial.ttf")  # Local fallback
        ]
        
        font = None
        for font_path in font_paths:
            try:
                font = ImageFont.truetype(font_path, 40)
                break
            except Exception as e:
                logger.warning(f"Failed to load font from {font_path}: {str(e)}")
                continue
        
        if font is None:
            logger.warning("Using default font as no system fonts were found")
            font = ImageFont.load_default()
        
        # Center align the text
        text_width = draw.textlength(user_name, font=font)
        text_x = circle_x - text_width // 2
        text_y = y2 + 20  # Increased spacing from circle
        
        # Add text with yellow color and black outline for better visibility
        outline_color = (0, 0, 0, 255)  # Black outline
        text_color = (255, 223, 0, 255)  # Yellow text
        
        # Draw text outline
        for offset in [(1,1), (-1,-1), (1,-1), (-1,1)]:
            draw.text((text_x + offset[0], text_y + offset[1]), user_name, fill=outline_color, font=font)
        
        # Draw main text
        draw.text((text_x, text_y), user_name, fill=text_color, font=font)
        
        return final_image
    except Exception as e:
        logger.error(f"Error in add_selfie_to_template: {str(e)}")
        logger.debug(traceback.format_exc())
        raise

@app.route('/back')
def index():
    try:
        logger.info("Loading index page")
        selfie_count = get_selfie_count()
        recent_images = load_recent_images()
        return render_template('index.html', selfie_count=selfie_count, recent_images=recent_images)
    except Exception as e:
        logger.error(f"Error loading index page: {str(e)}")
        logger.debug(traceback.format_exc())
        return "An error occurred loading the page. Please check the logs.", 500


@app.route('/')
def index():
    try:
        logger.info("Loading index page")
        return render_template('new.html')
    except Exception as e:
        logger.error(f"Error loading index page: {str(e)}")
        logger.debug(traceback.format_exc())
        return "An error occurred loading the page. Please check the logs.", 500

@app.route('/process', methods=['POST'])
def process_image():
    try:
        logger.info("Processing image request")
        # Get the image data and name from the request
        data = request.get_json()
        if not data or 'image' not in data or 'name' not in data:
            logger.warning("Missing required data (image or name)")
            return jsonify({
                'success': False,
                'error': 'Missing required data (image or name)'
            })

        image_data = data['image'].split(',')[1]  # Remove the data URL prefix
        user_name = data['name'].strip()
        
        if not user_name:
            logger.warning("Empty name provided")
            return jsonify({
                'success': False,
                'error': 'Name is required'
            })
        
        try:
            # Convert base64 to image
            image_bytes = base64.b64decode(image_data)
            user_image = Image.open(io.BytesIO(image_bytes)).convert("RGBA")
            logger.info(f"Successfully decoded image, size: {user_image.size}")
        except Exception as e:
            logger.error(f"Invalid image data: {str(e)}")
            logger.debug(traceback.format_exc())
            return jsonify({
                'success': False,
                'error': 'Invalid image data'
            })
        
        # Load template image
        template_path = os.path.join(BASE_DIR, "static", "images", "template.png")
        if not os.path.exists(template_path):
            logger.error(f"Template image not found at: {template_path}")
            return jsonify({
                'success': False,
                'error': 'Template image not found'
            })
            
        template_image = Image.open(template_path).convert("RGBA")
        logger.debug(f"Loaded template image, size: {template_image.size}")
        
        # Generate unique filenames
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        original_filename = f'original_{timestamp}.png'
        processed_filename = f'processed_{timestamp}.png'
        
        # Save paths
        original_filepath = os.path.join(ORIGINAL_FOLDER, original_filename)
        processed_filepath = os.path.join(PROCESSED_FOLDER, processed_filename)
        
        # Save the original image
        user_image.save(original_filepath, 'PNG')
        logger.debug(f"Saved original image to: {original_filepath}")
        
        try:
            # Process the image
            result = add_selfie_to_template(user_image, template_image, user_name)
            
            # Save the processed image
            result.save(processed_filepath, 'PNG')
            logger.debug(f"Saved processed image to: {processed_filepath}")
        except Exception as e:
            logger.error(f"Error processing image: {str(e)}")
            logger.debug(traceback.format_exc())
            return jsonify({
                'success': False,
                'error': 'Error processing image'
            })
        finally:
            if os.path.exists(original_filepath):
                os.remove(original_filepath)
                logger.debug(f"Cleaned up original image after processing error")

        
        # Save to recent images with processed path only
        save_recent_image(f'uploads/processed/{processed_filename}', user_name)
        
        # Convert to base64 for sending back to client
        img_io = io.BytesIO()
        result.save(img_io, 'PNG')
        img_io.seek(0)
        img_str = base64.b64encode(img_io.getvalue()).decode('utf-8')
        
        logger.info(f"Successfully processed image for user: {user_name}")
        return jsonify({
            'success': True,
            'image': f'data:image/png;base64,{img_str}',
            'selfie_count': get_selfie_count()
        })
    
    except Exception as e:
        logger.error(f"Unhandled exception in process_image: {str(e)}")
        logger.debug(traceback.format_exc())
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/recent-images')
def get_recent_images():
    try:
        logger.info("Getting recent images")
        images = load_recent_images()  # This will now always return max 10 images
        return jsonify({'images': images})  # Return images in the expected format
    except Exception as e:
        logger.error(f"Error getting recent images: {str(e)}")
        logger.debug(traceback.format_exc())
        return jsonify({'images': [], 'error': str(e)})

@app.errorhandler(Exception)
def handle_exception(e):
    """Handle all unhandled exceptions"""
    logger.error(f"Unhandled exception: {str(e)}")
    logger.debug(traceback.format_exc())
    return jsonify({
        'success': False,
        'error': 'Internal server error'
    }), 500

if __name__ == '__main__':
    logger.info("Selfie Generator startup")
    if os.environ.get('PRODUCTION'):
        # Production settings
        app.config['DEBUG'] = False
        app.config['ENV'] = 'production'
        port = int(os.environ.get('PORT', 8080))
        logger.info(f"Running in PRODUCTION mode on port {port}")
        app.run(host='0.0.0.0', port=port)
    else:
        # Development settings
        app.config['DEBUG'] = True
        logger.info("Running in DEVELOPMENT mode on port 5000")
        app.run(host='127.0.0.1', port=5000)

# WSGI entry point
application = app