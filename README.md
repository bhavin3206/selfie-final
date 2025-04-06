# Chip Hosting - Selfie Generator App

A web application for generating custom selfies with frames and text overlays.

## Features
- Upload and process images
- Add custom text overlays
- Store original and processed images
- View recent selfies
- Mobile-responsive design

## Requirements
- Python 3.8 or higher
- Dependencies listed in requirements.txt

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd chip-hosting-selfie
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Development
To run the app in development mode:
```bash
python app.py
```

## Production Deployment

1. Set environment variables:
```bash
export PRODUCTION=True  # On Windows: set PRODUCTION=True
export PORT=8080       # Optional, defaults to 8080
```

2. Run with gunicorn (Linux/Mac):
```bash
gunicorn wsgi:application
```

## Directory Structure
- `/static/uploads/original` - Original uploaded images
- `/static/uploads/processed` - Processed selfie images
- `/templates` - HTML templates
- `/static/css` - CSS files
- `/static/js` - JavaScript files

## Created by
Bluewave Solutions 