#!/bin/bash
# Activate both environments
source /var/www/chess.saphraxinos.com/myenv/bin/activate
source /var/www/chess.saphraxinos.com/myenv39/bin/activate

# Run Flask app using the Python from myenv
/var/www/chess.saphraxinos.com/myenv/bin/python /var/www/chess.saphraxinos.com/tts_api.py
