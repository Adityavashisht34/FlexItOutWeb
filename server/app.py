from flask import Flask, request, jsonify
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import cv2
import numpy as np
import base64
import mediapipe as mp
import os
import logging
import datetime
from flask_cors import CORS

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Initialize rate limiter
limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=["200 per day", "50 per hour"],
    storage_uri="memory://"
)

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    ba = a - b
    bc = c - b

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))
    return angle

# Exercise state tracking
exercise_states = {
    'Pushups': {'stage': None, 'reps': 0},
    'Squats': {'stage': None, 'reps': 0},
    'Bicep Curls': {'stage': None, 'reps': 0}
}

def process_frame(frame, exercise_type):
    try:
        # Convert frame to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # Process with MediaPipe Pose
        results = pose.process(image)

        if not results.pose_landmarks:
            logger.warning("No pose landmarks detected")
            return {'reps': exercise_states[exercise_type]['reps']}

        landmarks = results.pose_landmarks.landmark
        logger.debug(f"Processing frame for {exercise_type}")

        if exercise_type == 'Pushups':
            # Pushup logic
            left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                            landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            angle = calculate_angle(left_shoulder, left_elbow, [left_elbow[0], left_elbow[1] - 0.1])
            
            if angle > 100 and exercise_states[exercise_type]['stage'] != 'up':
                exercise_states[exercise_type]['stage'] = 'up'
                logger.debug("Pushup up position detected")
                
            if angle < 55 and exercise_states[exercise_type]['stage'] == 'up':
                exercise_states[exercise_type]['stage'] = 'down'
                exercise_states[exercise_type]['reps'] += 1
                logger.info(f"Pushup completed! Total reps: {exercise_states[exercise_type]['reps']}")
                
            return {'reps': exercise_states[exercise_type]['reps'], 'stage': exercise_states[exercise_type]['stage']}

        elif exercise_type == 'Squats':
            # Squat logic
            left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                       landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
            left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
            left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
            angle = calculate_angle(left_hip, left_knee, left_ankle)
            
            if angle > 150 and exercise_states[exercise_type]['stage'] != 'up':
                exercise_states[exercise_type]['stage'] = 'up'
                logger.debug("Squat up position detected")
                
            if angle < 110 and exercise_states[exercise_type]['stage'] == 'up':
                exercise_states[exercise_type]['stage'] = 'down'
                exercise_states[exercise_type]['reps'] += 1
                logger.info(f"Squat completed! Total reps: {exercise_states[exercise_type]['reps']}")
                
            return {'reps': exercise_states[exercise_type]['reps'], 'stage': exercise_states[exercise_type]['stage']}

        elif exercise_type == 'Bicep Curls':
            # Bicep curl logic
            left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                            landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
            angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
            
            if angle > 160 and exercise_states[exercise_type]['stage'] != 'down':
                exercise_states[exercise_type]['stage'] = 'down'
                logger.debug("Bicep curl down position detected")
                
            if angle < 30 and exercise_states[exercise_type]['stage'] == 'down':
                exercise_states[exercise_type]['stage'] = 'up'
                exercise_states[exercise_type]['reps'] += 1
                logger.info(f"Bicep curl completed! Total reps: {exercise_states[exercise_type]['reps']}")
                
            return {'reps': exercise_states[exercise_type]['reps'], 'stage': exercise_states[exercise_type]['stage']}

        return {'reps': 0}

    except Exception as e:
        print(f"Error processing frame: {e}")
        return {'reps': 0}

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    print("Server is running and healthy")
    return jsonify({
        'status': 'healthy',
        'message': 'Server is running successfully',
        'timestamp': datetime.datetime.now().isoformat()
    }), 200

@app.route('/process_frame', methods=['POST'])
@limiter.limit("10 per second")
def process_frame_endpoint():
    try:
        print("Received frame processing request")
        data = request.json
        if not data or 'frame' not in data or 'exercise_type' not in data:
            logger.warning("Invalid request data")
            return jsonify({'error': 'Invalid request data'}), 400

        frame_data = data['frame']
        exercise_type = data['exercise_type']

        if exercise_type not in ['Pushups', 'Squats', 'Bicep Curls']:
            logger.warning(f"Invalid exercise type: {exercise_type}")
            return jsonify({'error': 'Invalid exercise type'}), 400

        # Decode base64 frame
        frame_bytes = base64.b64decode(frame_data)
        frame_array = np.frombuffer(frame_bytes, dtype=np.uint8)
        frame = cv2.imdecode(frame_array, flags=cv2.IMREAD_COLOR)

        # Process frame with AI model
        result = process_frame(frame, exercise_type)

        logger.info(f"Successfully processed frame for {exercise_type}")
        return jsonify(result)
    except Exception as e:
        logger.error(f"Error processing frame: {str(e)}", exc_info=True)
        return jsonify({'error': 'Internal server error'}), 500

def webcam_interface():
    cap = cv2.VideoCapture(0)
    exercise_type = 'Pushups'  # Default exercise
    print("Press 'p' for Pushups, 's' for Squats, 'b' for Bicep Curls, 'q' to quit")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Process frame
        result = process_frame(frame, exercise_type)
        
        # Display rep count
        cv2.putText(frame, f"Exercise: {exercise_type}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Reps: {result['reps']}", (10, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.imshow('Fitness Tracker', frame)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('p'):
            exercise_type = 'Pushups'
        elif key == ord('s'):
            exercise_type = 'Squats'
        elif key == ord('b'):
            exercise_type = 'Bicep Curls'
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    print("Choose mode:")
    print("1. Webcam Interface")
    print("2. Web Server")
    choice = input("Enter choice (1 or 2): ")
    
    if choice == '1':
        webcam_interface()
    else:
        app.run(host='0.0.0.0', port=3000)
