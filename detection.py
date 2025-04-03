import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
import numpy as np
import time
import threading

# Constants
MODEL_PATH = "DA2_3\drowsiness_model.pth"
IMAGE_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DROWSY_THRESHOLD = 0.7  # Threshold to classify as drowsy
ALERT_THRESHOLD = 5  # Number of consecutive drowsy frames to trigger alert
PREDICTION_INTERVAL = 3  # Predict every N frames for smooth 60fps

# Create model
def create_model(architecture='efficientnet_b0'):
    if architecture == 'efficientnet_b0':
        model = models.efficientnet_b0(pretrained=False)
        num_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.3, inplace=True),
            nn.Linear(num_features, 2)
        )
    else:
        # Fallback to ResNet18 if architecture not recognized
        model = models.resnet18(pretrained=False)
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, 2)
    
    return model

# Image transformations
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Global variables for threading
frame_buffer = None
prediction = None
drowsy_count = 0
lock = threading.Lock()

# Prediction thread function
def prediction_thread(model):
    global frame_buffer, prediction, drowsy_count
    
    while True:
        if frame_buffer is None:
            time.sleep(0.01)
            continue
        
        with lock:
            frame = frame_buffer.copy()
        
        try:
            # Preprocess image
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = transform(image).unsqueeze(0).to(DEVICE)
            
            # Make prediction
            with torch.no_grad():
                outputs = model(image)
                probs = torch.nn.functional.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs, 1)
                drowsy_prob = probs[0][1].item()
            
            # Update prediction
            is_drowsy = drowsy_prob > DROWSY_THRESHOLD
            
            with lock:
                prediction = {
                    'is_drowsy': is_drowsy,
                    'confidence': drowsy_prob
                }
                
                if is_drowsy:
                    drowsy_count += 1
                else:
                    drowsy_count = 0
                    
        except Exception as e:
            print(f"Error in prediction: {e}")
        
        # Sleep to avoid consuming too much CPU
        time.sleep(0.05)

# Main function for real-time detection
def main():
    global frame_buffer, prediction, drowsy_count
    
    print(f"Loading model from {MODEL_PATH}...")
    try:
        # Load the model
        checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
        architecture = checkpoint.get('model_architecture', 'efficientnet_b0')
        model = create_model(architecture)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(DEVICE)
        model.eval()
        print(f"Model loaded successfully! (Architecture: {architecture})")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Start prediction thread
    pred_thread = threading.Thread(target=prediction_thread, args=(model,), daemon=True)
    pred_thread.start()
    
    # Open webcam
    print("Opening webcam...")
    cap = cv2.VideoCapture(0)
    
    # Try to set camera resolution and frame rate
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 60)
    
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    
    print("Starting real-time detection...")
    
    frame_count = 0
    start_time = time.time()
    fps_update_interval = 0.5  # Update FPS twice a second
    frames_since_last_update = 0
    current_fps = 0
    
    # Alert state
    is_alerting = False
    alert_start_time = 0
    
    try:
        while True:
            # Read frame
            ret, frame = cap.read()
            
            if not ret:
                print("Error: Failed to read frame.")
                break
            
            # Update frame buffer for prediction thread (only on certain frames)
            if frame_count % PREDICTION_INTERVAL == 0:
                with lock:
                    frame_buffer = frame.copy()
            
            # Calculate FPS
            frames_since_last_update += 1
            elapsed_time = time.time() - start_time
            
            if elapsed_time >= fps_update_interval:
                current_fps = frames_since_last_update / elapsed_time
                frames_since_last_update = 0
                start_time = time.time()
            
            frame_count += 1
            
            # Draw on frame
            display_frame = frame.copy()
            
            # Display FPS
            cv2.putText(display_frame, f"FPS: {current_fps:.1f}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Display prediction results
            if prediction is not None:
                with lock:
                    current_pred = prediction.copy()
                    current_drowsy_count = drowsy_count
                
                # Determine status and color
                if current_pred['is_drowsy']:
                    status = "DROWSY"
                    confidence = current_pred['confidence']
                    color = (0, 0, 255)  # red
                else:
                    status = "ALERT"
                    confidence = 1 - current_pred['confidence']
                    color = (0, 255, 0)  # green
                
                # Display status
                cv2.putText(display_frame, f"Status: {status}", (10, 70), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                
                cv2.putText(display_frame, f"Confidence: {confidence:.2f}", (10, 110), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                
                # Check for alert condition
                if current_drowsy_count >= ALERT_THRESHOLD and not is_alerting:
                    is_alerting = True
                    alert_start_time = time.time()
                    print("DROWSINESS ALERT! Wake up!")
                
                # Display alert
                if is_alerting:
                    # Flash red border
                    if int(time.time() * 2) % 2 == 0:
                        cv2.rectangle(display_frame, (0, 0), (display_frame.shape[1], display_frame.shape[0]), (0, 0, 255), 20)
                    
                    # Display alert text
                    cv2.putText(display_frame, "DROWSINESS ALERT!", (display_frame.shape[1]//2 - 150, display_frame.shape[0]//2), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
                    
                    # Reset alert after 3 seconds
                    if time.time() - alert_start_time > 3.0:
                        is_alerting = False
            
            # Show frame
            cv2.imshow('Driver Drowsiness Detection', display_frame)
            
            # Check for exit key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
    except KeyboardInterrupt:
        print("Interrupted by user")
    finally:
        # Clean up
        cap.release()
        cv2.destroyAllWindows()
        print("Detection stopped")

if __name__ == "__main__":
    main()
