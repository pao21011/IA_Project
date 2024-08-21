from flask import(Flask, 
                  render_template, 
                  request, 
                  redirect, 
                  url_for, 
                  Response, 
                  jsonify, 
                  send_from_directory, 
                  send_file, 
                  session)
from ultralytics import YOLO
import cv2
import os
from PIL import Image
import uuid
import threading
import time
import numpy as np

app = Flask(__name__)

# use model 1 ( predict what is it )
#model1 = YOLO('model_version/model1_v7.pt')

# use model 2 ( predict the state )
#model2 = YOLO('model_version/model2_v2.pt')

# video model 
#modelVideo = YOLO('model_version/model_video_v2.pt')

# use model 1 ( predict what is it )
model1 = YOLO('best.pt')

# use model 2 ( predict the state )
model2 = YOLO('best.pt')

# video model 
modelVideo = YOLO('best.pt')

#====================================================================================#

# give every image name
def generate_unique_filename(filename):
    _, extension = os.path.splitext(filename)
    unique_filename = str(uuid.uuid4()) + extension
    return unique_filename

@app.route('/')
def home():
    return render_template('index.html')

@app.route("/liveDetect")
def liveDetect():
    return render_template('LiveDetect.html')
    
@app.route("/uploadVideo")
def uploadVideo():
    return render_template('UploadVideo.html')
    
@app.route("/objectDetection")
def objectDetection():
    return render_template('ObjectDetection.html')
    
@app.route('/index.css')
def serve_static_file():
    return send_from_directory('static', 'index.css')

@app.route('/imgpred', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Upload image
        if 'image' not in request.files:
            return redirect(request.url)

        file = request.files['image']
        
        # Check user uploaded image
        if file.filename == '':
            return redirect(request.url)

        if file:
            # Ensure the required directories exist
            base_dir = os.path.join('static', 'images')
            os.makedirs(os.path.join(base_dir, 'originals'), exist_ok=True)
            os.makedirs(os.path.join(base_dir, 'results_model1'), exist_ok=True)
            os.makedirs(os.path.join(base_dir, 'results_model2'), exist_ok=True)

            unique_filename = generate_unique_filename(file.filename)
            original_image_path = os.path.join(base_dir, 'originals', unique_filename)
            file.save(original_image_path)
            
            # Model predictions
            # Model 1
            results1 = model1(original_image_path)
            result_image1 = results1[0].plot()
            result_path1 = os.path.join(base_dir, 'results_model1', 'result_model1_' + unique_filename)
            Image.fromarray(result_image1[..., ::-1]).save(result_path1)
            summary1 = summarize_results_model(results1, "Model 1")

            # Model 2
            results2 = model2(original_image_path)
            result_image2 = results2[0].plot()
            result_path2 = os.path.join(base_dir, 'results_model2', 'result_model2_' + unique_filename)
            Image.fromarray(result_image2[..., ::-1]).save(result_path2)
            summary2 = summarize_results_model(results2, "Model 2")
                
            # Check if "wet" is detected
            alert_message = "Warning: Wet condition detected!" if "wet" in summary2 else None

            return render_template('ObjectDetection.html', summary1=summary1, summary2=summary2, image_pred1=result_path1, image_pred2=result_path2, image_path=original_image_path, alert_message=alert_message)

    return render_template('index.html', image_path=None)
    
def summarize_results_model(results, model_name):
    detected_classes = {}
    
    for result in results:
        for box in result.boxes.data:
            class_id = int(box[5])
            confidence = float(box[4])
            class_name = get_class_name(class_id, model_name)  # Pass model_name for differentiation
            
            if class_name in detected_classes:
                detected_classes[class_name].append(confidence)
            else:
                detected_classes[class_name] = [confidence]

    summary = []
    for class_name, scores in detected_classes.items():
        max_confidence = max(scores)
        summary.append(f"{class_name}: {max_confidence:.2f}")

    return f"{model_name} detected: " + ", ".join(summary) if summary else f"{model_name} detected: No objects detected."

def get_class_name(class_id, model_name):
    # Map class IDs to names based on the model
    class_map_model1 = {
        0: "dirt",
        1: "stone",
        # Add other classes for model 1
    }
    
    class_map_model2 = {
        0: "dry",
        1: "nothing",
        2: "uk",
        3: "wet",
        # Add other classes for model 2
    }

    if model_name == "Model 1":
        return class_map_model1.get(class_id, "unknown")
    elif model_name == "Model 2":
        return class_map_model2.get(class_id, "unknown")
        
#====================================================================================#

# video
# Global variable to manage processing status
processing = False

def generate_unique_filename(filename):
    return filename

def save_frame(frame, frame_number, output_path):
    filename = os.path.join(output_path, f'frame_{frame_number:04d}.jpg')
    cv2.imwrite(filename, frame)

def process_video(video_path, output_folder):
    global processing
    cap = cv2.VideoCapture(video_path)
    os.makedirs(output_folder, exist_ok=True)

    frame_number = 0
    while cap.isOpened() and processing:
        ret, frame = cap.read()
        if not ret:
            break

        results3 = modelVideo(frame)
        annotated_frame3 = results3[0].plot()

        save_frame(annotated_frame3, frame_number, output_folder)
        frame_number += 1

    cap.release()

    # Get the folder name for naming the output video
    folder_name = os.path.basename(os.path.normpath(output_folder))
    output_video_folder = os.path.join('static', 'output_videos')  # Specify your desired output folder
    os.makedirs(output_video_folder, exist_ok=True)

    output_video_path = os.path.join(output_video_folder, f'{folder_name}_output_video.mp4')
    create_video_from_images(output_folder, output_video_path)
    print(f"Video created at: {output_video_path}")

def create_video_from_images(image_folder, output_video_path, fps=30):
    images = []
    for root, _, files in os.walk(image_folder):
        for file in files:
            if file.endswith((".jpg", ".png")):
                images.append(os.path.join(root, file))

    images.sort()
    print(f"Found images: {images}")

    if not images:
        print("No images found in the directory.")
        return

    first_image_path = images[0]
    first_image = cv2.imread(first_image_path)
    if first_image is None:
        print(f"Failed to read the first image: {first_image_path}")
        return

    height, width, _ = first_image.shape
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    for img_path in images:
        frame = cv2.imread(img_path)
        if frame is None:
            print(f"Failed to read image: {img_path}")
            continue
        video_writer.write(frame)
        print(f"Written frame: {img_path}")

    video_writer.release()
    print(f"Video created at {output_video_path}")

@app.route('/vidpred', methods=['GET', 'POST'])
def vidpred():
    global processing
    if request.method == 'POST':
        if 'video' not in request.files:
            return redirect(request.url)

        file = request.files['video']
        
        if file.filename == '':
            return redirect(request.url)

        if file:
            unique_filename = generate_unique_filename(file.filename)
            video_path = os.path.join('static', 'videos', unique_filename)
            os.makedirs(os.path.dirname(video_path), exist_ok=True)
            file.save(video_path)

            # Create a unique folder for processed frames using a UUID
            output_folder_name = str(uuid.uuid4())
            output_folder = os.path.join('static', 'processed', output_folder_name)
            os.makedirs(output_folder, exist_ok=True)
            processing = True
            threading.Thread(target=process_video, args=(video_path, output_folder)).start()

            return render_template('UploadVideo.html', filename=unique_filename)

    return render_template('UploadVideo.html')

def generate_video_frames(video_path):
    cap = cv2.VideoCapture(video_path)

    while cap.isOpened() and processing:
        ret, frame = cap.read()
        if not ret:
            break

        results3 = modelVideo(frame)
        annotated_frame3 = results3[0].plot()

        ret, buffer = cv2.imencode('.jpg', annotated_frame3)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/video_feed/<filename>')
def video_feed(filename):
    return Response(generate_video_frames(os.path.join('static', 'videos', filename)), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stop_processing', methods=['POST'])
def stop_processing():
    global processing
    processing = False
    return jsonify(success=True)


    
#====================================================================================#

@app.route('/live_feed')
def live_feed():
    return Response(generate_live_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

def generate_live_frames():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 使用 model1 进行预测
        results1 = model1(frame)
        annotated_frame1 = results1[0].plot()

        # 使用 model2 进行预测
        results2 = model2(frame)
        annotated_frame2 = results2[0].plot()

        # 合并两个模型标注的帧
        combined_frame = cv2.addWeighted(annotated_frame1, 0.5, annotated_frame2, 0.5, 0)

        # 将合并后的帧转换为 JPEG 格式
        ret, buffer = cv2.imencode('.jpg', combined_frame)
        frame_bytes = buffer.tobytes()

        # 将帧字节作为响应的一部分进行返回
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()
    cv2.destroyAllWindows()

#====================================================================================#
def delete_images_after_delay():
    while True:
        time.sleep(120)  # 等兩分鐘
        image_folder = 'static/images'
        for filename in os.listdir(image_folder):
            file_path = os.path.join(image_folder, filename)
            try:
                if os.path.isfile(file_path):
                    # 睇下文件係咪已經超過24小時
                    if os.stat(file_path).st_mtime < time.time() - 86400:
                        os.remove(file_path)
            except Exception as e:
                print(f"刪除文件 {file_path} 嘅時候出錯: {e}")

# Flask route to delete images after 2 minutes

#====================================================================================#

@app.route('/delete', methods=['GET'])
def delete():
    threading.Thread(target=delete_images_after_delay).start()
    return jsonify({"message": "Images will be deleted continuously after 2 minutes."})

#====================================================================================#

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)