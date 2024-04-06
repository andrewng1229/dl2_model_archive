import cv2
import tkinter as tk
from tkinter import ttk, END
from PIL import Image, ImageTk
import yolov5
#from moviepy.editor import VideoFileClip
from cv2_video import playVideo
import torch
import mediapipe as mp
from mediapipe.tasks.python import vision
import torch
import supervision
#import transformers
#import pytorch_lightning
#import timm
from transformers import DetrImageProcessor
import cv2
import matplotlib.pyplot as plt
from transformers import DetrForObjectDetection

class ObjectDetectionApp:
    def __init__(self, window):
        self.update_video_flag = True
        self.window = window
        self.window.title("Object Detection App")
        self.detected_classes = []
        self.model_options = ['yolos_best.pt', '../Mediapipe/naruto_hand_gestures.task', 'yolom_best.pt', 'yoloNano_best.pt', 'detr']
        self.timecount = 0
        
        self.selected_model = tk.StringVar()
        self.class_history = []  # Store class predictions over multiple frames
        self.patterns_name = ['water_dragon', 'ryuuka', 'mud_wall', 'housenka', 'death_reaper']
        self.patterns_to_detect = [
           #water dragon
            ['rat', 'dog', 'rabbit', 'tiger'],

            #ryuuka
            ['snake', 'dragon', 'rabbit', 'tiger'],

            #mud wall
            ['tiger', 'dog', 'ox', 'snake'],

            #housenka
            ['rat', 'tiger', 'dog', 'ox', 'rabbit', 'tiger'],

            #death_reaper
            ['snake', 'boar', 'tiger', 'rabbit', 'dog', 'rat', 'bird', 'horse', 'snake']
          ]  # Define the pattern to detect
        

        font_format = ('Ubuntu', 20)

        # Create GUI components
        self.video_label = tk.Label(window, font=font_format)
        #self.video_label.pack()
        self.video_label.grid(row=0, column=0)
        
        self.start_button = tk.Button(window, text="Start", command=self.start_detection, font=font_format)
        # self.start_button.pack()
        self.start_button.grid(row=1,column=0)

        self.stop_button = tk.Button(window, text="Stop", command=self.stop_detection, state=tk.DISABLED, font=font_format)
        # self.stop_button.pack()
        self.stop_button.grid(row=2,column=0)

               
        self.selected_model.set(self.model_options[0])
        self.model_dropdown = ttk.Combobox(window, textvariable=self.selected_model, values=self.model_options, font=font_format)
        self.model_dropdown.grid(row=3,column=0)

        # Create a frame to hold both listboxes
        #self.model_dropdown.bind("<<ComboboxSelected>>", self.update_model_selection)

        # Component - Detect Frames
        self.listbox_frame = tk.Frame(window)
        self.listbox_frame.grid(row=0,column=1)

        self.listbox_frame2 = tk.Frame(window)
        self.listbox_frame2.grid(row=0,column=2)

        # Create a frame for the left grid layout
        frame1 = tk.Frame(self.listbox_frame2)
        frame1.grid(row=0, column=0)

        self.label1 = tk.Label(frame1, text="Detection", font=font_format)
        self.label1.grid(row=0, column=0)

        self.detection_history = tk.Listbox(frame1, font=font_format)
        self.detection_history.grid(row=1, column=0)

        # Create a frame for the right grid layout
        frame2 = tk.Frame(self.listbox_frame2)
        frame2.grid(row=0, column=1)

        self.label2 = tk.Label(frame2, text="Hand Seal Formed", font=font_format)
        self.label2.grid(row=0, column=0)

        # Create the first listbox for class history
        self.class_history_listbox = tk.Listbox(frame2, font=font_format)
        self.class_history_listbox.grid(row=1, column=0)
       
      


    def start_detection(self):
        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        self.cap = cv2.VideoCapture(0)  # 0 for the default webcam
        print('cap is defined')
        self.update_video_flag = True

        #cv2 record video
        self.fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self.out = cv2.VideoWriter('output.mp4', self.fourcc, 10.0, (640, 480))


        #self.selected_model.set(self.model_dropdown.get())
        if self.model_dropdown.get() == 'yolos_best.pt' or \
            self.model_dropdown.get() == 'yolom_best.pt' or \
            self.model_dropdown.get() == 'yoloNano_best.pt':
            # Load pretrained model
            self.model = yolov5.load(self.selected_model.get())
            print('yolo loaded')
            # Set model parameters
            self.model.conf = 0.1  # NMS confidence threshold
            self.model.iou = 0.45   # NMS IoU threshold
            self.model.agnostic = False  # NMS class-agnostic
            self.model.multi_label = False  # NMS multiple labels per box
            self.model.max_det = 1  # Maximum number of detections per image
            self.update_video()
        elif self.model_dropdown.get() == '../Mediapipe/naruto_hand_gestures.task':
            self.recognizer = vision.GestureRecognizer.create_from_model_path('../Mediapipe/naruto_hand_gestures.task')
            self.recognizer_landmark = mp.solutions.hands.Hands(static_image_mode=False, max_num_hands=2)
            print('mediapipe loaded')
            self.update_video_mediapipe()
        elif self.model_dropdown.get() == 'detr':
            
            MODEL_PATH = 'detr_model_100'

            # loading model
            self.model = DetrForObjectDetection.from_pretrained(MODEL_PATH)
            DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.model.to(DEVICE)

            self.update_video_detr()
            pass
        
        else:
            print('No model selected!')
        
            
        # Open webcam
        print('model: ', self.selected_model)
        print('self.model_dropdown.get():', self.model_dropdown.get())
        
        

    def stop_detection(self):
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.update_video_flag = False
        # Release the webcam
        self.cap.release()
        self.out.release()
        cv2.destroyAllWindows()
        #root.destroy()

    def update_video_detr(self):
        id2label = { 0: 'bird', 1:'boar', 2:'dog', 3:'dragon', 4:'horse', 5:'monkey', 6:'ox',
             7:'rabbit', 8:'rat', 9:'sheep', 10:'snake', 11:'tiger'}

        image_processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
        box_annotator = supervision.BoxAnnotator()

        # Read a frame from the video
        ret, frame = self.cap.read()
        
        DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Perform object detection on the frame
        with torch.no_grad():
            # Load image and predict
            inputs = image_processor(images=frame, return_tensors='pt').to(DEVICE)
            outputs = self.model(**inputs)

            # Post-process
            target_sizes = torch.tensor([frame.shape[:2]]).to(DEVICE)
            results = image_processor.post_process_object_detection(
                outputs=outputs,
                threshold=0.5,
                target_sizes=target_sizes
            )[0]

            # Extract detections
            detections = supervision.Detections.from_transformers(transformers_results=results)
            #print(detections)
            labels = [f"{id2label[class_id]} {confidence:.2f}" for _, confidence, class_id, _ in detections]
            self.detection_history.insert(0, labels)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_detections = box_annotator.annotate(scene=frame.copy(), detections=detections, labels=labels)
            frame = frame_detections
            if detections.class_id > 0:
                detected_class = id2label[detections.class_id[0]]
                print(detected_class)
                self.detected_classes.append(detected_class)
                self.timecount = 0
            else:
                self.timecount += 1

        if self.timecount > 10:
            self.class_history = []
            self.timecount = 0
            self.detected_classes = []
        
        print('timecount:', self.timecount)

        # store the record of recent detect, max 5 times
        if len(self.detected_classes) > 5:
            self.detected_classes.pop(0)
        
        print("detected_classes: ", self.detected_classes)
        # if all the detection are the same, then insert to self.classhistory
        #situation 1: just start, the classhistory is empty
        if len(self.class_history)==0 and len(self.detected_classes)==5:
            if all(item == self.detected_classes[0] for item in self.detected_classes):
                self.class_history.append(self.detected_classes[0])
        #situation 2: the class history is not empty, checkt the first position class
            # if same with the detection class, then ignore
        elif len(self.class_history)>0 and len(self.detected_classes)==5:
            if all(item == self.detected_classes[0] for item in self.detected_classes) and self.class_history[-1] != self.detected_classes[0]:
                self.class_history.append(self.detected_classes[0])
        # store the class history, max 10 classes
        if len(self.class_history) > 10:
            self.class_history.pop(0)
        
    
        # for i in range(len(self.class_history) - 2):
        #         if all(item in self.class_history[i:i+3] for item in self.pattern_to_detect):
        #             # Pattern detected, draw "ninja" text on the frame
        #             cv2.putText(frame, "NINJA", (50, 50),
        #                         cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 2)
        #             break
        for pattern in self.patterns_to_detect:
            pattern_detected = False
            for i in range(len(self.class_history) - len(pattern) + 1):
                if self.class_history[i:i+len(pattern)] == pattern:
                    # Pattern detected, draw pattern_name text on the frame
                    cv2.putText(frame, self.patterns_name[self.patterns_to_detect.index(pattern)], (50, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 2)
                    playVideo(self.patterns_name[self.patterns_to_detect.index(pattern)]+ '.mp4')
                    pattern_detected = True
                    self.class_history = []
                    break
            if pattern_detected:
                break  # Exit the outer loop
        self.class_history_listbox.delete(0, tk.END)
        for item in self.class_history:
            self.class_history_listbox.insert(tk.END, item)

        print('class_history:', self.class_history)

        # Convert the frame to ImageTk format
        img = Image.fromarray(frame)
        imgtk = ImageTk.PhotoImage(image=img)

        # Update the video label
        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk)

        # Call update_video recursively after 10 ms
        self.window.after(10, self.update_video_detr)
        
    def update_video_mediapipe(self):
        if not self.update_video_flag:
            return 
        ret, frame = self.cap.read()
        frame = cv2.flip(frame, 1)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.recognizer_landmark.process(frame)
        cv2.imwrite('temp_frame.jpg', frame)
        image = mp.Image.create_from_file('temp_frame.jpg')

        recognition_result = self.recognizer.recognize(results)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.timecount += 1
                pass

        if results.multi_handedness:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                hand_label = handedness.classification[0].label
                mp.solutions.drawing_utils.draw_landmarks(frame, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)
                if hand_label == "Left":
                    cv2.putText(frame, f'Hand: {hand_label}', (45, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                else:
                    cv2.putText(frame, f'Hand: {hand_label}', (45, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                if recognition_result.gestures:
                    top_gesture = recognition_result.gestures[0][0]
                    self.detected_classes.append(top_gesture.category_name)
                    self.timecount = 0
                    cv2.putText(frame, f'Gesture recognized: {top_gesture.category_name} ({top_gesture.score})',
                                (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        if self.timecount > 10:
            self.class_history = []
            self.timecount = 0
            self.detected_classes = []
        
        print('timecount:', self.timecount)

        # store the record of recent detect, max 5 times
        if len(self.detected_classes) > 5:
            self.detected_classes.pop(0)
        
        print("detected_classes: ", self.detected_classes)
        # if all the detection are the same, then insert to self.classhistory
        #situation 1: just start, the classhistory is empty
        if len(self.class_history)==0 and len(self.detected_classes)==5:
            if all(item == self.detected_classes[0] for item in self.detected_classes):
                self.class_history.append(self.detected_classes[0])
        #situation 2: the class history is not empty, checkt the first position class
            # if same with the detection class, then ignore
        elif len(self.class_history)>0 and len(self.detected_classes)==5:
            if all(item == self.detected_classes[0] for item in self.detected_classes) and self.class_history[-1] != self.detected_classes[0]:
                self.class_history.append(self.detected_classes[0])
        # store the class history, max 50 classes
        if len(self.class_history) > 10:
            self.class_history.pop(0)
        
    
        # for i in range(len(self.class_history) - 2):
        #         if all(item in self.class_history[i:i+3] for item in self.pattern_to_detect):
        #             # Pattern detected, draw "ninja" text on the frame
        #             cv2.putText(frame, "NINJA", (50, 50),
        #                         cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 2)
        #             break
        for pattern in self.patterns_to_detect:
            pattern_detected = False
            for i in range(len(self.class_history) - len(pattern) + 1):
                if self.class_history[i:i+len(pattern)] == pattern:
                    # Pattern detected, draw pattern_name text on the frame
                    cv2.putText(frame, self.patterns_name[self.patterns_to_detect.index(pattern)], (50, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 2)
                    playVideo(self.patterns_name[self.patterns_to_detect.index(pattern)]+ '.mp4')
                    pattern_detected = True
                    self.class_history = []
                    break
            if pattern_detected:
                break  # Exit the outer loop
        self.class_history_listbox.delete(0, tk.END)
        for item in self.class_history:
            self.class_history_listbox.insert(tk.END, item)

        print('class_history:', self.class_history)

        # Convert the frame to ImageTk format
        img = Image.fromarray(frame)
        imgtk = ImageTk.PhotoImage(image=img)

        # Update the video label
        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk)

        # Call update_video recursively after 10 ms
        self.window.after(10, self.update_video_mediapipe)
    
    def update_video(self):
        if not self.update_video_flag:
            return 
        ret, frame = self.cap.read()  # Capture frame-by-frame

        # Write the frame into the output video file
        self.out.write(frame)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        

        # Perform inference
        results = self.model(frame)
        print('timecount :, ', self.timecount)
        if len(results.pred[0]) == 0:
            self.timecount += 1
        # Draw bounding boxes on the frame
        for pred in results.pred[0]:
            xmin, ymin, xmax, ymax, conf, cls = map(int, pred[:6])
            conf = pred[4]
            print(conf)
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            cv2.putText(frame, f'{self.model.names[int(cls)]} {conf:.2f}', (xmin, ymin - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            self.detection_history.insert(0, f'{self.model.names[int(cls)]} {conf:.2f}')
            class_name = self.model.names[int(cls)]
            self.detected_classes.append(class_name)
            self.timecount = 0
        
        print(results.pred[0])

        if self.timecount > 10:
            self.class_history = []
            self.timecount = 0
            self.detected_classes = []
        
        # store the record of recent detect, max 5 times
        if len(self.detected_classes) > 5:
            self.detected_classes.pop(0)
        
        print("detected_classes: ", self.detected_classes)
        # if all the detection are the same, then insert to self.classhistory
        #situation 1: just start, the classhistory is empty
        if len(self.class_history)==0 and len(self.detected_classes)==5:
            if all(item == self.detected_classes[0] for item in self.detected_classes):
                self.class_history.append(self.detected_classes[0])
        #situation 2: the class history is not empty, checkt the first position class
            # if same with the detection class, then ignore
        elif len(self.class_history)>0 and len(self.detected_classes)==5:
            if all(item == self.detected_classes[0] for item in self.detected_classes) and self.class_history[-1] != self.detected_classes[0]:
                self.class_history.append(self.detected_classes[0])
        # store the class history, max 50 classes
        if len(self.class_history) > 10:
            self.class_history.pop(0)
        
    
        # for i in range(len(self.class_history) - 2):
        #         if all(item in self.class_history[i:i+3] for item in self.pattern_to_detect):
        #             # Pattern detected, draw "ninja" text on the frame
        #             cv2.putText(frame, "NINJA", (50, 50),
        #                         cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 2)
        #             break
        for pattern in self.patterns_to_detect:
            pattern_detected = False
            for i in range(len(self.class_history) - len(pattern) + 1):
                if self.class_history[i:i+len(pattern)] == pattern:
                    # Pattern detected, draw pattern_name text on the frame
                    cv2.putText(frame, self.patterns_name[self.patterns_to_detect.index(pattern)], (50, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 2)
                    playVideo(self.patterns_name[self.patterns_to_detect.index(pattern)]+ '.mp4')
                    pattern_detected = True
                    self.class_history = []
                    break
            if pattern_detected:
                break  # Exit the outer loop
        self.class_history_listbox.delete(0, tk.END)
        for item in self.class_history:
            self.class_history_listbox.insert(tk.END, item)

        print('class_history:', self.class_history)

        # Convert the frame to ImageTk format
        img = Image.fromarray(frame)
        imgtk = ImageTk.PhotoImage(image=img)

        # Update the video label
        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk)

        # Call update_video recursively after 10 ms
        self.window.after(10, self.update_video)

# Create the main window
root = tk.Tk()

# Create the ObjectDetectionApp instance
app = ObjectDetectionApp(root)

# Run the Tkinter event loop
root.mainloop()
