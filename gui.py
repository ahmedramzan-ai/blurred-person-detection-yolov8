import customtkinter as ctk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
from ultralytics import YOLO
import cv2
import threading
import os

# -----------------------------
# SETTINGS
# -----------------------------


ROOT_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

BLUR_MODEL_PATH = os.path.join(
    ROOT_DIR,
    "models",
    "blur_finetuned.pt"
)

# -----------------------------
# GUI
# -----------------------------

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

app = ctk.CTk()
app.title("Blurred Person Detection using YOLOv8")
app.geometry("900x650")

selected_file = None

# -----------------------------
# TITLE
# -----------------------------

title = ctk.CTkLabel(
    app,
    text="Blurred Person Detection using YOLOv8",
    font=("Arial", 24, "bold")
)
title.pack(pady=20)

# -----------------------------
# MODEL SELECTION
# -----------------------------

model_var = ctk.StringVar(value="Blur-Finetuned YOLOv8")

model_menu = ctk.CTkOptionMenu(
    app,
    values=[
        "Blur-Finetuned YOLOv8",
        "COCO YOLOv8"
    ],
    variable=model_var
)

model_menu.pack(pady=10)

# -----------------------------
# CONFIDENCE
# -----------------------------

conf_label = ctk.CTkLabel(app, text="Confidence: 0.25")
conf_label.pack()

def update_conf(value):
    conf_label.configure(
        text=f"Confidence: {float(value):.2f}"
    )

conf_slider = ctk.CTkSlider(
    app,
    from_=0.1,
    to=1.0,
    number_of_steps=18,
    command=update_conf
)

conf_slider.set(0.25)
conf_slider.pack(pady=10)

# -----------------------------
# FILE STATUS
# -----------------------------

file_label = ctk.CTkLabel(
    app,
    text="No file selected"
)
file_label.pack()

# -----------------------------
# IMAGE PREVIEW
# -----------------------------

preview_label = ctk.CTkLabel(
    app,
    text=""
)

preview_label.pack(pady=10)

# -----------------------------
# DETECTION COUNT
# -----------------------------

count_label = ctk.CTkLabel(
    app,
    text="Detections: 0",
    font=("Arial", 18)
)

count_label.pack()

# -----------------------------
# STATUS
# -----------------------------

status_label = ctk.CTkLabel(
    app,
    text="Ready"
)

status_label.pack(pady=10)

# -----------------------------
# LOAD MODEL
# -----------------------------

def get_model():

    selected_model = model_var.get()

    if selected_model == "Blur-Finetuned YOLOv8":
        return YOLO(BLUR_MODEL_PATH)

    else:
        return YOLO("yolov8n.pt")

# -----------------------------
# UPLOAD IMAGE
# -----------------------------

def upload_image():

    global selected_file

    file_path = filedialog.askopenfilename(
        filetypes=[
            ("Images", "*.jpg *.jpeg *.png")
        ]
    )

    if file_path:

        selected_file = file_path

        file_label.configure(
            text=os.path.basename(file_path)
        )

        img = Image.open(file_path)

        img.thumbnail((400, 300))

        photo = ImageTk.PhotoImage(img)

        preview_label.configure(
            image=photo,
            text=""
        )

        preview_label.image = photo

# -----------------------------
# UPLOAD VIDEO
# -----------------------------

def upload_video():

    global selected_file

    file_path = filedialog.askopenfilename(
        filetypes=[
            ("Videos", "*.mp4 *.avi *.mov")
        ]
    )

    if file_path:

        selected_file = file_path

        file_label.configure(
            text=os.path.basename(file_path)
        )

        preview_label.configure(
            image=None,
            text="Video Selected"
        )

# -----------------------------
# IMAGE DETECTION
# -----------------------------

def detect_image():

    status_label.configure(
        text="Running Image Detection..."
    )

    model = get_model()

    conf = conf_slider.get()

    results = model.predict(
        selected_file,
        conf=conf,
        verbose=False
    )

    detections = len(results[0].boxes)

    count_label.configure(
        text=f"Detections: {detections}"
    )

    result_img = results[0].plot()

    result_img = cv2.cvtColor(
        result_img,
        cv2.COLOR_BGR2RGB
    )

    img = Image.fromarray(result_img)

    img.thumbnail((500, 350))

    photo = ImageTk.PhotoImage(img)

    preview_label.configure(
        image=photo,
        text=""
    )

    preview_label.image = photo

    status_label.configure(
        text="Done"
    )

# -----------------------------
# VIDEO DETECTION
# -----------------------------

def detect_video():

    status_label.configure(
        text="Running Video Detection..."
    )

    model = get_model()

    conf = conf_slider.get()

    cap = cv2.VideoCapture(selected_file)

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        results = model(
            frame,
            conf=conf
        )

        detections = len(
            results[0].boxes
        )

        count_label.configure(
            text=f"Detections: {detections}"
        )

        annotated = results[0].plot()

        cv2.imshow(
            "YOLO Detection",
            annotated
        )

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    status_label.configure(
        text="Done"
    )

# -----------------------------
# RUN DETECTION
# -----------------------------

def run_detection():

    if not selected_file:

        messagebox.showerror(
            "Error",
            "Please select a file first."
        )

        return

    ext = os.path.splitext(
        selected_file
    )[1].lower()

    if ext in [".jpg", ".jpeg", ".png"]:

        threading.Thread(
            target=detect_image
        ).start()

    else:

        threading.Thread(
            target=detect_video
        ).start()

# -----------------------------
# BUTTONS
# -----------------------------

btn_frame = ctk.CTkFrame(app)
btn_frame.pack(pady=20)

img_btn = ctk.CTkButton(
    btn_frame,
    text="Upload Image",
    command=upload_image
)

img_btn.grid(
    row=0,
    column=0,
    padx=10
)

video_btn = ctk.CTkButton(
    btn_frame,
    text="Upload Video",
    command=upload_video
)

video_btn.grid(
    row=0,
    column=1,
    padx=10
)

run_btn = ctk.CTkButton(
    app,
    text="Run Detection",
    height=40,
    command=run_detection
)

run_btn.pack(pady=20)

# -----------------------------
# START
# -----------------------------

app.mainloop()
