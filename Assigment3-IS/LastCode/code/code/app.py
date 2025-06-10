import tkinter as tk
from tkinter import ttk, filedialog, Label, Button, Frame, Canvas, Scale, messagebox
from PIL import ImageTk, Image, ImageOps
import torch
import torchvision.models as models
import torch.nn as nn
from torchvision import transforms
import numpy as np
import threading
import queue
import time
import os
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import json

# ========== Constants ==========
MODEL_PATH = r"G:\Group Project\PY CODE\FINALISE CODE 2\data\best_model_final.pth"
IMG_SIZE = 224
TOP_K = 5
BG_COLOR = "#eaf4fc"         # Updated soft blue background
ACCENT_COLOR = "#2a8dd2"     # Richer blue
DARK_COLOR = "#2c3e50"

# ========== Global ==========
model = None
CLASS_NAMES = []
prediction_queue = queue.Queue()
current_results = {}
prediction_in_progress = False

# ========== Load Model ==========
def load_model():
    global model, CLASS_NAMES
    try:
        checkpoint = torch.load(MODEL_PATH, map_location=torch.device('cpu'))
        CLASS_NAMES = checkpoint.get('classes', [])
        model = models.resnet50(weights=None)
        model.fc = nn.Linear(model.fc.in_features, len(CLASS_NAMES))
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        return True
    except Exception as e:
        messagebox.showerror("Model Error", f"Failed to load model: {str(e)}")
        return False

# ========== Image Transformation ==========
def get_transform():
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

# ========== Prediction ==========
def predict_image(image_path):
    try:
        transform = get_transform()
        img = Image.open(image_path).convert("RGB")
        img_tensor = transform(img).unsqueeze(0)
        with torch.no_grad():
            outputs = model(img_tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1)[0]
            top_probs, top_indices = torch.topk(probs, TOP_K)
            top_classes = [CLASS_NAMES[i] for i in top_indices]
            conf_dist = {cls: prob.item() for cls, prob in zip(top_classes, top_probs)}
            return conf_dist, img
    except Exception as e:
        return {"Error": str(e)}, None

def threaded_prediction(file_path):
    try:
        start_time = time.time()
        conf_dist, img = predict_image(file_path)
        prediction_time = time.time() - start_time
        return {
            "conf_dist": conf_dist,
            "image": img,
            "file_path": file_path,
            "prediction_time": prediction_time
        }
    except Exception as e:
        return {"error": str(e)}

# ========== GUI Update ==========
def update_loading_animation(step):
    loading_text = "Predicting" + "." * (step % 4)
    label_result.config(text=loading_text)
    if prediction_in_progress:
        root.after(500, update_loading_animation, step + 1)

def update_results(results):
    global prediction_in_progress, current_results

    if "error" in results:
        label_result.config(text=f"Error: {results['error']}")
        prediction_in_progress = False
        return

    current_results = results
    conf_dist = results["conf_dist"]
    img = results["image"]
    file_path = results["file_path"]
    prediction_time = results["prediction_time"]

    display_img = img.resize((350, 350))
    display_img = ImageOps.expand(display_img, border=2, fill=ACCENT_COLOR)
    img_tk = ImageTk.PhotoImage(display_img)
    panel.config(image=img_tk)
    panel.image = img_tk

    filename = os.path.basename(file_path)
    file_label.config(text=f"File: {filename}", fg=DARK_COLOR)

    top_class = list(conf_dist.keys())[0]
    top_conf = conf_dist[top_class]
    label_result.config(text=f"Predicted Surah: {top_class}\nConfidence: {top_conf:.2%}")

    update_confidence_chart(conf_dist)
    perf_label.config(text=f"Prediction time: {prediction_time:.2f}s | Model: ResNet50")
    export_btn.config(state=tk.NORMAL)
    prediction_in_progress = False

def update_confidence_chart(conf_dist):
    for widget in chart_frame.winfo_children():
        widget.destroy()

    fig = Figure(figsize=(5, 3), dpi=80, facecolor=BG_COLOR)
    ax = fig.add_subplot(111)
    classes = list(conf_dist.keys())
    confidences = [conf_dist[c] for c in classes]
    colors = [(0.16, 0.5, 0.72, 0.7 + 0.3 * i / len(classes)) for i in range(len(classes))]
    y_pos = np.arange(len(classes))
    ax.barh(y_pos, confidences, align='center', color=colors)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(classes)
    ax.invert_yaxis()
    ax.set_xlabel('Confidence')
    ax.set_title('Top Predictions')
    ax.set_xlim([0, 1.0])
    for i, v in enumerate(confidences):
        ax.text(v + 0.01, i, f"{v:.2%}", color='black', va='center')

    canvas = FigureCanvasTkAgg(fig, master=chart_frame)
    canvas.draw()
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

# ========== Handlers ==========
def upload_image():
    global prediction_in_progress
    file_path = filedialog.askopenfilename(
        filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
    )
    if file_path:
        img = Image.open(file_path).resize((350, 350))
        display_img = ImageOps.expand(img, border=2, fill="#cccccc")
        img_tk = ImageTk.PhotoImage(display_img)
        panel.config(image=img_tk)
        panel.image = img_tk
        filename = os.path.basename(file_path)
        file_label.config(text=f"File: {filename}", fg=DARK_COLOR)
        label_result.config(text="")
        perf_label.config(text="")
        for widget in chart_frame.winfo_children():
            widget.destroy()
        export_btn.config(state=tk.DISABLED)
        prediction_in_progress = True
        update_loading_animation(0)
        threading.Thread(
            target=lambda q, path: q.put(threaded_prediction(path)),
            args=(prediction_queue, file_path)
        ).start()
        root.after(100, check_prediction_queue)

def check_prediction_queue():
    try:
        results = prediction_queue.get_nowait()
        update_results(results)
    except queue.Empty:
        if prediction_in_progress:
            root.after(100, check_prediction_queue)

def export_results():
    if not current_results:
        messagebox.showwarning("Export", "No results to export.")
        return
    save_path = filedialog.asksaveasfilename(
        defaultextension=".png",
        filetypes=[("PNG Image", "*.png"), ("All files", "*.*")]
    )
    if not save_path:
        return
    try:
        img = current_results["image"]
        img.save(save_path)
        json_path = os.path.splitext(save_path)[0] + "_results.json"
        with open(json_path, "w") as jf:
            json.dump(current_results["conf_dist"], jf, indent=2)
        messagebox.showinfo("Export", f"Saved image to:\n{save_path}\nand results to:\n{json_path}")
    except Exception as e:
        messagebox.showerror("Export Error", str(e))

# ========== GUI Setup ==========
root = tk.Tk()
root.title("Quran Surah Image Classifier")
root.geometry("1000x800")
root.configure(background=BG_COLOR)
root.minsize(900, 700)

# Styles
style = ttk.Style()
style.configure("TButton", font=("Arial", 12), padding=10)
style.configure("Title.TLabel", font=("Arial", 20, "bold"), background=BG_COLOR, foreground=DARK_COLOR)
style.configure("Result.TLabel", font=("Arial", 14, "bold"), background=BG_COLOR, foreground=DARK_COLOR)

# Header
header = ttk.Label(root, text="Quran Surah Image Classifier", style="Title.TLabel")
header.pack(pady=(10, 20))

# Layout
main_frame = Frame(root, bg=BG_COLOR)
main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)

left_frame = Frame(main_frame, bg=BG_COLOR)
left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)

panel = Label(left_frame, bg="white", bd=2, relief="groove")
panel.pack(pady=(0, 15))

file_label = Label(left_frame, text="No file selected", font=("Arial", 10), bg=BG_COLOR, fg="#666")
file_label.pack()

btn_upload = ttk.Button(left_frame, text="Upload Image", command=upload_image)
btn_upload.pack(pady=10)

export_btn = ttk.Button(left_frame, text="Export Results", command=export_results, state=tk.DISABLED)
export_btn.pack(pady=5)

label_result = Label(left_frame, text="", font=("Arial", 14, "bold"), bg=BG_COLOR, fg=ACCENT_COLOR)
label_result.pack(pady=10)

perf_label = Label(left_frame, text="", font=("Arial", 9), bg=BG_COLOR, fg="#666")
perf_label.pack()

right_frame = Frame(main_frame, bg=BG_COLOR)
right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)

chart_header = Label(right_frame, text="Prediction Confidence", font=("Arial", 14, "bold"), bg=BG_COLOR, fg=DARK_COLOR)
chart_header.pack(anchor="n", pady=(0, 10))

chart_frame = Frame(right_frame, bg="white", bd=2, relief="groove")
chart_frame.pack(fill=tk.BOTH, expand=True)

status_bar = Label(root, text="Ready", bd=1, relief=tk.SUNKEN, anchor=tk.W, 
                  font=("Arial", 9), bg="#e0e0e0", fg="#333")
status_bar.pack(side=tk.BOTTOM, fill=tk.X)

# Load model
if load_model():
    status_bar.config(text=f"Model loaded successfully | {len(CLASS_NAMES)} classes available")
else:
    status_bar.config(text="Model failed to load", fg="red")

root.mainloop()
