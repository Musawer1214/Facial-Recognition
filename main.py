import cv2
import numpy as np
from insightface.app import FaceAnalysis
import pickle
import os
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import Toplevel
from tkinter import Canvas
from PIL import Image, ImageTk

# Initialize the FaceAnalysis app
app_insight = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
app_insight.prepare(ctx_id=0, det_size=(640, 640))

# Specify the new database file name
database_file = "embedding_database_arcface.pkl"

# Load known embeddings from the new database file or create a new dictionary
try:
    with open(database_file, "rb") as f:
        embedding_database = pickle.load(f)
        if not isinstance(embedding_database, dict):
            embedding_database = {}
        else:
            # Normalize stored embeddings
            for name in embedding_database:
                embeddings = embedding_database[name]
                if isinstance(embeddings, np.ndarray):
                    embeddings = [embeddings]
                normalized_embeddings = [embedding / np.linalg.norm(embedding) for embedding in embeddings]
                embedding_database[name] = normalized_embeddings
except FileNotFoundError:
    embedding_database = {}
    # Create a new database file if it doesn't exist
    with open(database_file, "wb") as f:
        pickle.dump(embedding_database, f)

# Function to recognize the face
def recognize_face(new_embedding, embedding_database, threshold=0.6):  # Adjusted threshold
    if len(embedding_database) == 0:
        return "Unknown"

    max_similarity = -1
    identity = "Unknown"

    # Normalize the new embedding
    new_embedding_norm = new_embedding / np.linalg.norm(new_embedding)

    for name, embeddings in embedding_database.items():
        for idx, stored_embedding in enumerate(embeddings):
            # Compute cosine similarity
            similarity = np.dot(stored_embedding, new_embedding_norm)
            print(f"Comparing with {name} (embedding {idx}), similarity: {similarity}")
            if similarity > threshold and similarity > max_similarity:
                max_similarity = similarity
                identity = name

    print(f"Maximum similarity: {max_similarity}, Identity: {identity}")
    return identity

# Function to display image with detected or recognized names
def display_image_with_name(img, window_title="Image"):
    # Convert image from BGR to RGB for displaying
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)

    # Create a new window to display the image
    top = Toplevel()
    top.title(window_title)
    top.configure(bg="#2C3E50")
    top.geometry("800x600")

    img_label = tk.Label(top, bg="#2C3E50")
    img_label.pack(pady=10)

    img_display = pil_img.copy()
    img_display.thumbnail((800, 600))
    img_tk = ImageTk.PhotoImage(img_display)
    img_label.configure(image=img_tk)
    img_label.image = img_tk

# Function to add a new face to the embedding database
def add_new_face(image_path, name):
    # Load the image from the given path
    img = cv2.imread(image_path)
    if img is None:
        messagebox.showerror("Error", f"Unable to load image at {image_path}")
        return

    # Detect faces and get embeddings using InsightFace
    faces = app_insight.get(img)

    if faces:
        face_added = False
        for face in faces:
            # Get the face embedding and normalize it
            embedding = face.embedding
            embedding_norm = embedding / np.linalg.norm(embedding)

            # Add the new face embedding to the database
            if name not in embedding_database:
                embedding_database[name] = []
            embedding_database[name].append(embedding_norm)
            face_added = True

            # Draw the bounding box and name on the image
            bbox = face.bbox.astype(int)
            x1, y1, x2, y2 = bbox
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        if face_added:
            with open(database_file, "wb") as f:
                pickle.dump(embedding_database, f)
            print(f"Names in database: {list(embedding_database.keys())}")  # Debugging statement
            messagebox.showinfo("Success", f"Face(s) enrolled as '{name}'")

            # Display the image with name(s)
            display_image_with_name(img, f"Enrolled: {name}")
        else:
            messagebox.showerror("Error", "No face detected in the provided image.")
    else:
        messagebox.showerror("Error", "No face detected in the provided image.")

# Function to add a new face using the webcam
def add_new_face_from_camera(name):
    # Use the webcam to capture an image
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        messagebox.showerror("Error", "Unable to access the camera")
        return
    ret, img = cap.read()
    cap.release()
    if not ret:
        messagebox.showerror("Error", "Unable to capture image from camera")
        return

    # Detect faces and get embeddings using InsightFace
    faces = app_insight.get(img)

    if faces:
        face_added = False
        for face in faces:
            # Get the face embedding and normalize it
            embedding = face.embedding
            embedding_norm = embedding / np.linalg.norm(embedding)

            # Add the new face embedding to the database
            if name not in embedding_database:
                embedding_database[name] = []
            embedding_database[name].append(embedding_norm)
            face_added = True

            # Draw the bounding box and name on the image
            bbox = face.bbox.astype(int)
            x1, y1, x2, y2 = bbox
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        if face_added:
            with open(database_file, "wb") as f:
                pickle.dump(embedding_database, f)
            print(f"Names in database: {list(embedding_database.keys())}")  # Debugging statement
            messagebox.showinfo("Success", f"Face(s) enrolled as '{name}'")

            # Display the image with name(s)
            display_image_with_name(img, f"Enrolled: {name}")
        else:
            messagebox.showerror("Error", "No face detected in the captured image.")
    else:
        messagebox.showerror("Error", "No face detected in the captured image.")

# Function to open a file dialog and select an image
def select_image():
    file_path = filedialog.askopenfilename()
    return file_path

# Function to handle adding a new face
def handle_add_face():
    name_window = Toplevel()
    name_window.title("Enter Name")
    name_window.geometry("300x150")
    name_window.configure(bg="#2A3E4C")

    tk.Label(name_window, text="Enter Name:", bg="#2A3E4C", fg="#ECF0F1", font=("Helvetica", 12)).pack(pady=10)
    name_entry = tk.Entry(name_window, font=("Helvetica", 12))
    name_entry.pack(pady=5)

    def handle_name():
        name = name_entry.get().strip()
        if name:
            response = messagebox.askquestion("Add Face", "Do you want to add a face from an image file? Click 'No' to use the camera.")
            if response == 'yes':
                image_path = select_image()
                if image_path:
                    add_new_face(image_path, name)
            else:
                add_new_face_from_camera(name)
        else:
            messagebox.showerror("Error", "Please enter a name for the person.")
        name_window.destroy()

    tk.Button(name_window, text="Submit", command=handle_name, bg="#3498DB", fg="white", font=("Helvetica", 12)).pack(pady=10)

# Function to detect and recognize faces
def handle_recognize_face():
    response = messagebox.askquestion("Recognition Option", "Do you want to recognize a face from an image file? Click 'No' to use the camera.")
    if response == 'yes':
        image_path = select_image()
        if image_path:
            detect_and_recognize_face(image_path)
    else:
        detect_and_recognize_face()

# Function to detect and recognize faces
def detect_and_recognize_face(image_path=None):
    if image_path:
        # Load the image from the given path
        img = cv2.imread(image_path)
        if img is None:
            messagebox.showerror("Error", f"Unable to load image at {image_path}")
            return
    else:
        # Use the webcam to capture an image
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            messagebox.showerror("Error", "Unable to access the camera")
            return
        ret, img = cap.read()
        cap.release()
        if not ret:
            messagebox.showerror("Error", "Unable to capture image from camera")
            return

    # Detect faces and get embeddings using InsightFace
    faces = app_insight.get(img)
    print(f"Number of faces detected: {len(faces)}")  # Debugging statement

    if faces:
        recognition_results = []
        for face in faces:
            # Get the face embedding and normalize it
            embedding = face.embedding
            embedding_norm = embedding / np.linalg.norm(embedding)

            # Recognize the face
            name = recognize_face(embedding_norm, embedding_database)

            recognition_results.append(name)

            # Draw the bounding box and name on the image
            bbox = face.bbox.astype(int)
            x1, y1, x2, y2 = bbox
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img, name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        # Display the image with names
        display_image_with_name(img, "Recognition Results")

        # Show recognition results
        result_message = "\n".join([f"Face {i+1}: {name}" for i, name in enumerate(recognition_results)])
        messagebox.showinfo("Recognition Results", result_message)
    else:
        messagebox.showerror("Error", "No face detected in the image.")

# Create the GUI application
app = tk.Tk()
app.title("Face Recognition System")
app.geometry("600x400")
app.configure(bg="#2A3E4C")

# Header Canvas for Modern Look
header = Canvas(app, height=80, width=600, bg="#1A252F", highlightthickness=0)
header.create_text(300, 40, text="Face Recognition System", fill="#FFFFFF", font=("Helvetica", 20, "bold"))
header.pack()

# UI Elements
frame = tk.Frame(app, bg="#34495E")
frame.pack(pady=20, padx=20, fill="both", expand=True)

add_face_button = tk.Button(frame, text="Add New Face", command=handle_add_face, bg="#3498DB", fg="white", font=("Helvetica", 14, "bold"), activebackground="#2980B9", relief="raised", bd=5)
add_face_button.grid(row=0, column=0, pady=10, padx=10, sticky="w")

recognize_face_button = tk.Button(frame, text="Recognize Face", command=handle_recognize_face, bg="#E74C3C", fg="white", font=("Helvetica", 14, "bold"), activebackground="#C0392B", relief="raised", bd=5)
recognize_face_button.grid(row=1, column=0, pady=10, padx=10, sticky="w")

# Run the application
app.mainloop()
