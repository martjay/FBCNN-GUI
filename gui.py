import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os
from PIL import Image, ImageTk
import torch
import numpy as np
import cv2
from models.network_fbcnn import FBCNN as net
import threading
import shutil
import logging  # Import logging module

try:
    import customtkinter as ctk
except ImportError:
    print("Please install the 'customtkinter' library using: pip install customtkinter")
    exit()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ProcessedImageWidget(tk.Frame):
    def __init__(self, master, processed_image, gui_instance, *args, **kwargs):
        super().__init__(master, *args, **kwargs)
        self.processed_image = processed_image
        self.gui_instance = gui_instance  # Store the FBCNNGUI instance
        
        self.canvas = tk.Canvas(self, bg=self.gui_instance.colors["bg_100"], highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        self.canvas.bind("<Configure>", self.resize)
        
        self.draw_image()
        
    def resize(self, event):
        self.canvas_width = event.width
        self.canvas_height = event.height
        self.draw_image()
        
    def draw_image(self):
        if not hasattr(self, 'canvas_width') or not self.processed_image:
            return
        
        self.canvas.delete("all")
        
        # Resize image to fit the canvas while maintaining aspect ratio
        img_width, img_height = self.processed_image.size
        scale = min(self.canvas_width/img_width, self.canvas_height/img_height)
        
        new_width = int(img_width * scale)
        new_height = int(img_height * scale)
        
        resized_image = self.processed_image.resize((new_width, new_height), Image.LANCZOS)
        self.tk_image = ImageTk.PhotoImage(resized_image)
        
        # Calculate position to center the image
        x = (self.canvas_width - new_width) // 2
        y = (self.canvas_height - new_height) // 2
        
        self.canvas.create_image(x, y, anchor=tk.NW, image=self.tk_image)
        
        # Add label
        self.canvas.create_text(10, 10, anchor=tk.NW, text="Processed", fill=self.gui_instance.colors["text_100"], font=("Arial", 12, "bold"))

class ImageComparisonWidget(tk.Frame):
    def __init__(self, master, original_image, processed_image, gui_instance, *args, **kwargs):
        super().__init__(master, *args, **kwargs)
        self.original_image = original_image
        self.processed_image = processed_image
        self.gui_instance = gui_instance
        self.current_image = self.processed_image  # Start with processed image
        
        self.canvas = tk.Canvas(self, bg=self.gui_instance.colors["bg_100"], highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        self.canvas.bind("<Configure>", self.resize)
        
        self.draw_image()
        
        self.is_original = False  # Track if showing original image
        
    def resize(self, event):
        self.canvas_width = event.width
        self.canvas_height = event.height
        self.draw_image()
        
    def draw_image(self):
        if not hasattr(self, 'canvas_width') or not self.current_image:
            return
        
        self.canvas.delete("all")
        
        # Resize image to fit the canvas while maintaining aspect ratio
        img_width, img_height = self.current_image.size
        scale = min(self.canvas_width/img_width, self.canvas_height/img_height)
        
        new_width = int(img_width * scale)
        new_height = int(img_height * scale)
        
        resized_image = self.current_image.resize((new_width, new_height), Image.LANCZOS)
        self.tk_image = ImageTk.PhotoImage(resized_image)
        
        # Calculate position to center the image
        x = (self.canvas_width - new_width) // 2
        y = (self.canvas_height - new_height) // 2
        
        self.canvas.create_image(x, y, anchor=tk.NW, image=self.tk_image)
        
        # Add label
        label_text = "Original" if self.is_original else "Processed"
        self.canvas.create_text(10, 10, anchor=tk.NW, text=label_text, fill=self.gui_instance.colors["text_100"], font=("Arial", 12, "bold"))
        
    def toggle_image(self):
        self.is_original = not self.is_original
        self.current_image = self.original_image if self.is_original else self.processed_image
        self.draw_image()

class FBCNNGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("FBCNN - JPEG Artifact Removal")
        self.root.geometry("1280x800")
        
        # Define color scheme
        self.colors = {
            "primary_100": "#8FBF9F",
            "primary_200": "#68a67d",
            "primary_300": "#24613b",
            "accent_100": "#F18F01",
            "accent_200": "#833500",
            "text_100": "#353535",
            "text_200": "#5f5f5f",
            "bg_100": "#F5ECD7",
            "bg_200": "#ebe2cd",
            "bg_300": "#c2baa6"
        }
        
        # Apply the color scheme
        self.style = ttk.Style()
        self.style.theme_use('default')
        
        # Configure colors
        self.style.configure("TFrame", background=self.colors["bg_100"])
        self.style.configure("TLabel", background=self.colors["bg_100"], foreground=self.colors["text_100"])
        self.style.configure("TButton", background=self.colors["primary_200"], foreground=self.colors["text_100"])
        self.style.map("TButton", background=[("active", self.colors["primary_300"])])
        self.style.configure("TCombobox", fieldbackground=self.colors["bg_200"], foreground=self.colors["text_100"])
        self.style.configure("Horizontal.TScale", background=self.colors["bg_100"], troughcolor=self.colors["bg_300"])
        self.style.configure("Horizontal.TProgressbar", background=self.colors["accent_100"])
        
        self.input_paths = []
        self.model = None
        self.current_model_type = None
        self.original_image = None
        self.processed_image = None
        self.current_image_index = 0
        
        self.cache_dir = "fbcnn_cache"
        os.makedirs(self.cache_dir, exist_ok=True)
        
        self.device = "cpu"  # Default to CPU
        if torch.cuda.is_available():
            self.device = "cuda"  # Default to GPU if available
            logging.info("CUDA detected, defaulting to GPU for processing.")
        else:
            logging.info("CUDA not detected, defaulting to CPU for processing.")
        
        self.create_widgets()

    def create_widgets(self):
        self.main_frame = ttk.Frame(self.root, padding="10", style="TFrame")
        self.main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        
        # Image frame
        self.image_frame = ttk.Frame(self.main_frame, style="TFrame")
        self.image_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Control frame
        self.control_frame = ttk.LabelFrame(self.main_frame, text=" ", padding="5", style="TFrame")
        self.control_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=10)
        
        # Model selection
        ttk.Label(self.control_frame, text="Select Model:", style="TLabel").grid(row=0, column=0, padx=5)
        self.model_var = tk.StringVar()
        self.model_combo = ttk.Combobox(self.control_frame, textvariable=self.model_var, style="TCombobox")
        self.model_combo['values'] = self.get_available_models()
        self.model_combo.grid(row=0, column=1, padx=5)
        self.model_combo.bind('<<ComboboxSelected>>', self.load_model)

        # Device selection
        ttk.Label(self.control_frame, text="Select Device:", style="TLabel").grid(row=0, column=2, padx=5)
        self.device_var = tk.StringVar(value=self.device)
        self.device_combo = ttk.Combobox(self.control_frame, textvariable=self.device_var, values=["CPU", "GPU"], style="TCombobox")
        self.device_combo.grid(row=0, column=3, padx=5)
        self.device_combo.bind('<<ComboboxSelected>>', self.set_device)

        # Buttons
        ctk.CTkButton(self.control_frame, text="Import Images", command=self.load_images).grid(row=0, column=4, padx=5)
        ctk.CTkButton(self.control_frame, text="Start Processing", command=self.process_all_images).grid(row=0, column=5, padx=5)
        
        # JPEG Quality slider
        ttk.Label(self.control_frame, text="JPEG Quality:", style="TLabel").grid(row=1, column=2, padx=5, pady=5)
        self.quality_var = tk.IntVar(value=50)
        self.quality_slider = ttk.Scale(self.control_frame, from_=1, to=100, orient=tk.HORIZONTAL, variable=self.quality_var, style="Horizontal.TScale")
        self.quality_slider.grid(row=1, column=3, padx=5, pady=5)
        
        # Save All button
        ctk.CTkButton(self.control_frame, text="Save Results", command=self.save_all_results).grid(row=1, column=4, padx=5, pady=5)
        
        # File list
        self.file_list_frame = ttk.LabelFrame(self.main_frame, text="File List", padding="5", style="TFrame")
        self.file_list_frame.grid(row=0, column=2, rowspan=3, sticky=(tk.N, tk.S, tk.E, tk.W), padx=10)
        
        self.file_listbox = tk.Listbox(self.file_list_frame, bg=self.colors["bg_200"], fg=self.colors["text_100"])
        self.file_listbox.pack(fill=tk.BOTH, expand=True)
        self.file_listbox.bind('<<ListboxSelect>>', self.on_file_select)

        # Navigation buttons
        ctk.CTkButton(self.control_frame, text="Previous", command=self.show_previous_image).grid(row=3, column=2, padx=5, pady=5)
        ctk.CTkButton(self.control_frame, text="Next", command=self.show_next_image).grid(row=3, column=3, padx=5, pady=5)
        
        # Clear buttons
        ctk.CTkButton(self.control_frame, text="Clear List", command=self.clear_file_list).grid(row=1, column=0, padx=5, pady=5)
        ctk.CTkButton(self.control_frame, text="Clear Cache", command=self.clear_cache).grid(row=1, column=1, padx=5, pady=5)
        
        # Progress bar
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(self.control_frame, variable=self.progress_var, maximum=100, style="Horizontal.TProgressbar")
        self.progress_bar.grid(row=2, column=0, columnspan=6, sticky=(tk.W, tk.E), padx=5, pady=5)
        
        # Toggle image button
        ctk.CTkButton(self.control_frame, text="Compare", command=self.toggle_image_display).grid(row=3, column=4, padx=5, pady=5)
        
        # Configure grid weights
        self.main_frame.columnconfigure(0, weight=3)
        self.main_frame.columnconfigure(1, weight=3)
        self.main_frame.columnconfigure(2, weight=1)
        self.main_frame.rowconfigure(0, weight=1)

    def get_available_models(self):
        model_dir = 'model_zoo'
        return [f for f in os.listdir(model_dir) if f.endswith('.pth')]

    def load_model(self, event=None):
        selected_model = self.model_var.get()
        logging.info(f"Loading model: {selected_model}")
        if not selected_model:
            messagebox.showwarning("Warning", "Please select a model first!")
            return
        
        try:
            model_path = os.path.join('model_zoo', selected_model)
            if 'gray' in selected_model:
                self.model = net(in_nc=1, out_nc=1)
                self.current_model_type = 'gray'
            else:
                self.model = net(in_nc=3, out_nc=3)
                self.current_model_type = 'color'
            
            self.model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'), weights_only=True))
            self.model.eval()
            if self.device == "cuda" and torch.cuda.is_available():
                self.model.to(self.device)  # Move model to selected device
            messagebox.showinfo("Done", f"Model {selected_model} loaded successfully!")
            logging.info(f"Model {selected_model} loaded successfully!")
        except Exception as e:
            messagebox.showerror("Error", f"Error loading model: {str(e)}")
            logging.error(f"Error loading model: {str(e)}")

    def set_device(self, event=None):
        selected_device = self.device_var.get()
        if selected_device == "GPU":
            if torch.cuda.is_available():
                self.device = "cuda"
            else:
                self.device = "cpu"
                messagebox.showwarning("Warning", "CUDA is not available, using CPU for processing.")
        else:
            self.device = "cpu"
        logging.info(f"Device set to: {self.device}")
        if self.model and torch.cuda.is_available():
            self.model.to(self.device)

    def load_images(self):
        logging.info("Loading images...")
        file_paths = filedialog.askopenfilenames(filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff *.gif")])
        if file_paths:
            self.input_paths.extend(file_paths)
            self.update_file_list()
            self.load_image(self.input_paths[0])
            logging.info(f"{len(file_paths)} images loaded.")
        else:
            logging.info("Image loading cancelled.")

    def update_file_list(self):
        self.file_listbox.delete(0, tk.END)
        for path in self.input_paths:
            self.file_listbox.insert(tk.END, os.path.basename(path))

    def on_file_select(self, event):
        selection = self.file_listbox.curselection()
        if selection:
            index = selection[0]
            self.load_image(self.input_paths[index])

    def load_image(self, path):
        try:
            self.original_image = Image.open(path)
            cached_path = os.path.join(self.cache_dir, f"processed_{os.path.basename(path)}")
            if os.path.exists(cached_path):
                self.processed_image = Image.open(cached_path)
            else:
                self.processed_image = self.original_image.copy()
            self.current_image_index = self.input_paths.index(path)
            self.update_image_widget()
        except Exception as e:
            messagebox.showerror("Error", f"Error opening image: {str(e)}")
            logging.error(f"Error opening image: {str(e)}")

    def process_all_images(self):
        if not self.input_paths:
            messagebox.showwarning("Warning", "No images opened!")
            return
        
        if self.model is None:
            messagebox.showwarning("Warning", "Please load a model first!")
            return
        
        self.progress_var.set(0)
        self.root.update_idletasks()
        
        def process_thread():
            total_images = len(self.input_paths)
            logging.info(f"Starting to process all {total_images} images...")
            for i, path in enumerate(self.input_paths):
                logging.info(f"Processing image {i + 1}/{total_images}: {os.path.basename(path)}")
                self.load_image(path)
                self.process_image()
                progress = (i + 1) / total_images * 100
                self.progress_var.set(progress)
                self.root.update_idletasks()
                logging.info(f"Image {os.path.basename(path)} processed.")
            logging.info("All images processed.")
        
        threading.Thread(target=process_thread, daemon=True).start()

    def process_image(self):
        if self.original_image is None:
            messagebox.showwarning("Warning", "Please open an image first!")
            return
        
        if self.model is None:
            messagebox.showwarning("Warning", "Please load a model first!")
            return
            
        try:
            img = np.array(self.original_image)
            if self.current_model_type == 'gray':
                img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                img = img[np.newaxis, ...]
            else:
                img = img.transpose((2, 0, 1))  # HWC to CHW
            
            img = img.astype(np.float32) / 255.
            img = torch.from_numpy(img).unsqueeze(0).to(self.device) # Move input to selected device
            
            with torch.no_grad():
                output = self.model(img)

            if isinstance(output, tuple):
                output = output[0] 
 
            output = output.squeeze().cpu().numpy()
            output = np.clip(output, 0, 1) * 255
            
            if self.current_model_type == 'gray':
                output = np.repeat(output[..., np.newaxis], 3, axis=2)
            else:
                output = output.transpose((1, 2, 0))
            
            self.processed_image = Image.fromarray(output.astype(np.uint8))
            self.update_image_widget()
            
            # Save processed image to cache
            cache_path = os.path.join(self.cache_dir, f"processed_{os.path.basename(self.input_paths[self.current_image_index])}")
            self.processed_image.save(cache_path)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to process image: {str(e)}")
            logging.error(f"Failed to process image: {str(e)}")
            print(f"Error details: {str(e)}")

    def update_image_widget(self):
        for widget in self.image_frame.winfo_children():
            widget.destroy()
        
        if self.processed_image and self.original_image:
            self.image_widget = ImageComparisonWidget(self.image_frame, self.original_image, self.processed_image, self)
            self.image_widget.pack(fill=tk.BOTH, expand=True)

    def save_all_results(self):
        if not self.input_paths:
            messagebox.showwarning("Warning", "No images have been processed!")
            return
        
        save_dir = filedialog.askdirectory()
        if save_dir:
            logging.info(f"Saving all images to: {save_dir}")
            for path in self.input_paths:
                cached_path = os.path.join(self.cache_dir, f"processed_{os.path.basename(path)}")
                if os.path.exists(cached_path):
                    shutil.copy(cached_path, os.path.join(save_dir, f"processed_{os.path.basename(path)}"))
                    logging.info(f"Saved: {os.path.basename(path)}")
            messagebox.showinfo("Done", "All images have been processed!")
            logging.info("All images saved.")
        else:
            logging.info("Save all images cancelled.")

    def show_previous_image(self):
        if self.input_paths:
            self.current_image_index = (self.current_image_index - 1) % len(self.input_paths)
            self.load_image(self.input_paths[self.current_image_index])

    def show_next_image(self):
        if self.input_paths:
            self.current_image_index = (self.current_image_index + 1) % len(self.input_paths)
            self.load_image(self.input_paths[self.current_image_index])

    def clear_file_list(self):
        logging.info("Clearing file list.")
        self.input_paths = []
        self.file_listbox.delete(0, tk.END)
        self.original_image = None
        self.processed_image = None
        self.update_image_widget()
        logging.info("File list cleared.")

    def clear_cache(self):
        logging.info("Clearing cache...")
        for file in os.listdir(self.cache_dir):
            file_path = os.path.join(self.cache_dir, file)
            if os.path.isfile(file_path):
                os.unlink(file_path)
        messagebox.showinfo("Done", "Cache cleared!")
        logging.info("Cache cleared.")
        
    def toggle_image_display(self):
        if hasattr(self, 'image_widget') and isinstance(self.image_widget, ImageComparisonWidget):
            self.image_widget.toggle_image()

def main():
    root = ctk.CTk() # Changed to customtkinter
    app = FBCNNGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()