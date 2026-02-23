[main.py](https://github.com/user-attachments/files/25496646/main.py)
from tkinter import Tk, Label, Button, filedialog, Text, Frame
from tkinter.ttk import Combobox, Style
from PIL import Image, ImageTk
import cv2
from ultralytics import YOLO

class ObjectDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("YOLOv8 Nesne Algılama Uygulaması")
        self.root.geometry("1400x800")
        self.root.configure(bg="#1e1e2f")

        # Stilleri tanımla
        style = Style()
        style.configure("TButton", font=("Arial", 12), padding=5)

        # Ana düzen çerçeveleri
        self.left_frame = Frame(root, bg="#2e2e3f")
        self.left_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

        self.right_frame = Frame(root, bg="#2e2e3f")
        self.right_frame.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")

        # Grid düzeni ayarları
        root.grid_columnconfigure(0, weight=4)  # Sol çerçeve için ağırlık artırıldı
        root.grid_columnconfigure(1, weight=1)  # Sağ çerçeve için ağırlık azaldı
        root.grid_rowconfigure(0, weight=1)

        # Sol çerçeve: Görüntü alanı
        self.image_label = Label(self.left_frame, bg="#3e3e4f", relief="solid")
        self.image_label.pack(padx=10, pady=10, fill="both", expand=True)

        # Sağ çerçeve: Kontroller ve sonuçlar
        Label(self.right_frame, text="YOLOv8 Nesne Algılama", font=("Arial", 16, "bold"), bg="#2e2e3f",
              fg="#ffffff").pack(pady=20)

        # Dosya seçme butonu
        self.select_button = Button(self.right_frame, text="Görüntü/Video Seç", command=self.select_file,
                                    font=("Arial", 12), bg="#4caf50", fg="#ffffff", activebackground="#388e3c")
        self.select_button.pack(pady=10, fill="x", padx=20)

        # Algılama butonu
        self.detect_button = Button(self.right_frame, text="Nesne Algıla", command=self.detect_objects,
                                    font=("Arial", 12), bg="#2196f3", fg="#ffffff", activebackground="#1976d2")
        self.detect_button.pack(pady=10, fill="x", padx=20)

        # Temizleme butonu
        self.clear_button = Button(self.right_frame, text="Temizle", command=self.clear, font=("Arial", 12),
                                   bg="#f44336", fg="#ffffff", activebackground="#d32f2f")
        self.clear_button.pack(pady=10, fill="x", padx=20)

        # Kaydetme butonu
        self.save_button = Button(self.right_frame, text="Kaydet", command=self.save_image,
                                  font=("Arial", 12), bg="#673ab7", fg="#ffffff", activebackground="#512da8")
        self.save_button.pack(pady=10, fill="x", padx=20)

        # Model seçimi
        Label(self.right_frame, text="Model Seç:", font=("Arial", 12, "bold"), bg="#2e2e3f", fg="#ffffff").pack(pady=5)
        self.model_combo = Combobox(self.right_frame,
                                    values=["yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt"],
                                    state="readonly")
        self.model_combo.set("yolov8n.pt")
        self.model_combo.pack(pady=5, padx=20, fill="x")

        # Sonuçlar metin alanı
        self.text_area = Text(self.right_frame, height=15, bg="#27293d", fg="#76c7c0", font=("Consolas", 12))
        self.text_area.pack(pady=10, padx=20, fill="both", expand=True)

        # Dosya yolu ve sonuçlar
        self.file_path = None
        self.result_image = None
        self.is_video = False
        self.results = None
        self.cap = None
        self.frame = None
        self.out = None
        self.video_out_path = None

    def select_file(self):
        self.file_path = filedialog.askopenfilename(
            title="Görüntü veya Video Seç",
            filetypes=(("Görüntü ve Video Dosyaları", "*.jpg *.jpeg *.png *.mp4"), ("Tüm Dosyalar", "*.*"))
        )
        if self.file_path:
            if self.file_path.endswith(('.mp4', '.avi', '.mov')):
                self.is_video = True
                self.image_label.config(image=None)
                self.image_label.image = None
            else:
                self.is_video = False
                image = Image.open(self.file_path)
                image.thumbnail((1200, 700))
                photo = ImageTk.PhotoImage(image)
                self.image_label.config(image=photo)
                self.image_label.image = photo

            if self.is_video:
                self.save_button.config(command=self.save_video)
            else:
                self.save_button.config(command=self.save_image)

    def detect_objects(self):
        if not self.file_path:
            return

        model_path = self.model_combo.get()
        model = YOLO(model_path)

        if self.is_video:
            self.detect_objects_in_video(model)
        else:
            self.detect_objects_in_image(model)

    def detect_objects_in_image(self, model):
        self.results = model(self.file_path)
        annotated_image = self.results[0].plot()
        cv_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
        self.result_image = Image.fromarray(cv_image)

        # Görüntüyü sol çerçeveye sığacak şekilde yeniden boyutlandır
        left_frame_width = self.left_frame.winfo_width()
        left_frame_height = self.left_frame.winfo_height()
        self.result_image.thumbnail((left_frame_width, left_frame_height), Image.Resampling.LANCZOS)

        photo = ImageTk.PhotoImage(self.result_image)
        self.image_label.config(image=photo)
        self.image_label.image = photo

        self.text_area.delete(1.0, "end")
        detections = self.results[0].boxes
        self.text_area.insert("end", f"Algılanan Nesneler: {len(detections)}\n")
        for box in detections:
            class_name = self.results[0].names[int(box.cls[0])]
            confidence = box.conf[0]
            self.text_area.insert("end", f"{class_name}: {confidence:.2f}\n")

    def detect_objects_in_video(self, model):
        if self.cap:
            self.cap.release()

        self.cap = cv2.VideoCapture(self.file_path)
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(self.cap.get(cv2.CAP_PROP_FPS))

        # Video kaydetmek için ayar
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.video_out_path = filedialog.asksaveasfilename(defaultextension=".mp4",
                                                           filetypes=[("MP4 Dosyaları", "*.mp4"),
                                                                      ("Tüm Dosyalar", "*.*")])
        if self.video_out_path:
            self.out = cv2.VideoWriter(self.video_out_path, fourcc, fps, (width, height))

        def process_frame():
            ret, frame = self.cap.read()
            if not ret:
                self.cap.release()
                if self.out:
                    self.out.release()
                return

            results = model(frame)
            annotated_frame = results[0].plot()

            # Video çerçevesini göster
            cv_image = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(cv_image)
            pil_image.thumbnail((self.left_frame.winfo_width(), self.left_frame.winfo_height()), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(pil_image)

            self.image_label.config(image=photo)
            self.image_label.image = photo

            # Video kaydı
            if self.out:
                self.out.write(annotated_frame)

            # Sonraki kareyi işlemek için tekrar çağır
            self.root.after(int(1000 / fps), process_frame)

        process_frame()

    def save_image(self):
        if not self.result_image:
            return

        save_path = filedialog.asksaveasfilename(defaultextension=".png",
                                                 filetypes=[("PNG Dosyaları", "*.png"), ("JPEG Dosyaları", "*.jpg"),
                                                            ("Tüm Dosyalar", "*.*")])
        if save_path:
            self.result_image.save(save_path)

    def save_video(self):
        if self.out:
            self.out.release()

    def clear(self):
        self.file_path = None
        self.result_image = None
        self.image_label.config(image=None)
        self.image_label.image = None
        self.text_area.delete(1.0, "end")

        if self.cap:
            self.cap.release()
            self.cap = None

        if self.out:
            self.out.release()
            self.out = None

if __name__ == "__main__":
    root = Tk()
    app = ObjectDetectionApp(root)
    root.mainloop()
