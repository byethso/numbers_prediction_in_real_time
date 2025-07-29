
import tkinter as tk
from PIL import Image, ImageOps, ImageFilter
import numpy as np
import torch
#print(f"PyTorch version: {torch.__version__}")
from torch import nn
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
import math


train_mnist_data = MNIST('.', train=True, transform=transforms.ToTensor(), download=True)
test_mnist_data = MNIST('.', train=False, transform=transforms.ToTensor(), download=True)

class NetCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3),  # 1 канал (ч/б), 32 фильтра
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.linear_layers = nn.Sequential(
            nn.Linear(64 * 5 * 5, 128),  # Размер после пулинга: [64, 5, 5]
            nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # Развертка в вектор
        return self.linear_layers(x)

class PixelDrawer:
    def __init__(self, model_filename="mnist_cnn.pth", pixel_size=20, brush_size=1.5, n=28):  # n=28
        self.n = n
        self.pixel_size = pixel_size
        self.canvas_size = n * pixel_size
        self.model_filename = model_filename
        self.brush_size = brush_size

        self.root = tk.Tk()
        self.root.title("Рисовалка цифр")

        self.canvas = tk.Canvas(self.root, width=self.canvas_size, height=self.canvas_size, bg="white")
        self.canvas.pack()

        self.pixels = [[(255, 255, 255) for _ in range(n)] for _ in range(n)]

        self.drawing = False
        self.canvas.bind("<ButtonPress-1>", self.start_drawing)
        self.canvas.bind("<B1-Motion>", self.draw)
        self.canvas.bind("<ButtonRelease-1>", self.stop_drawing)

        save_button = tk.Button(self.root, text="Сохранить", command=self.save_image)
        save_button.pack(pady=10)

        clear_button = tk.Button(self.root, text="Очистить холст", command=self.clear_canvas)
        clear_button.pack(pady=10)

        self.predict_button = tk.Button(self.root, text="Распознать цифру", command=self.predict_digit)
        self.predict_button.pack(pady=10)

        self.prediction_label = tk.Label(self.root, text="Предсказание: ")
        self.prediction_label.pack(pady=5)

        self.model = self.load_model()
        if self.model is None:
            print("Невозможно запустить распознавание. Модель не загружена.")

    def load_model(self):
        try:
            model = NetCNN()
            model.load_state_dict(torch.load(self.model_filename))
            model.eval()
            print(f"Модель загружена из {self.model_filename}")
            return model
        except FileNotFoundError:
            print(f"Ошибка: Файл модели {self.model_filename} не найден.")
            return None

    def predict_digit(self):
        if self.model is None:
            print("Модель не загружена. Невозможно выполнить распознавание.")
            return

        # Преобразование изображения
        img = Image.new("L", (self.n, self.n), "white")  # Создаем ч/б изображение
        img.putdata([sum(pixel) // 3 for row in self.pixels for pixel in row])  # RGB -> Grayscale

        # Предобработка как в MNIST
        img = ImageOps.invert(img)
        img = img.resize((28, 28), Image.Resampling.LANCZOS)
        img = img.filter(ImageFilter.GaussianBlur(radius=0.5))

        # Конвертация в тензор
        img_array = np.array(img).astype('float32') / 255.0
        img_tensor = torch.from_numpy(img_array).float()
        img_tensor = img_tensor.unsqueeze(0).unsqueeze(0)  # [1, 1, 28, 28]

        # Предсказание с вероятностями
        self.model.eval()
        with torch.no_grad():
            output = self.model(img_tensor)
            probabilities = torch.softmax(output, dim=1)[0]

        # Получаем предсказанную цифру и уверенность
        confidence, prediction = torch.max(probabilities, 0)
        confidence = confidence.item() * 100

        # Форматируем текст с цветом в зависимости от уверенности
        color = "green" if confidence > 80 else "orange" if confidence > 50 else "red"

        # Создаем строку с распределением вероятностей
        prob_text = "\n".join([f"{i}: {prob * 100:.1f}%" for i, prob in enumerate(probabilities)])

        self.prediction_label.config(
            text=f"Предсказание: {prediction.item()} (уверенность: {confidence:.1f}%)\n\n{prob_text}",
            fg=color,
            justify=tk.LEFT
        )

        self.model.eval()
        with torch.no_grad():
            output = self.model(img_tensor)
            probabilities = torch.softmax(output, dim=1)[0]  # Применяем softmax для получения вероятностей


        prediction = torch.argmax(probabilities).item()


        prob_text = "\n".join([f"{i}: {prob:.2%}" for i, prob in enumerate(probabilities)])

        self.prediction_label.config(
            text=f"Предсказание: {prediction}\n\nВероятности:\n{prob_text}",
            justify=tk.LEFT
        )
    def start_drawing(self, event):
        self.drawing = True
        self.draw(event)

    def stop_drawing(self, event):
        self.drawing = False

    def draw(self, event):
        if not self.drawing:
            return
        x = event.x // self.pixel_size
        y = event.y // self.pixel_size

        brush_size = 0.6

        # Функция гауссового распределения
        def gaussian(x, y, sigma=brush_size):
            return math.exp(-((x ** 2 + y ** 2) / (2.0 * sigma ** 2)))

        for i in range(-int(brush_size * 2), int(brush_size * 2) + 1):
            for j in range(-int(brush_size * 2), int(brush_size * 2) + 1):
                draw_x = x + i
                draw_y = y + j
                if 0 <= draw_x < self.n and 0 <= draw_y < self.n:
                    # Вычисляем "прозрачность" пикселя
                    distance = gaussian(i, j)

                    # Смешиваем цвет пикселя с черным цветом, учитывая "прозрачность"
                    current_color = self.pixels[draw_y][draw_x]
                    new_color = (
                        int(current_color[0] * (1 - distance)),  # R
                        int(current_color[1] * (1 - distance)),  # G
                        int(current_color[2] * (1 - distance)),  # B
                    )

                    self.pixels[draw_y][draw_x] = new_color

                    # Рисуем прямоугольник
                    self.canvas.create_rectangle(
                        draw_x * self.pixel_size, draw_y * self.pixel_size,
                        (draw_x + 1) * self.pixel_size, (draw_y + 1) * self.pixel_size,
                        fill="#%02x%02x%02x" % new_color,  # Преобразуем цвет в hex-формат
                        outline=""
                    )

        self.canvas.update_idletasks()

        self.predict_digit()

    def save_image(self):
        image = Image.new("RGB", (self.n, self.n))
        flat_pixels = [color for row in self.pixels for color in row]
        image.putdata(flat_pixels)
        image.save("drawing.png")
        print("Изображение сохранено как drawing.png")

    def clear_canvas(self):
        self.pixels = [[(255, 255, 255) for _ in range(self.n)] for _ in range(self.n)]  # сброс сетки
        self.canvas.delete("all")  # очистка холста
        print("Холст очищен")

    def run(self):
        self.root.mainloop()

# Запуск
if __name__ == "__main__":
    drawer = PixelDrawer(model_filename="mnist_cnn.pth", pixel_size=10, brush_size=2.5, n=28)
    drawer.run()
