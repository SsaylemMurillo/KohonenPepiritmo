import sys
from tkinter import END, Button, Tk, filedialog, messagebox, Text, Scrollbar, VERTICAL, RIGHT, Y, Frame, Label, ttk
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import pandas as pd
from minisom import MiniSom
from sklearn.preprocessing import MinMaxScaler
import time
import threading

app_running = True
pause_training = False

stop_event = threading.Event()

class KohonenAlgorithm:
    def __init__(self, entries, neurons, iterations, competition_type, text_widget, timer_label, iteration_label, remaining_time_label, progress_bar, dm_label):
        self.entries = np.array(entries, dtype=float)
        self.text_widget = text_widget
        self.timer_label = timer_label
        self.iteration_label = iteration_label
        self.remaining_time_label = remaining_time_label
        self.progress_bar = progress_bar
        self.dm_label = dm_label

        if neurons % 2 != 0:
            raise ValueError("El número de neuronas debe ser par.")

        self.neurons = max(neurons, 2 * self.entries.shape[1])
        self.iterations = iterations
        self.competition_type = competition_type

        x = int(np.sqrt(self.neurons))
        y = int(np.sqrt(self.neurons))
        input_len = self.entries.shape[1]

        self.som = MiniSom(x=x, y=y, input_len=input_len)

        fake_entries = np.random.uniform(-1, 1, self.entries.shape)

        self.som.random_weights_init(fake_entries)
        self.initial_weights = self.som.get_weights().copy()

        self.log(f"Dimensiones de pesos iniciales: {self.initial_weights.shape[0]}, {self.initial_weights.shape[1]}, {self.initial_weights.shape[2]}")

        self.dm_history = []
        self.weights_history = []
        self.vencedoras = []

        formatted_weights = np.array_str(self.initial_weights, precision=3, suppress_small=True)

        self.log("PESOS INICIALES -------")
        self.log(formatted_weights)
    
    def start_dm_graph(self):
        self.dm_history = []
        self.time_history = []
        self.start_time = time.time()

        fig, ax = plt.subplots()
        ax.set_xlabel("Tiempo (s)")
        ax.set_ylabel("DM")

        def update_graph(i):
            current_time = time.time() - self.start_time
            self.time_history.append(current_time)
            if hasattr(self, 'current_dm'):
                self.dm_history.append(self.current_dm)

            ax.clear()
            ax.set_xlabel("Tiempo (s)")
            ax.set_ylabel("DM")
            ax.plot(self.time_history, self.dm_history, color='blue')

        ani = animation.FuncAnimation(fig, update_graph, interval=5000)
        plt.show()
    
    def train(self):
        start_time = time.time()
        for iteration in range(self.iterations):
            current_time = time.time() - start_time
            if stop_event.is_set():
                break
            while pause_training:
                time.sleep(0.1)
            self.update_timer(current_time, iteration)
            self.update_progress(iteration + 1)

            distancias_iteracion = []

            for entry in self.entries:
                winner = self.som.winner(entry)
                winner_weights = self.som.get_weights()[winner[0], winner[1]]

                distance = np.linalg.norm(entry - winner_weights)
                distancias_iteracion.append(distance)
                self.vencedoras.append(distance)

                self.som.update(entry, winner, iteration, self.iterations)

            self.current_dm = self.calculate_dm(distancias_iteracion)  # Guardar el valor de DM actual
            self.dm_label.config(text=f"DM: {self.current_dm:.4f}")

            if self.current_dm <= 0.01:
                self.log("DM ha alcanzado el umbral de 0.01. Entrenamiento detenido.")
                break

        self.final_weights = self.som.get_weights().copy()

        formatted_weights = np.array_str(self.final_weights, precision=3, suppress_small=True)
        self.log("PESOS FINALES -------")
        self.log(formatted_weights)

        self.update_timer(time.time() - start_time, self.iterations, finished=True)

    def calculate_dm(self, distancias_iteracion):
        return np.mean(distancias_iteracion)

    def refresh_weights_display(self):
        """Muestra los pesos actuales en el Text widget."""
        #self.text_widget.delete(1.0, 'end')  # Limpiar el widget de texto
        self.log("PESOS ACTUALES -------")
        self.log(self.som.get_weights())

    def import_weights(self):
        file_path = filedialog.askopenfilename(filetypes=[("Numpy files", "*.npy")])
        if file_path:
            self.weights = np.load(file_path)
            self.som._weights = self.weights
            messagebox.showinfo("Éxito", "Pesos iniciales cargados correctamente.")
            self.refresh_weights_display()

    def update_weights(self, entry, winner, iteration):
        if self.competition_type == "bland":
            coef_vecindad = 0.2
            weights_shape = self.som.get_weights().shape
            for x in range(weights_shape[0]):
                for y in range(weights_shape[1]):
                    neighbor_distance = np.linalg.norm(np.array([x, y]) - np.array(winner))
                    if neighbor_distance < coef_vecindad:
                        self.som.update(entry, (x, y), iteration, self.iterations)
        else:
            self.som.update(entry, winner, iteration, self.iterations)

    def log(self, message):
        self.text_widget.insert('end', f"{message}\n")
        self.text_widget.see('end')

    def update_timer(self, elapsed_time, iteration, finished=False):
        if not finished:
            remaining_time = (elapsed_time / (iteration + 1)) * (self.iterations - iteration - 1)
            self.iteration_label.config(text=f"Iteración: {iteration + 1}/{self.iterations}")
            self.remaining_time_label.config(text=f"Tiempo restante: {int(remaining_time)}s")
        else:
            self.timer_label.config(text="Entrenamiento finalizado")
            self.remaining_time_label.config(text="Tiempo restante: 0s")

        minutes, seconds = divmod(int(elapsed_time), 60)
        self.timer_label.config(text=f"Tiempo transcurrido: {minutes:02d}:{seconds:02d}")

    def export_initial_weights(self):
        file_path = filedialog.asksaveasfilename(defaultextension=".npy", filetypes=[("Numpy files", "*.npy")])
        if file_path:
            np.save(file_path, self.initial_weights)
            messagebox.showinfo("Éxito", "Pesos iniciales exportados correctamente.")

    def export_final_weights(self):
        file_path = filedialog.asksaveasfilename(defaultextension=".npy", filetypes=[("Numpy files", "*.npy")])
        if file_path:
            np.save(file_path, self.final_weights)
            messagebox.showinfo("Éxito", "Pesos finales exportados correctamente.")

    def update_progress(self, iteration):
        self.progress_bar['value'] = (iteration / self.iterations) * 100

def toggle_pause_training():
    global pause_training
    pause_training = not pause_training

def on_closing(root):
    global app_running
    if messagebox.askokcancel("Salir", "¿Realmente quieres cerrar la aplicación?"):
        stop_event.set()
        app_running = False
        root.quit()
        root.destroy()
        sys.exit()

def run_kohonen(text_widget, timer_label, iteration_label, remaining_time_label, progress_bar, dm_label):
    entries = load_file()
    if entries is not None and len(entries) > 0:
        neurons = 250
        iterations = 10000
        competition_type = "bland"

        global kohonen_instance
        kohonen_instance = KohonenAlgorithm(entries, neurons, iterations, competition_type, text_widget, timer_label, iteration_label, remaining_time_label, progress_bar, dm_label)
        kohonen_instance.refresh_weights_display()
    else:
        messagebox.showerror("Error", "No se encontraron entradas válidas en el archivo.")

def train_kohonen():
    if 'kohonen_instance' in globals():
        threading.Thread(target=kohonen_instance.start_dm_graph).start()
        threading.Thread(target=kohonen_instance.train).start()
    else:
        messagebox.showerror("Error", "Debe cargar los datos primero.")

def clear_all(text_widget, timer_label, iteration_label, remaining_time_label, dm_label, progress_bar):
    text_widget.delete(1.0, END)
    timer_label.config(text="Tiempo transcurrido: 00:00")
    iteration_label.config(text="Iteración: 0/0")
    remaining_time_label.config(text="Tiempo restante: 0s")
    dm_label.config(text="DM: ---")
    progress_bar['value'] = 0

def load_file():
    file_path = filedialog.askopenfilename(filetypes=[("Excel files", "*.xlsx;*.xls")])
    if file_path:
        try:
            df = pd.read_excel(file_path)
            if df.empty:
                messagebox.showerror("Error", "El archivo Excel está vacío o no contiene datos válidos.")
                return None
            numeric_df = df.select_dtypes(include=[np.number])
            if numeric_df.empty:
                messagebox.showerror("Error", "No se encontraron columnas numéricas en el archivo.")
                return None

            entries = numeric_df.values
            print(entries)
            if np.isnan(entries).any():
                messagebox.showerror("Error", "El archivo contiene datos no numéricos o valores faltantes.")
                return None

            messagebox.showinfo("Éxito", "Datos cargados correctamente.")
            return entries
        except Exception as e:
            messagebox.showerror("Error", f"Error al cargar el archivo: {str(e)}")
            return None

def create_gui():
    root = Tk()
    root.title("Algoritmo Kohonen - IA")

    window_width = 800
    window_height = 800
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()

    position_top = int(screen_height / 2 - window_height / 2)
    position_right = int(screen_width / 2 - window_width / 2)

    root.geometry(f"{window_width}x{window_height}+{position_right}+{position_top}")

    top_frame = Frame(root)
    top_frame.pack(pady=10)

    timer_label = Label(top_frame, text="Tiempo transcurrido: 00:00", font=("Helvetica", 12))
    timer_label.pack(side='left')

    iteration_label = Label(top_frame, text="Iteración: 0/0", font=("Helvetica", 12))
    iteration_label.pack(side='left', padx=10)

    remaining_time_label = Label(top_frame, text="Tiempo restante: 0s", font=("Helvetica", 12))
    remaining_time_label.pack(side='left')

    dm_label = Label(top_frame, text="DM: ---", font=("Helvetica", 16, "bold"))
    dm_label.pack()

    text_frame = Frame(root)
    text_frame.pack(fill='both', expand=True, pady=10, padx=10)

    scrollbar = Scrollbar(text_frame, orient=VERTICAL)
    scrollbar.pack(side=RIGHT, fill=Y)

    text_widget = Text(text_frame, wrap='word', yscrollcommand=scrollbar.set)
    text_widget.pack(fill='both', expand=True)

    scrollbar.config(command=text_widget.yview)

    bottom_frame = Frame(root)
    bottom_frame.pack(pady=10)

    progress_bar = ttk.Progressbar(bottom_frame, orient='horizontal', length=200, mode='determinate')
    progress_bar.pack(side='top', padx=0, pady=10)

    Button(bottom_frame, text="Pausar Entrenamiento", command=toggle_pause_training).pack(side='top', padx=10)

    Button(bottom_frame, text="Cargar archivo Excel", command=lambda: run_kohonen(text_widget, timer_label, iteration_label, remaining_time_label, progress_bar, dm_label)).pack(side='left', padx=10)

    Button(bottom_frame, text="Entrenar Kohonen", command=train_kohonen).pack(side='left', padx=10)

    Button(bottom_frame, text="Importar Pesos Iniciales", command=lambda: kohonen_instance.import_weights() if 'kohonen_instance' in globals() else messagebox.showerror("Error", "Debe cargar los datos primero.")).pack(side='left', padx=10)

    Button(bottom_frame, text="Exportar Pesos Iniciales", command=lambda: kohonen_instance.export_initial_weights() if 'kohonen_instance' in globals() else messagebox.showerror("Error", "Debe cargar los datos primero.")).pack(side='left', padx=10)

    Button(bottom_frame, text="Exportar Pesos Finales", command=lambda: kohonen_instance.export_final_weights()).pack(side='left', padx=10)

    Button(bottom_frame, text="Limpiar", command=lambda: clear_all(text_widget, timer_label, iteration_label, remaining_time_label, dm_label, progress_bar)).pack(side='left', padx=10)

    root.protocol("WM_DELETE_WINDOW", lambda: on_closing(root))
    root.mainloop()

if __name__ == "__main__":
    create_gui()