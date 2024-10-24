from tkinter import Button, Tk, filedialog, messagebox, Text, Scrollbar, VERTICAL, RIGHT, Y, Frame, Label
import numpy as np
import pandas as pd
from minisom import MiniSom
from sklearn.preprocessing import MinMaxScaler
import time
import threading

class KohonenAlgorithm:
    def __init__(self, entries, neurons, iterations, competition_type, text_widget, timer_label, iteration_label, remaining_time_label):
        self.entries = np.array(entries, dtype=float)
        self.text_widget = text_widget
        self.timer_label = timer_label
        self.iteration_label = iteration_label
        self.remaining_time_label = remaining_time_label

        if neurons % 2 != 0:
            raise ValueError("El número de neuronas debe ser par.")

        self.neurons = max(neurons, 2 * self.entries.shape[1])
        self.iterations = iterations
        self.competition_type = competition_type 
        self.som = MiniSom(x=int(np.sqrt(self.neurons)), y=int(np.sqrt(self.neurons)), input_len=self.entries.shape[1], sigma=1.0, learning_rate=0.01)
        
        self.log("CANTIDAD ENTRADAS -------")
        self.log(self.entries.shape[0])
        self.log("CANTIDAD NEURONAS -------")
        self.log(self.neurons)

        self.weights = np.random.uniform(-1, 1, (self.entries.shape[0], self.entries.shape[1]))

        #self.som.random_weights_init(self.weights)
        self.som._weights = self.weights
        self.initial_weights = self.som.get_weights().copy()
        self.dm_history = []
        self.weights_history = []
        self.vencedoras = []

        self.log("PESOS INICIALES -------")
        self.log(self.initial_weights)

    def train(self):
        start_time = time.time()

        for iteration in range(self.iterations):
            current_time = time.time() - start_time
            self.update_timer(current_time, iteration)
            self.log(f"Entrenando iteración {iteration + 1}/{self.iterations}")
            distancias_iteracion = []

            for entry in self.entries:
                winner = self.som.winner(entry)
                winner_weights = self.som.get_weights()[winner[0], winner[1]]
                distance = np.linalg.norm(entry - winner_weights)
                distancias_iteracion.append(distance)
                self.vencedoras.append(distance)

                self.update_weights(entry, winner, iteration)

            self.calculate_dm(distancias_iteracion)
            self.weights_history.append(self.som.get_weights().copy())

        self.final_weights = self.som.get_weights().copy()
        self.log("PESOS FINALES -------")
        self.log(self.final_weights)

        self.show_dm_results()
        self.update_timer(time.time() - start_time, self.iterations, finished=True)

    def refresh_weights_display(self):
        """Muestra los pesos actuales en el Text widget."""
        self.text_widget.delete(1.0, 'end')  # Limpiar el widget de texto
        self.log("PESOS ACTUALES -------")
        self.log(self.som.get_weights())

    def import_weights(self):
        file_path = filedialog.askopenfilename(filetypes=[("Numpy files", "*.npy")])
        if file_path:
            self.weights = np.load(file_path)
            self.som._weights = self.weights  # Asignar directamente a _weights
            messagebox.showinfo("Éxito", "Pesos iniciales cargados correctamente.")
            self.refresh_weights_display()  # Refrescar los pesos en la interfaz

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

    def calculate_dm(self, distancias_iteracion):
        num_patterns = len(distancias_iteracion)
        if num_patterns > 0:
            dm = sum(distancias_iteracion) / num_patterns
        else:
            dm = 0
        self.dm_history.append(dm)
        self.log(f"DM en iteración {len(self.dm_history)}: {round(dm, 4)}")

    def show_dm_results(self):
        average_dm = np.mean(self.dm_history)
        min_dm = np.min(self.dm_history)
        max_dm = np.max(self.dm_history)

        result_message = (f"Resultados de la Distancia Media (DM):\n"
                          f"Promedio de la DM: {average_dm:.10f}\n"
                          f"DM mínima: {min_dm:.10f}\n"
                          f"DM máxima: {max_dm:.10f}")
        messagebox.showinfo("Resultados de la DM", result_message)

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

def run_kohonen(text_widget, timer_label, iteration_label, remaining_time_label):
    entries = load_file()
    if entries is not None and len(entries) > 0:
        neurons = 2
        iterations = 1000
        competition_type = "bland"

        global kohonen_instance
        kohonen_instance = KohonenAlgorithm(entries, neurons, iterations, competition_type, text_widget, timer_label, iteration_label, remaining_time_label)
    else:
        messagebox.showerror("Error", "No se encontraron entradas válidas en el archivo.")

def train_kohonen():
    if 'kohonen_instance' in globals():
        threading.Thread(target=kohonen_instance.train).start()  # Ejecutar en un hilo separado para no bloquear la GUI
    else:
        messagebox.showerror("Error", "Debe cargar los datos primero.")

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
    window_height = 600
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()

    position_top = int(screen_height / 2 - window_height / 2)
    position_right = int(screen_width / 2 - window_width / 2)

    root.geometry(f"{window_width}x{window_height}+{position_right}+{position_top}")

    top_frame = Frame(root)
    top_frame.pack(pady=10)

    timer_label = Label(top_frame, text="Tiempo transcurrido: 00:00", font=("Helvetica", 16))
    timer_label.pack(side='left', padx=10)

    iteration_label = Label(top_frame, text="Iteración: 0/0", font=("Helvetica", 16))
    iteration_label.pack(side='left', padx=10)

    remaining_time_label = Label(top_frame, text="Tiempo restante: 0s", font=("Helvetica", 16))
    remaining_time_label.pack(side='left', padx=10)

    text_frame = Frame(root)
    text_frame.pack(expand=True, fill='both')

    text_widget = Text(text_frame, wrap='word')
    text_widget.pack(side='left', expand=True, fill='both')

    scrollbar = Scrollbar(text_frame, command=text_widget.yview, orient=VERTICAL)
    scrollbar.pack(side=RIGHT, fill=Y)
    text_widget.config(yscrollcommand=scrollbar.set)

    button_frame = Frame(root)
    button_frame.pack(pady=10)

    load_button = Button(button_frame, text="Cargar archivo Excel", command=lambda: run_kohonen(text_widget, timer_label, iteration_label, remaining_time_label))
    load_button.pack(side='left', padx=10)

    train_button = Button(button_frame, text="Entrenar Kohonen", command=train_kohonen)
    train_button.pack(side='left', padx=10)

    export_button = Button(button_frame, text="Exportar Pesos Iniciales", command=lambda: kohonen_instance.export_initial_weights() if 'kohonen_instance' in globals() else messagebox.showerror("Error", "Debe cargar los datos primero."))
    export_button.pack(side='left', padx=10)

    import_button = Button(button_frame, text="Importar Pesos", command=lambda: kohonen_instance.import_weights() if 'kohonen_instance' in globals() else messagebox.showerror("Error", "Debe cargar los datos primero."))
    import_button.pack(side='left', padx=10)

    root.mainloop()

if __name__ == "__main__":
    create_gui()