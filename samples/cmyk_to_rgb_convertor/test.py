from tkinter import Tk, ttk, Canvas, StringVar
from pathlib import Path
import time
import numpy as np

from neural_network import NeuralNetwork
from samples.cmyk_to_rgb_convertor.common import rgb_to_hex, cmyk_to_rgb, load_from_file

data_file_name = "data.pkl"


def clamp(value, min_value=0, max_value=255):
    return max(min_value, min(max_value, value))


def validate_input(value):
    # If the entry is empty, return a special flag (or you could choose to return a default value)
    if value.strip() == "":
        return None  # Or return a default value like 0.0
    try:
        val = float(value)
        if 0.0 <= val <= 1.0:
            return val
        else:
            raise ValueError("Value out of range")
    except ValueError:
        return None


def on_update(*args):
    c = validate_input(c_entry.get())
    m = validate_input(m_entry.get())
    y = validate_input(y_entry.get())
    k = validate_input(k_entry.get())

    # Only update if all entries have a valid number; otherwise, do nothing.
    if None in [c, m, y, k]:
        # Optional: you could update to a default color or simply skip the update.
        # For now, we log and set to default black.
        print("Invalid or incomplete input detected! Values must be numbers between 0 and 1.")
        cmyk_box.itemconfig(rectangle_id, fill=rgb_to_hex(0, 0, 0))
        rgb_box.itemconfig(rectangle_rgb, fill=rgb_to_hex(0, 0, 0))
        return

    print(f"Input values - Cyan: {c}, Magenta: {m}, Yellow: {y}, Key: {k}")  # Debugging print

    try:
        # Convert CMYK to RGB
        rgb = cmyk_to_rgb(c, m, y, k)
        rgb_exp_var.set(f"Red: {rgb[0]}  Green: {rgb[1]}  Blue: {rgb[2]}")
        # Convert RGB to HEX and update the rectangle color
        hex_color = rgb_to_hex(*rgb)
        cmyk_box.itemconfig(rectangle_id, fill=hex_color)
        rgb_box.itemconfig(rectangle_rgb,fill=hex_color)
    except Exception as e:
        print(f"Error: {e}")
        cmyk_box.itemconfig(rectangle_id, fill=rgb_to_hex(0, 0, 0))  # Default to black


def convert(*args):
    c = validate_input(args[0])
    m = validate_input(args[1])
    y = validate_input(args[2])
    k = validate_input(args[3])

    if None in [c, m, y, k]:
        print("Invalid input detected in convert! Values must be numbers between 0 and 1.")
        rgb_nn_box.itemconfig(rectangle_rgb_nn, fill=rgb_to_hex(0, 0, 0))  # Default to black
        return

    x_batch = np.array([[c, m, y, k]])
    nn = load_from_file(data_file_name)
    try:
        output = 255 * nn.forward(x_batch)

        r = clamp(int(output[0][0]))
        g = clamp(int(output[0][1]))
        b = clamp(int(output[0][2]))
        rgb_nn_var.set(f"Red: {r}  Green: {g}  Blue: {b}")
        print([r, g, b])
        rgb_nn_box.itemconfig(rectangle_rgb_nn, fill=rgb_to_hex(r, g, b))
    except Exception as e:
        print(f"Error during neural network processing: {e}")
        rgb_nn_box.itemconfig(rectangle_rgb_nn, fill=rgb_to_hex(0, 0, 0))  # Default to black


if __name__ == "__main__":
    if not Path(data_file_name).exists():
        print("Run train.py before testing!")
        time.sleep(3)
    else:
        root = Tk()
        root.title("CMYK to RGB Converter")
        root.geometry("1000x800")
        root.resizable(False, False)

        label1 = ttk.Label(text="Choose CMYK color or generate it randomly", font=("Arial", 25))
        label1.pack()

        # Use StringVar for each entry field to trace changes
        c_var = StringVar(value="1")
        m_var = StringVar(value="1")
        y_var = StringVar(value="1")
        k_var = StringVar(value="1")

        rgb_nn_var = StringVar(value="Red: 0  Green: 0  Blue: 0")
        rgb_exp_var = StringVar(value="Red: 0  Green: 0  Blue: 0")

        c_label = ttk.Label(text="Cyan", font=("Arial", 12))
        c_label.place(x=10, y=170)
        c_entry = ttk.Entry(textvariable=c_var)
        c_entry.place(x=10, y=200, width=70)

        m_label = ttk.Label(text="Magenta", font=("Arial", 12))
        m_label.place(x=100, y=170)
        m_entry = ttk.Entry(textvariable=m_var)
        m_entry.place(x=100, y=200, width=70)

        y_label = ttk.Label(text="Yellow", font=("Arial", 12))
        y_label.place(x=200, y=170)
        y_entry = ttk.Entry(textvariable=y_var)
        y_entry.place(x=200, y=200, width=70)

        k_label = ttk.Label(text="Key", font=("Arial", 12))
        k_label.place(x=300, y=170)
        k_entry = ttk.Entry(textvariable=k_var)
        k_entry.place(x=300, y=200, width=70)

        cmyk_box = Canvas(root, width=300, height=300)
        cmyk_box.place(x=10, y=300)
        rectangle_id = cmyk_box.create_rectangle(0, 0, 300, 300, fill=rgb_to_hex(0, 0, 0))

        label_nn_result = ttk.Label(text="NeuralNetwork result (press Convert button to see it)", font=("Arial", 14))
        label_nn_result.place(x=550, y=100)

        label_nn_rgb_values = ttk.Label(textvariable=rgb_nn_var, font=("Arial", 12))
        label_nn_rgb_values.place(x=700, y=150)

        rgb_nn_box = Canvas(root, width=200, height=200)
        rgb_nn_box.place(x=700, y=200)
        rectangle_rgb_nn = rgb_nn_box.create_rectangle(0, 0, 300, 300, fill=rgb_to_hex(0, 0, 0))

        label_exp_result = ttk.Label(text="Expected result:", font=("Arial", 14))
        label_exp_result.place(x=700, y=450)

        label_exp_rgb_values = ttk.Label(textvariable=rgb_exp_var, font=("Arial", 12))
        label_exp_rgb_values.place(x=700, y=500)

        rgb_box = Canvas(root, width=200, height=200)
        rgb_box.place(x=700, y=550)
        rectangle_rgb = rgb_box.create_rectangle(0, 0, 300, 300, fill=rgb_to_hex(0, 0, 0))

        # Trace changes in the entries to update the CMYK color preview
        c_var.trace_add("write", on_update)
        m_var.trace_add("write", on_update)
        y_var.trace_add("write", on_update)
        k_var.trace_add("write", on_update)

        convert_button = ttk.Button(
            text="Convert",
            command=lambda: convert(c_entry.get(), m_entry.get(), y_entry.get(), k_entry.get())
        )
        convert_button.place(x=350, y=700, width=200, height=50)

        root.mainloop()
