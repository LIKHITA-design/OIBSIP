
import tkinter as tk
from tkinter import messagebox
from pathlib import Path
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from PIL import Image, ImageTk
# --------------------------------
# STEP 1: LOAD IRIS DATASET
# --------------------------------

iris = load_iris()

X = iris.data
y = iris.target

# --------------------------------
# STEP 2: TRAIN MACHINE LEARNING MODEL
# --------------------------------

model = LogisticRegression(max_iter=200)
model.fit(X, y)

print("Model trained successfully!")

# --------------------------------
# STEP 3: CREATE MAIN WINDOW
# --------------------------------

root = tk.Tk()

root.title("Iris Flower Prediction")
root.geometry("550x680")
root.configure(bg="#EAF2F8")
root.resizable(False, False)
# Flower image display
image_label = tk.Label(root, bg="#EAF2F8")
image_label.pack(pady=10)

# --------------------------------
# STEP 4: HEADING
# --------------------------------

title = tk.Label(
    root,
    text="Iris Flower Predictor",
    font=("Arial", 24, "bold"),
    bg="#EAF2F8",
    fg="#154360"
)

title.pack(pady=20)

subtitle = tk.Label(
    root,
    text="Enter the flower measurements below",
    font=("Arial", 12),
    bg="#EAF2F8"
)

subtitle.pack(pady=5)

# --------------------------------
# STEP 5: INPUT FIELDS
# --------------------------------

fields = [
    "Sepal Length (cm)",
    "Sepal Width (cm)",
    "Petal Length (cm)",
    "Petal Width (cm)"
]

entries = []

for field in fields:

    label = tk.Label(
        root,
        text=field,
        font=("Arial", 11),
        bg="#EAF2F8"
    )

    label.pack(pady=(10, 3))

    entry = tk.Entry(
        root,
        font=("Arial", 12),
        width=25,
        justify="center"
    )

    entry.pack()

    entries.append(entry)

# --------------------------------
# STEP 6: PREDICTION FUNCTION
# --------------------------------

def predict_flower():

    try:

        values = [
            float(entry.get())
            for entry in entries
        ]

        # Validate measurements

        if any(value <= 0 or value > 10 for value in values):

            messagebox.showerror(
                "Invalid Input",
                "Enter measurements greater than 0 and at most 10 cm."
            )

            return

        new_flower = [values]

        # Predict flower species

        prediction = model.predict(new_flower)

        probabilities = model.predict_proba(new_flower)

        species = iris.target_names[prediction[0]]
        show_flower_image(species)
        confidence = max(probabilities[0]) * 100

        result_label.config(
    text=f"Predicted Flower: {species.title()}\nConfidence: {confidence:.2f}%",
    fg="#117A65"
)
        

        # Display probabilities

        probability_text = "Prediction Probabilities:\n\n"

        for name, probability in zip(
            iris.target_names,
            probabilities[0]
        ):

            probability_text += (
                f"{name.title()}: {probability * 100:.2f}%\n"
            )

        probability_label.config(
            text=probability_text
        )

    except ValueError:

        messagebox.showerror(
            "Invalid Input",
            "Please enter valid numbers in all four fields."
        )


# --------------------------------
# STEP 7: RESET FUNCTION
# --------------------------------

def reset_fields():

    for entry in entries:
        entry.delete(0, tk.END)

    result_label.config(text="")

    probability_label.config(text="")
def show_flower_image(species):
    image_path = Path(__file__).resolve().parent / "images" / f"{species}.jpg"

    img = Image.open(image_path)
    img = img.resize((180, 180))

    photo = ImageTk.PhotoImage(img)

    image_label.config(image=photo)
    image_label.image = photo
def reset_fields():
    for entry in entries:
        entry.delete(0, tk.END)
    result_label.config(text="")
    probability_label.config(text="")
    image_label.config(image="")
    image_label.image = None
# --------------------------------
# STEP 8: BUTTONS
# --------------------------------

button_frame = tk.Frame(
    root,
    bg="#EAF2F8"
)

button_frame.pack(pady=20)

predict_button = tk.Button(
    button_frame,
    text="Predict Flower",
    command=predict_flower,
    font=("Arial", 12, "bold"),
    bg="#2874A6",
    fg="white",
    padx=20,
    pady=10,
    cursor="hand2"
)

predict_button.pack(side="left", padx=10)

reset_button = tk.Button(
    button_frame,
    text="Reset",
    command=reset_fields,
    font=("Arial", 12, "bold"),
    bg="#CB4335",
    fg="white",
    padx=20,
    pady=10,
    cursor="hand2"
)

reset_button.pack(side="left", padx=10)

# --------------------------------
# STEP 9: RESULT DISPLAY
# --------------------------------

result_label = tk.Label(
    root,
    text="",
    font=("Arial", 16, "bold"),
    bg="#EAF2F8"
)

result_label.pack(pady=15)

probability_label = tk.Label(
    root,
    text="",
    font=("Arial", 12),
    bg="#EAF2F8",
    justify="left"
)

probability_label.pack(pady=10)

# --------------------------------
# STEP 10: RUN APPLICATION
# --------------------------------#

root.mainloop()
