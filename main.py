import tkinter as tk
from tkinter import messagebox, filedialog
from tkinter import ttk

class CipherAlgorithm:
    def encrypt(self, text):
        raise NotImplementedError
    def decrypt(self, text):
        raise NotImplementedError

class CaesarCipher(CipherAlgorithm):
    def __init__(self, shift=3):
        self.shift = shift
    def encrypt(self, text):
        result = ''
        for char in text:
            if char.isalpha():
                stay_in_alphabet = ord('A') if char.isupper() else ord('a')
                result += chr((ord(char)-stay_in_alphabet + self.shift) % 26 + stay_in_alphabet)
            else:
                result += char
        return result
    def decrypt(self, text):
        result = ''
        for char in text:
            if char.isalpha():
                stay_in_alphabet = ord('A') if char.isupper() else ord('a')
                result += chr((ord(char)-stay_in_alphabet - self.shift) % 26 + stay_in_alphabet)
            else:
                result += char
        return result

class ReverseCipher(CipherAlgorithm):
    def encrypt(self, text):
        return text[::-1]
    def decrypt(self, text):
        return text[::-1]

class EncryptionApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Aplikacja szyfrująca")
        self.geometry("640x540")
        self.minsize(540, 540)
        self.current_frame = None
        style = ttk.Style(self)
        style.theme_use('clam')
        self.show_menu()

    def show_menu(self):
        if self.current_frame:
            self.current_frame.destroy()
        self.current_frame = ttk.Frame(self)
        self.current_frame.pack(expand=True, fill='both')
        ttk.Label(self.current_frame, text="Wybierz szyfr", font=("Arial", 18)).pack(pady=44)
        ttk.Button(self.current_frame, text="Szyfr Cezara", command=self.show_caesar_cipher).pack(pady=16)
        ttk.Button(self.current_frame, text="Reverse Cipher", command=self.show_reverse_cipher).pack(pady=8)

    def show_caesar_cipher(self):
        self.show_cipher_frame("Caesar Cipher")

    def show_reverse_cipher(self):
        self.show_cipher_frame("Reverse Cipher")

    def show_cipher_frame(self, cipher_name):
        if self.current_frame:
            self.current_frame.destroy()
        self.current_frame = ttk.Frame(self)
        self.current_frame.pack(expand=True, fill='both')

        # Przycisk powrotu na górze
        ttk.Button(self.current_frame, text="Wróć", command=self.show_menu).pack(anchor='nw', padx=5, pady=5)

        input_data = tk.StringVar()
        shift_var = tk.IntVar(value=3)
        mode_var = tk.StringVar(value="encrypt")

        ttk.Label(self.current_frame, text=f"Tryb: {cipher_name}", font=("Arial", 14)).pack(pady=4)
        ttk.Label(self.current_frame, text="Dane wejściowe:").pack(pady=2)
        ttk.Entry(self.current_frame, textvariable=input_data, width=54).pack(pady=3)

        def load_from_file():
            file_path = filedialog.askopenfilename(filetypes=[("Text Files", "*.txt"), ("All Files", "*.*")])
            if file_path:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = f.read()
                        input_data.set(data)
                except Exception as e:
                    messagebox.showerror("Błąd", f"Nie można wczytać pliku. {e}")

        ttk.Button(self.current_frame, text="Wczytaj z pliku", command=load_from_file).pack(pady=4)

        if cipher_name == "Caesar Cipher":
            ttk.Label(self.current_frame, text="Przesunięcie:").pack()
            ttk.Entry(self.current_frame, textvariable=shift_var, width=6).pack(pady=2)

        def process():
            text = input_data.get()
            if not text.strip():
                messagebox.showerror("Błąd", "Dane wejściowe są puste.")
                return
            try:
                if cipher_name == "Caesar Cipher":
                    algo = CaesarCipher(shift_var.get())
                else:
                    algo = ReverseCipher()
                result = algo.encrypt(text) if mode_var.get() == "encrypt" else algo.decrypt(text)
                result_text.config(state='normal')
                result_text.delete('1.0', tk.END)
                result_text.insert(tk.END, result)
                result_text.config(state='disabled')
            except Exception as e:
                messagebox.showerror("Błąd", f"Nieprawidłowe dane lub algorytm. {e}")

        def save_result():
            file_path = filedialog.asksaveasfilename(defaultextension=".txt",
                filetypes=[("Text Files", "*.txt"), ("All Files", "*.*")])
            if file_path:
                try:
                    content = result_text.get("1.0", tk.END).strip()
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(content)
                    messagebox.showinfo("Zapisano", f"Wynik został zapisany do pliku: {file_path}")
                except Exception as e:
                    messagebox.showerror("Błąd", f"Nie można zapisać pliku. {e}")

        container = ttk.Frame(self.current_frame)
        container.pack(pady=2)
        ttk.Radiobutton(container, text="Szyfruj", variable=mode_var, value="encrypt").pack(side='left', padx=8)
        ttk.Radiobutton(container, text="Odszyfruj", variable=mode_var, value="decrypt").pack(side='left', padx=8)

        ttk.Button(self.current_frame, text="Wykonaj", command=process).pack(pady=8)

        ttk.Label(self.current_frame, text="Wynik:").pack()
        result_text = tk.Text(self.current_frame, wrap='word', height=9, width=60)
        result_text.pack(pady=4, fill='both', expand=True)
        result_text.config(state='disabled')

        ttk.Button(self.current_frame, text="Zapisz wynik do pliku", command=save_result).pack(pady=8)

if __name__ == "__main__":
    app = EncryptionApp()
    app.mainloop()
