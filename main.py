import tkinter as tk
from tkinter import filedialog, messagebox
 
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
                result += chr((ord(char) - stay_in_alphabet + self.shift) % 26 + stay_in_alphabet)
            else:
                result += char
        return result
    def decrypt(self, text):
        result = ''
        for char in text:
            if char.isalpha():
                stay_in_alphabet = ord('A') if char.isupper() else ord('a')
                result += chr((ord(char) - stay_in_alphabet - self.shift) % 26 + stay_in_alphabet)
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
        self.geometry("520x420")
        self.input_data = tk.StringVar()
        self.selected_algorithm = tk.StringVar(value="Caesar Cipher")
        self.shift_var = tk.IntVar(value=3)
 
        tk.Label(self, text="Dane wejściowe:").pack(pady=5)
        tk.Entry(self, textvariable=self.input_data, width=52).pack(pady=5)
        tk.Button(self, text="Wczytaj z pliku", command=self.load_from_file).pack(pady=5)
 
        tk.Label(self, text="Wybierz algorytm:").pack(pady=5)
        tk.OptionMenu(self, self.selected_algorithm, "Caesar Cipher", "Reverse Cipher").pack(pady=5)
 
        tk.Label(self, text="Przesunięcie (dla cezara):").pack(pady=5)
        tk.Entry(self, textvariable=self.shift_var, width=7).pack(pady=2)
 
        tk.Button(self, text="Szyfruj", command=self.encrypt_input).pack(pady=7)
        tk.Button(self, text="Odszyfruj", command=self.decrypt_input).pack(pady=5)
 
        tk.Label(self, text="Wynik:").pack(pady=5)
 
        # Text widget do wyświetlania wyniku z możliwością zaznaczania i kopiowania
        self.result_text = tk.Text(self, wrap='word', height=10, width=60)
        self.result_text.pack(pady=5, fill='both', expand=True)
 
        tk.Button(self, text="Zapisz wynik do pliku", command=self.save_result).pack(pady=11)
 
    def load_from_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("Text Files", "*.txt"), ("All Files", "*.*")])
        if file_path:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = f.read()
                    self.input_data.set(data)
            except Exception as e:
                messagebox.showerror("Błąd", f"Nie można wczytać pliku. {e}")
 
    def encrypt_input(self):
        text = self.input_data.get()
        if not text.strip():
            messagebox.showerror("Błąd", "Dane wejściowe są puste.")
            return
        try:
            algo = CaesarCipher(self.shift_var.get()) if self.selected_algorithm.get() == "Caesar Cipher" else ReverseCipher()
            result = algo.encrypt(text)
            self.result_text.delete('1.0', tk.END)  # wyczyść pole wyniku
            self.result_text.insert(tk.END, result)  # wpisz nowy wynik
        except Exception as e:
            messagebox.showerror("Błąd", f"Nieprawidłowe dane lub algorytm. {e}")
 
    def decrypt_input(self):
        text = self.input_data.get()
        if not text.strip():
            messagebox.showerror("Błąd", "Dane wejściowe są puste.")
            return
        try:
            algo = CaesarCipher(self.shift_var.get()) if self.selected_algorithm.get() == "Caesar Cipher" else ReverseCipher()
            result = algo.decrypt(text)
            self.result_text.delete('1.0', tk.END)
            self.result_text.insert(tk.END, result)
        except Exception as e:
            messagebox.showerror("Błąd", f"Nieprawidłowe dane lub algorytm. {e}")
 
    def save_result(self):
        file_path = filedialog.asksaveasfilename(defaultextension=".txt",
            filetypes=[("Text Files", "*.txt"), ("All Files", "*.*")])
        if file_path:
            try:
                content = self.result_text.get("1.0", tk.END).strip()
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                messagebox.showinfo("Zapisano", f"Wynik został zapisany do pliku: {file_path}")
            except Exception as e:
                messagebox.showerror("Błąd", f"Nie można zapisać pliku. {e}")
 
if __name__ == "__main__":
    app = EncryptionApp()
    app.mainloop()