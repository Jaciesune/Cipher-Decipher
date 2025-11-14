import tkinter as tk
from tkinter import messagebox, filedialog
from tkinter import ttk
import random
import math

class CipherAlgorithm: # klasa bazowa
    def encrypt(self, text):
        raise NotImplementedError
    def decrypt(self, text):
        raise NotImplementedError

# Funkcje do klas

sbox = [99, 124, 119, 123, 242, 107, 111, 197, 48, 1, 103, 43, 254, 215, 171, 118,
        202, 130, 201, 125, 250, 89, 71, 240, 173, 212, 162, 175, 156, 164, 114, 192,
        183, 253, 147, 38, 54, 63, 247, 204, 52, 165, 229, 241, 113, 216, 49, 21,
        4, 199, 35, 195, 24, 150, 5, 154, 7, 18, 128, 226, 235, 39, 178, 117,
        9, 131, 44, 26, 27, 110, 90, 160, 82, 59, 214, 179, 41, 227, 47, 132,
        83, 209, 0, 237, 32, 252, 177, 91, 106, 203, 190, 57, 74, 76, 88, 207,
        208, 239, 170, 251, 67, 77, 51, 133, 69, 249, 2, 127, 80, 60, 159, 168,
        81, 163, 64, 143, 146, 157, 56, 245, 188, 182, 218, 33, 16, 255, 243, 210,
        205, 12, 19, 236, 95, 151, 68, 23, 196, 167, 126, 61, 100, 93, 25, 115,
        96, 129, 79, 220, 34, 42, 144, 136, 70, 238, 184, 20, 222, 94, 11, 219,
        224, 50, 58, 10, 73, 6, 36, 92, 194, 211, 172, 98, 145, 149, 228, 121,
        231, 200, 55, 109, 141, 213, 78, 169, 108, 86, 244, 234, 101, 122, 174, 8,
        186, 120, 37, 46, 28, 166, 180, 198, 232, 221, 116, 31, 75, 189, 139, 138,
        112, 62, 181, 102, 72, 3, 246, 14, 97, 53, 87, 185, 134, 193, 29, 158,
        225, 248, 152, 17, 105, 217, 142, 148, 155, 30, 135, 233, 206, 85, 40, 223,
        140, 161, 137, 13, 191, 230, 66, 104, 65, 153, 45, 15, 176, 84, 187, 22]

inv_sbox = [
    82, 9, 106, 213, 48, 54, 165, 56, 191, 64, 163, 158, 129, 243, 215, 251,
    124, 227, 57, 130, 155, 47, 255, 135, 52, 142, 67, 68, 196, 222, 233, 203,
    84, 123, 148, 50, 166, 194, 35, 61, 238, 76, 149, 11, 66, 250, 195, 78,
    8, 46, 161, 102, 40, 217, 36, 178, 118, 91, 162, 73, 109, 139, 209, 37,
    114, 248, 246, 100, 134, 104, 152, 22, 212, 164, 92, 204, 93, 101, 182, 146,
    108, 112, 72, 80, 253, 237, 185, 218, 94, 21, 70, 87, 167, 141, 157, 132,
    144, 216, 171, 0, 140, 188, 211, 10, 247, 228, 88, 5, 184, 179, 69, 6,
    208, 44, 30, 143, 202, 63, 15, 2, 193, 175, 189, 3, 1, 19, 138, 107,
    58, 145, 17, 65, 79, 103, 220, 234, 151, 242, 207, 206, 240, 180, 230, 115,
    150, 172, 116, 34, 231, 173, 53, 133, 226, 249, 55, 232, 28, 117, 223, 110,
    71, 241, 26, 113, 29, 41, 197, 137, 111, 183, 98, 14, 170, 24, 190, 27,
    252, 86, 62, 75, 198, 210, 121, 32, 154, 219, 192, 254, 120, 205, 90, 244,
    31, 221, 168, 51, 136, 7, 199, 49, 177, 18, 16, 89, 39, 128, 236, 95,
    96, 81, 127, 169, 25, 181, 74, 13, 45, 229, 122, 159, 147, 201, 156, 239,
    160, 224, 59, 77, 174, 42, 245, 176, 200, 235, 187, 60, 131, 83, 153, 97,
    23, 43, 4, 126, 186, 119, 214, 38, 225, 105, 20, 99, 85, 33, 12, 125
]

rcon = [0, 1, 2, 4, 8, 16, 32, 64, 128, 27, 54]

def pad(data):
    pad_len = 16 - len(data) % 16
    return data + bytes([pad_len] * pad_len)

def unpad(data):
    pad_len = data[-1]
    return data[:-pad_len]

def sub_bytes(state):
    return [sbox[b] for b in state]

def shift_rows(s):
    return [
        s[0], s[5], s[10], s[15],
        s[4], s[9], s[14], s[3],
        s[8], s[13], s[2], s[7],
        s[12], s[1], s[6], s[11]
    ]

def xtime(a):
    return ((a << 1) ^ 0x1B) & 0xFF if (a & 0x80) else (a << 1)

def mix_single_column(a):
    t = a[0] ^ a[1] ^ a[2] ^ a[3]
    b0 = a[0] ^ t ^ xtime(a[0] ^ a[1])
    b1 = a[1] ^ t ^ xtime(a[1] ^ a[2])
    b2 = a[2] ^ t ^ xtime(a[2] ^ a[3])
    b3 = a[3] ^ t ^ xtime(a[3] ^ a[0])
    return [b0, b1, b2, b3]

def mix_columns(s):
    out = []
    for i in range(4):
        col = s[i*4:i*4+4]
        out += mix_single_column(col)
    return out

def add_round_key(s, k):
    return [b ^ rk for b, rk in zip(s, k)]

def rotate(word):
    return word[1:] + word[:1]

def key_expansion(key):
    w = [list(key[i*4:(i+1)*4]) for i in range(4)]
    for i in range(4, 44):
        temp = w[i-1].copy()
        if i % 4 == 0:
            temp = [sbox[x] for x in rotate(temp)]
            temp[0] ^= rcon[i//4]
        w.append([w[i-4][j] ^ temp[j] for j in range(4)])
    expanded = []
    for i in range(0, len(w), 4):
        expanded.append([item for sublist in w[i:i+4] for item in sublist])
    return expanded

def inv_shift_rows(s):
    return [
        s[0], s[13], s[10], s[7],
        s[4], s[1],  s[14], s[11],
        s[8], s[5],  s[2],  s[15],
        s[12],s[9],  s[6],  s[3]
    ]

def inv_sub_bytes(state):
    return [inv_sbox[b] for b in state]

def inv_mix_single_column(a):
    mul = lambda x, y: (
        ((y & 1) * x) ^
        ((y >> 1 & 1) * xtime(x)) ^
        ((y >> 2 & 1) * xtime(xtime(x))) ^
        ((y >> 3 & 1) * xtime(xtime(xtime(x))))
    )
    return [
        mul(a[0], 14) ^ mul(a[1], 11) ^ mul(a[2], 13) ^ mul(a[3], 9),
        mul(a[0], 9)  ^ mul(a[1], 14) ^ mul(a[2], 11) ^ mul(a[3], 13),
        mul(a[0], 13) ^ mul(a[1], 9)  ^ mul(a[2], 14) ^ mul(a[3], 11),
        mul(a[0], 11) ^ mul(a[1], 13) ^ mul(a[2], 9)  ^ mul(a[3], 14),
    ]

def inv_mix_columns(s):
    out = []
    for i in range(4):
        col = s[i*4:i*4+4]
        out += inv_mix_single_column(col)
    return out

def inc_bytes(bs):
        as_list = list(bs)
        for i in reversed(range(len(as_list))):
            as_list[i] = (as_list[i] + 1) % 256
            if as_list[i] != 0:
                break
        return bytes(as_list)

# Funkcja mnożenia w GF(2^128)
def gf_mul(X, Y):
    R = 0xe1000000000000000000000000000000
    x = int.from_bytes(X, "big")
    y = int.from_bytes(Y, "big")
    z = 0
    for i in range(128):
        if y & (1 << (127 - i)):
            z ^= x
        if x & 1:
            x = (x >> 1) ^ R
        else:
            x >>= 1
    return z.to_bytes(16, "big")

#Funkcja GHASH
def ghash(H, A, C):
    a_len = len(A)
    c_len = len(C)
    A_padded = A + b"\x00" * ((16 - a_len % 16) % 16)
    C_padded = C + b"\x00" * ((16 - c_len % 16) % 16)
    X = b"\x00" * 16
    X_input = A_padded + C_padded + (a_len * 8).to_bytes(8, "big") + (c_len * 8).to_bytes(8, "big")
    blocks = [X_input[i:i+16] for i in range(0, len(X_input), 16)]
    for block in blocks:
        X = gf_mul(bytes([x ^ y for x, y in zip(X, block)]), H)
    return X


#Funkcje dla RSA

def is_prime(n, k=8):
    if n < 2:
        return False
    for p in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]:
        if n % p == 0:
            return n == p
    # Miller-Rabin test
    d, s = n - 1, 0
    while d % 2 == 0:
        d //= 2
        s += 1
    for _ in range(k):
        a = random.randrange(2, n - 2)
        x = pow(a, d, n)
        if x == 1 or x == n - 1:
            continue
        for _ in range(s - 1):
            x = pow(x, 2, n)
            if x == n - 1:
                break
        else:
            return False
    return True

def generate_prime(bits):
    while True:
        num = random.getrandbits(bits)
        num |= 1 << bits - 1 | 1  # zapewnij odpowiedni rozmiar i nieparzystość
        if is_prime(num):
            return num

def gcd(a, b):
    while b:
        a, b = b, a % b
    return a

def modinv(a, m):
    # Extended Euclidean Algorithm
    g, x, y = extended_gcd(a, m)
    if g != 1:
        raise Exception('Brak odwrotności modulo')
    return x % m

def extended_gcd(a, b):
    if a == 0:
        return b, 0, 1
    else:
        g, y, x = extended_gcd(b % a, a)
        return g, x - (b // a) * y, y

# -------------------------------------- #

# RSA 

class RSA(CipherAlgorithm):
    def __init__(self, bits=512):
        self.bits = bits
        self.e = 65537
        self._generate_keys()

    def _generate_keys(self):
        p = generate_prime(self.bits)
        q = generate_prime(self.bits)
        while q == p:
            q = generate_prime(self.bits)
        self.n = p * q
        phi = (p - 1) * (q - 1)
        while gcd(self.e, phi) != 1:
            self.e += 2
        self.d = modinv(self.e, phi)

    def encrypt(self, text):
        # Zamienia tekst na liczby, szyfruje blokowo (blok < n)
        # Prosta wersja: koduje utf-8 jako int, wymaga podziału na małe bloki
        plain_bytes = text.encode('utf-8')
        block_size = (self.n.bit_length() - 1) // 8
        blocks = [plain_bytes[i:i+block_size] for i in range(0, len(plain_bytes), block_size)]
        encrypted_blocks = [pow(int.from_bytes(block, 'big'), self.e, self.n) for block in blocks]
        return encrypted_blocks  # lista liczb

    def decrypt(self, ciphertext):
        block_size = (self.n.bit_length() - 1) // 8
        plain_bytes = b''.join([pow(c, self.d, self.n).to_bytes(block_size, 'big').lstrip(b'\x00') for c in ciphertext])
        return plain_bytes.decode('utf-8', errors='replace')

    def public_key(self):
        return (self.n, self.e)

    def private_key(self):
        return (self.n, self.d)


# -- RSA -- #

class AES_ECB(CipherAlgorithm):
    def __init__(self, key):
        assert len(key) == 16
        self.round_keys = key_expansion(key)
        
    def encrypt_block(self, block):
        assert len(block) == 16
        state = list(block)
        state = add_round_key(state, self.round_keys[0])
        for r in range(1, 10):
            state = sub_bytes(state)
            state = shift_rows(state)
            state = mix_columns(state)
            state = add_round_key(state, self.round_keys[r])
        state = sub_bytes(state)
        state = shift_rows(state)
        state = add_round_key(state, self.round_keys[10])
        return bytes(state)

    def encrypt(self, text):
        data = pad(text.encode("utf-8") if isinstance(text, str) else text)
        return b"".join(self.encrypt_block(data[i:i+16]) for i in range(0, len(data), 16))

    def decrypt_block(self, block):
        assert len(block) == 16
        state = list(block)
        state = add_round_key(state, self.round_keys[10])
        state = inv_shift_rows(state)
        state = inv_sub_bytes(state)
        for r in range(9, 0, -1):
            state = add_round_key(state, self.round_keys[r])
            state = inv_mix_columns(state)
            state = inv_shift_rows(state)
            state = inv_sub_bytes(state)
        state = add_round_key(state, self.round_keys[0])
        return bytes(state)

    def decrypt(self, data):
        blocks = [data[i:i+16] for i in range(0, len(data), 16)]
        dec = b"".join(self.decrypt_block(b) for b in blocks)
        return unpad(dec)
    
class AES_CBC(CipherAlgorithm):
    def __init__(self, key, iv):
        assert len(key) == 16
        assert len(iv) == 16
        self.iv = iv
        self.ecb = AES_ECB(key)
    def encrypt(self, text):
        data = pad(text.encode("utf-8") if isinstance(text, str) else text)
        blocks = [data[i:i+16] for i in range(0, len(data), 16)]
        result = []
        prev = self.iv
        for block in blocks:
            xored = bytes([b ^ p for b, p in zip(block, prev)])
            enc_blk = self.ecb.encrypt_block(xored)
            result.append(enc_blk)
            prev = enc_blk
        return b"".join(result)
    def decrypt(self, data):
        blocks = [data[i:i+16] for i in range(0, len(data), 16)]
        result = []
        prev = self.iv
        for block in blocks:
            dec_blk = self.ecb.decrypt_block(block)
            plain_blk = bytes([b ^ p for b, p in zip(dec_blk, prev)])
            result.append(plain_blk)
            prev = block
        return unpad(b"".join(result))

class AES_CTR(CipherAlgorithm):
    def __init__(self, key, nonce):
        assert len(key) == 16
        assert len(nonce) == 16
        self.nonce = nonce
        self.ecb = AES_ECB(key)

    def process(self, data):
        blocks = [data[i:i+16] for i in range(0, len(data), 16)]
        ctr = self.nonce
        out = []
        for block in blocks:
            stream = self.ecb.encrypt_block(ctr)
            out_blk = bytes([b ^ s for b, s in zip(block, stream)])
            out.append(out_blk)
            ctr = inc_bytes(ctr)
        return b"".join(out)
    def encrypt(self, text):
        data = text.encode("utf-8") if isinstance(text, str) else text
        return self.process(data)
    def decrypt(self, data):
        return self.process(data)
    
class AES_GCM(CipherAlgorithm):
    def __init__(self, key, nonce, aad=b""):
        assert len(key) == 16
        assert len(nonce) == 16 or len(nonce) == 12
        self.nonce = nonce
        self.key = key
        self.ecb = AES_ECB(key)
        self.aad = aad

    def encrypt(self, text):
        data = text.encode("utf-8") if isinstance(text, str) else text
        # Krok 1: gen. subkey h
        H = self.ecb.encrypt_block(b"\x00" * 16)
        if len(self.nonce) == 12:
            J0 = self.nonce + b"\x00\x00\x00\x01"
        else:
            J0 = self.nonce

        # CTR:
        ctr_block = bytearray(J0)
        ctr_block[-1] = (ctr_block[-1] + 1) % 256

        # Szyfrowanie jak w CTR
        blocks = [data[i:i+16] for i in range(0, len(data), 16)]
        ciphertext = []
        for block in blocks:
            stream = self.ecb.encrypt_block(bytes(ctr_block))
            cipher = bytes([b ^ s for b, s in zip(block.ljust(16, b"\x00"), stream)])
            ciphertext.append(cipher[:len(block)])
            # Inkrementuj licznik
            for i in reversed(range(len(ctr_block))):
                ctr_block[i] = (ctr_block[i] + 1) % 256
                if ctr_block[i] != 0:
                    break
        ciphertext_bytes = b"".join(ciphertext)

        # Tag: GHASH(H, AAD, ciphertext)
        tag_input = ghash(H, self.aad, ciphertext_bytes)
        S = self.ecb.encrypt_block(J0)
        tag = bytes([x ^ y for x, y in zip(tag_input, S)])
        return ciphertext_bytes, tag

    def decrypt(self, data, tag):
        # Proces ten sam co przy szyfrowaniu
        H = self.ecb.encrypt_block(b"\x00" * 16)
        if len(self.nonce) == 12:
            J0 = self.nonce + b"\x00\x00\x00\x01"
        else:
            J0 = self.nonce
        ctr_block = bytearray(J0)
        ctr_block[-1] = (ctr_block[-1] + 1) % 256
        blocks = [data[i:i+16] for i in range(0, len(data), 16)]
        plaintext = []
        for block in blocks:
            stream = self.ecb.encrypt_block(bytes(ctr_block))
            plain = bytes([b ^ s for b, s in zip(block.ljust(16, b"\x00"), stream)])
            plaintext.append(plain[:len(block)])
            for i in reversed(range(len(ctr_block))):
                ctr_block[i] = (ctr_block[i] + 1) % 256
                if ctr_block[i] != 0:
                    break
        plaintext_bytes = b"".join(plaintext)
        # Tag: GHASH(H, AAD, ciphertext)
        expected_tag_input = ghash(H, self.aad, data)
        S = self.ecb.encrypt_block(J0)
        expected_tag = bytes([x ^ y for x, y in zip(expected_tag_input, S)])
        if expected_tag != tag:
            raise ValueError("Tag/MAC niezgodny — dane podrobione lub z błędem!")
        return plaintext_bytes

class CaesarCipher(CipherAlgorithm): #Szyfr Cezara
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

class ReverseCipher(CipherAlgorithm): #Szyfr odwracający ciąg znaków
    def encrypt(self, text):
        return text[::-1]
    def decrypt(self, text):
        return text[::-1]
    
class BeaufortCipher(CipherAlgorithm): #Szyfr Beaufort'a
    def __init__(self, key):
        self.key = key

    def _process(self, text, encrypt=True):
        result = ''
        key = self.key
        key_len = len(key)
        for i, char in enumerate(text):
            if char.isalpha():
                stay_in_alphabet = ord('A') if char.isupper() else ord('a')
                p = ord(char.upper()) - ord('A')
                k = ord(key[i % key_len].upper()) - ord('A')
                c = (k - p) % 26
                result += chr(c + stay_in_alphabet)
            else:
                result += char
        return result

    def encrypt(self, text):
        return self._process(text)

    def decrypt(self, text):
        return self._process(text)

class RunningKeyCipher(CipherAlgorithm):
    def __init__(self, key):
        self.key = key
    
    def encrypt(self, text):
        result = ""
        key = self.key
        for i, char in enumerate(text):
            if char.isalpha():
                stay_in_alphabet = ord('A') if char.isupper() else ord('a')
                pi = ord(char.upper()) - ord('A')
                ki = ord(key[i % len(key)].upper()) - ord('A')
                ci = (pi + ki) % 26
                result += chr(ci + stay_in_alphabet)
            else:
                result += char
        return result
    
    def decrypt(self, text):
        result = ""
        key = self.key
        for i, char in enumerate(text):
            if char.isalpha():
                stay_in_alphabet = ord('A') if char.isupper() else ord('a')
                pi = ord(char.upper()) - ord('A')
                ki = ord(key[i % len(key)].upper()) - ord('A')
                ci = (pi - ki) % 26
                result += chr(ci + stay_in_alphabet)
            else:
                result += char
        return result

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
        ttk.Button(self.current_frame, text="Szyfr Beaufort'a", command=self.show_beaufort_cipher).pack(pady=8)
        ttk.Button(self.current_frame, text="Szyfr z kluczem bieżącym", command=self.show_running_key_cipher).pack(pady=8)
        ttk.Button(self.current_frame, text="AES", command=self.show_aes_cipher).pack(pady=8)
        ttk.Button(self.current_frame, text="RSA", command=self.show_rsa_cipher).pack(pady=8)

    def show_caesar_cipher(self):
        self.show_cipher_frame("Caesar Cipher")

    def show_reverse_cipher(self):
        self.show_cipher_frame("Reverse Cipher")
    
    def show_beaufort_cipher(self):
        self.show_cipher_frame("Beaufort Cipher")

    def show_running_key_cipher(self):
        self.show_cipher_frame("Running Key Cipher")

    def show_aes_cipher(self):
        self.show_cipher_frame("AES")

    def show_rsa_cipher(self):
        self.show_cipher_frame("RSA")

    def show_cipher_frame(self, cipher_name):
        if self.current_frame:
            self.current_frame.destroy()
        self.current_frame = ttk.Frame(self)
        self.current_frame.pack(expand=True, fill='both')
        input_data = tk.StringVar()

        # Przycisk powrotu na górze
        ttk.Button(self.current_frame, text="Wróć", command=self.show_menu).pack(anchor='nw', padx=5, pady=5)

        if cipher_name == "RSA":
            bits_var = tk.IntVar(value=256)
            key_text = tk.StringVar()
            rsa_obj = [None]

            def generate_keys():
                try:
                    rsa = RSA(bits=bits_var.get())
                    pubkey = rsa.public_key()
                    privkey = rsa.private_key()
                    key_info = f"Publiczny:\nn={pubkey[0]}\ne={pubkey[1]}\n\nPrywatny:\nd={privkey[1]}"
                    key_text.set(key_info)
                    rsa_obj[0] = rsa
                except Exception as e:
                    messagebox.showerror("Błąd", f"Błąd generowania kluczy: {e}")

            def rsa_encrypt():
                try:
                    if not rsa_obj[0]:
                        raise Exception("Brak wygenerowanego klucza!")
                    text = input_data.get()
                    encrypted = rsa_obj[0].encrypt(text)
                    result_text.config(state='normal')
                    result_text.delete('1.0', tk.END)
                    print(encrypted, type(encrypted))
                    result_text.insert(tk.END, ",".join(map(str, encrypted)))
                    result_text.config(state='disabled')
                except Exception as e:
                    messagebox.showerror("Błąd", str(e))

            def rsa_decrypt():
                try:
                    if not rsa_obj[0]:
                        raise Exception("Brak wygenerowanego klucza!")
                    ciphertext = input_data.get()
                    blocks = [int(part.strip()) for part in ciphertext.strip().split(',')]
                    decrypted = rsa_obj[0].decrypt(blocks)
                    result_text.config(state='normal')
                    result_text.delete('1.0', tk.END)
                    result_text.insert(tk.END, decrypted)
                    result_text.config(state='disabled')
                except Exception as e:
                    messagebox.showerror("Błąd", str(e))

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

            ttk.Label(self.current_frame, text="Długość klucza (bity):").pack()
            ttk.Entry(self.current_frame, textvariable=bits_var, width=8).pack(pady=2)
            ttk.Button(self.current_frame, text="Generuj klucze", command=generate_keys).pack(pady=2)
            ttk.Label(self.current_frame, text="Parametry klucza:").pack()
            ttk.Entry(self.current_frame, textvariable=key_text, width=60, state="readonly").pack(pady=2, fill='x')
            ttk.Label(self.current_frame, text="Dane wejściowe:").pack()
            ttk.Entry(self.current_frame, textvariable=input_data, width=54).pack(pady=2)
            container = ttk.Frame(self.current_frame)
            container.pack(pady=2)
            ttk.Button(container, text="Szyfruj", command=rsa_encrypt).pack(side='left', padx=8)
            ttk.Button(container, text="Odszyfruj", command=rsa_decrypt).pack(side='left', padx=8)
            ttk.Label(self.current_frame, text="Wynik:").pack()
            result_text = tk.Text(self.current_frame, wrap='word', height=9, width=60)
            result_text.pack(pady=4, fill='both', expand=True)
            result_text.config(state='disabled')
            ttk.Button(self.current_frame, text="Zapisz wynik do pliku", command=save_result).pack(pady=8)
            return

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

        def load_key_from_file():
            file_path = filedialog.askopenfilename(filetypes=[("Text Files", "*.txt"), ("All Files", "*.*")])
            if file_path:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        key_data = f.read()
                        key_data = ''.join([char for char in key_data if char.isalpha()])
                        key_var.set(key_data)
                except Exception as e:
                    messagebox.showerror("Błąd", f"Nie można wczytać klucza. {e}")

        ttk.Button(self.current_frame, text="Wczytaj z pliku", command=load_from_file).pack(pady=4)

        if cipher_name == "Caesar Cipher":
            ttk.Label(self.current_frame, text="Przesunięcie:").pack()
            ttk.Entry(self.current_frame, textvariable=shift_var, width=6).pack(pady=2)

        key_var = tk.StringVar()
        if cipher_name == "Beaufort Cipher":
            ttk.Label(self.current_frame, text="Klucz:").pack()
            ttk.Entry(self.current_frame, textvariable=key_var, width=20).pack(pady=2)
            ttk.Button(self.current_frame, text="Wczytaj klucz z pliku", command=load_key_from_file).pack(pady=2)
        
        if cipher_name == "Running Key Cipher":
            ttk.Button(self.current_frame, text="Wczytaj klucz z pliku", command=load_key_from_file).pack(pady=2)

        if cipher_name == "AES":
            ttk.Label(self.current_frame, text="Tryb AES:").pack()
            mode_choice = tk.StringVar(value="ECB")
            ttk.Combobox(self.current_frame, textvariable=mode_choice, values=["ECB", "CBC", "CTR", "GCM"], state="readonly", width=12).pack(pady=2)

            aes_key_var = tk.StringVar()
            ttk.Label(self.current_frame, text="Klucz (16 znaków):").pack()
            ttk.Entry(self.current_frame, textvariable=aes_key_var, width=20).pack(pady=2)

            iv_var = tk.StringVar()
            ttk.Label(self.current_frame, text="IV/Nonce (16 znaków, tylko CBC/CTR/GCM):").pack()
            ttk.Entry(self.current_frame, textvariable=iv_var, width=20).pack(pady=2)

        def process():
            text = input_data.get()
            if not text.strip():
                messagebox.showerror("Błąd", "Dane wejściowe są puste.")
                return
            try:
                if cipher_name == "Caesar Cipher":
                    algo = CaesarCipher(shift_var.get())
                elif cipher_name == "Reverse Cipher":
                    algo = ReverseCipher()
                elif cipher_name == "Beaufort Cipher":
                    key = key_var.get()
                    if not key.isalpha() or not key:
                        messagebox.showerror("Błąd", "Klucz musi być niepusty i alfabetyczny.")
                        return
                    algo = BeaufortCipher(key)
                elif cipher_name == "Running Key Cipher":
                    key = key_var.get()
                    if len(key) < len(text) or not key.isalpha():
                        messagebox.showerror("Błąd", "Klucz musi być alfabetyczny i nie krótszy niż wiadomość.")
                        return
                    algo = RunningKeyCipher(key)

                elif cipher_name == "AES":
                    try:
                        key = aes_key_var.get().encode("utf-8")
                        mode = mode_choice.get()
                        iv = iv_var.get().encode("utf-8") if mode in ["CBC", "CTR", "GCM"] else None
                        if len(key) != 16:
                            messagebox.showerror("Błąd", "Klucz AES musi mieć 16 znaków (128 bitów)!")
                            return
                        if mode == "ECB":
                            aes = AES_ECB(key)
                        elif mode == "CBC":
                            if not iv or len(iv) != 16:
                                messagebox.showerror("Błąd", "IV do CBC musi mieć 16 znaków!")
                                return
                            aes = AES_CBC(key, iv)
                        elif mode == "CTR":
                            if not iv or len(iv) != 16:
                                messagebox.showerror("Błąd", "Nonce/IV do CTR musi mieć 16 znaków!")
                                return
                            aes = AES_CTR(key, iv)
                        
                        elif mode == "GCM":
                            if not iv or not (len(iv) == 12 or len(iv) == 16):
                                messagebox.showerror("Błąd", "Nonce/IV do GCM musi mieć 12 lub 16 znaków!")
                                return
                            aes = AES_GCM(key, iv)
                            if mode_var.get() == "encrypt":
                                cipher_bytes, tag = aes.encrypt(text)
                                # Pokazujemy tekst i tag (tag powinien być HEX do skopiowania/deszyfrowania!)
                                result = cipher_bytes.hex() + "\nTAG: " + tag.hex()
                            else:
                                # Dla deszyfrowania użytkownik musi podać HEX ciphertext + HEX tag (np. sklejane: tekst\nTAG:hex)
                                if "TAG:" in text:
                                    enc, tag_hex = text.strip().split("TAG:")
                                    enc_bytes = bytes.fromhex(enc.strip())
                                    tag_bytes = bytes.fromhex(tag_hex.strip())
                                    try:
                                        plain_bytes = aes.decrypt(enc_bytes, tag_bytes)
                                        result = plain_bytes.decode("utf-8", errors="ignore")
                                    except Exception as err:
                                        messagebox.showerror("Błąd", str(err))
                                        return
                                else:
                                    messagebox.showerror("Błąd", "Podaj szyfrogram oraz tag poprawnie (sklejone HEX + 'TAG:hex').")
                                    return
                            result_text.config(state='normal')
                            result_text.delete('1.0', tk.END)
                            result_text.insert(tk.END, result)
                            result_text.config(state='disabled')
                            return

                        else:
                            messagebox.showinfo("Info", "Tryb GCM niezaimplementowany.")
                            return

                        if mode_var.get() == "encrypt":
                            result_bytes = aes.encrypt(text)
                            result = result_bytes.hex()
                        else:
                            decoded = bytes.fromhex(text.strip())
                            result_bytes = aes.decrypt(decoded)
                            result = result_bytes.decode("utf-8", errors="ignore")

                        result_text.config(state='normal')
                        result_text.delete('1.0', tk.END)
                        result_text.insert(tk.END, result)
                        result_text.config(state='disabled')
                        return
                    except Exception as e:
                        messagebox.showerror("Błąd", f"Nieprawidłowe dane lub algorytm. {e}")
                        return

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
        if cipher_name != "Beaufort Cipher":
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
