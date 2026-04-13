with open(r'd:\Studia\2stopień\Magisterka\datasets\f1\f1_dataset.txt', 'r', encoding='utf-8') as f:
    lines = [line.strip() for line in f if line.strip()]

max_len = max(len(l) for l in lines)
chars = set("".join(lines))
print(f"Lines: {len(lines)}")
print(f"Max length: {max_len}")
print(f"Alphabet size: {len(chars)}")
print(f"Alphabet: {''.join(sorted(chars))}")
