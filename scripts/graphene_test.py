import regex

text = "क्षेत्रज्ञान"

clusters = regex.findall(r"\X", text)

print(clusters)

print()

print("Number of graphemes:", len(clusters))