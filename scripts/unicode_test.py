texts = [
    "नमस्ते",
    "मराठी भाषा",
    "संस्कृतम्",
    "कंप्यूटर",
    "ज्ञान",
    "प्रज्ञा",
    "श्रद्धा",
    "क्षेत्र",
    "त्रिकोण",
    "ज्ञानी"
]

for text in texts:

    print(text)

    for ch in text:
        print(ch, hex(ord(ch)))

    print("-"*40)