from transformers import AutoTokenizer


def sanity_check():

    print("=" * 60)

    tokenizer_name = "ai4bharat/IndicBERTv2-MLM-only"

    print("Loading tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    sentence = "संगणक प्रणाली द्वारा भाषा प्रसंस्करण"

    print("Input:")
    print(sentence)

    tokens = tokenizer.tokenize(sentence)
    ids = tokenizer.convert_tokens_to_ids(tokens)

    print("\nTokens")
    print(tokens)

    print("\nIDs")
    print(ids)

    print("\nRecovered")
    print(tokenizer.decode(ids))

    print("=" * 60)
    print("Environment PASSED")


if __name__ == "__main__":
    sanity_check()