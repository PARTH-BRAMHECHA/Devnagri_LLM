import os
import time
from datasets import load_dataset
from tqdm import tqdm

# Configuration for targets (in Bytes)
# 50 MB = 50 * 1024 * 1024 Bytes
TARGET_50MB = 50 * 1024 * 1024

# Retry behavior for transient network errors (e.g. WinError 10038 mid-stream)
MAX_RETRIES = 5
RETRY_BACKOFF_SECONDS = 3

CONFIG = {
    "hindi": {
        "wikipedia": {
            "path": "wikimedia/wikipedia",
            "name": "20231101.hi",
            "field": "text",
            "target": TARGET_50MB,
        },
        "indiccorp": {
            "path": "ai4bharat/IndicCorpV2",       # NOTE: no hyphen before "V2"
            "name": "indiccorp_v2",                # fixed config name for this dataset
            "split": "hin_Deva",                   # language is chosen via split, not data_dir
            "field": "text",
            "target": TARGET_50MB,
        },
    },
    "marathi": {
        "wikipedia": {
            "path": "wikimedia/wikipedia",
            "name": "20231101.mr",
            "field": "text",
            "target": TARGET_50MB,
        },
        "indiccorp": {
            "path": "ai4bharat/IndicCorpV2",
            "name": "indiccorp_v2",
            "split": "mar_Deva",
            "field": "text",
            "target": TARGET_50MB,
        },
    },
}


def load_streaming_dataset(dataset_path, dataset_name, split="train"):
    """Wraps load_dataset so datasets that select their language via a
    custom split name (e.g. IndicCorpV2's 'hin_Deva') work the same way
    as ones that just use the default 'train' split."""
    return load_dataset(dataset_path, dataset_name, split=split, streaming=True)


def save_text_until_limit(lang, source, dataset_path, dataset_name, text_field,
                           target_bytes, split="train"):
    output_dir = f"data/{lang}/{source}"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "corpus.txt")

    print(f"\n::: Processing {lang.upper()} from {source.upper()} :::")
    print(f"Target size: {target_bytes / (1024*1024):.2f} MB")

    # --- Resume support: skip if we already hit the target from a prior run ---
    if os.path.exists(output_file) and os.path.getsize(output_file) >= target_bytes:
        print(f"Already have {os.path.getsize(output_file) / (1024*1024):.2f} MB "
              f"at {output_file}, skipping.")
        return True

    bytes_written = os.path.getsize(output_file) if os.path.exists(output_file) else 0
    file_mode = "a" if bytes_written > 0 else "w"
    if bytes_written > 0:
        print(f"Resuming from existing {bytes_written / (1024*1024):.2f} MB")

    attempt = 0
    while attempt < MAX_RETRIES:
        attempt += 1
        try:
            dataset = load_streaming_dataset(dataset_path, dataset_name, split)
        except Exception as e:
            print(f"Error loading dataset {dataset_path} ({dataset_name}): {e}")
            return False

        try:
            with tqdm(total=target_bytes, initial=bytes_written, unit='B',
                      unit_scale=True, desc=f"Downloading {source}") as pbar:
                with open(output_file, file_mode, encoding="utf-8") as f:
                    for row in dataset:
                        text = (row.get(text_field) or "").strip()
                        if not text:
                            continue

                        text_bytes_len = len(text.encode('utf-8'))

                        if bytes_written + text_bytes_len > target_bytes:
                            remaining_bytes = target_bytes - bytes_written
                            truncated_text = text.encode('utf-8')[:remaining_bytes].decode(
                                'utf-8', errors='ignore')
                            if truncated_text:
                                f.write(truncated_text + "\n")
                                pbar.update(len(truncated_text.encode('utf-8')))
                            bytes_written = target_bytes
                            break

                        f.write(text + "\n")
                        bytes_written += text_bytes_len
                        pbar.update(text_bytes_len)

                        if bytes_written >= target_bytes:
                            break

            # Reached here without an exception -> success
            print(f"Saved: {output_file} ({os.path.getsize(output_file) / (1024*1024):.2f} MB)")
            return True

        except KeyboardInterrupt:
            print("\nInterrupted by user.")
            raise

        except Exception as e:
            # Covers transient network issues mid-stream, e.g. WinError 10038,
            # connection resets, timeouts, etc.
            print(f"Stream interrupted on attempt {attempt}/{MAX_RETRIES}: {e}")
            file_mode = "a"  # keep whatever we already wrote, append on retry
            if attempt < MAX_RETRIES:
                wait = RETRY_BACKOFF_SECONDS * attempt
                print(f"Retrying in {wait}s...")
                time.sleep(wait)
            else:
                print(f"Giving up on {lang}/{source} after {MAX_RETRIES} attempts.")
                return False

    return False


def main():
    results = {}

    for lang, sources in CONFIG.items():
        for source_name, src_config in sources.items():
            key = f"{lang}/{source_name}"
            try:
                ok = save_text_until_limit(
                    lang=lang,
                    source=source_name,
                    dataset_path=src_config["path"],
                    dataset_name=src_config["name"],
                    text_field=src_config["field"],
                    target_bytes=src_config["target"],
                    split=src_config.get("split", "train"),
                )
                results[key] = "OK" if ok else "FAILED"
            except KeyboardInterrupt:
                results[key] = "INTERRUPTED"
                break

    print("\n--- Download Task Summary ---")
    for key, status in results.items():
        print(f"  {key}: {status}")

    print("\nNote: For Sanskrit (DCS & GRETIL), please download their specialized "
          "linguistic text files directly from their official repositories, as they "
          "are not standardly hosted via Hugging Face streaming APIs.")


if __name__ == "__main__":
    main()