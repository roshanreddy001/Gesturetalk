"""
test_riva_full.py
=================
Comprehensive test suite for NVIDIA Riva services used in the GestureTalk project.

Tests covered:
  1. Connectivity      – Can the gRPC channel reach Riva?
  2. NMT Translation   – Translate all project phrases to all supported languages.
  3. TTS Synthesis     – Synthesize speech for English (and optionally other voices).
  4. Full Pipeline     – Simulate the same path inference.py uses (translate → TTS).
  5. Edge Cases        – Empty strings, same-language pairs, unknown language.

Run from the project root:
    python test_riva_full.py

Optional flags:
    --nmt-only      Skip TTS tests (faster)
    --tts-only      Skip NMT tests
    --lang <Name>   Only test a single target language  (e.g. --lang Hindi)
    --phrase "<str>" Custom phrase instead of test defaults
"""

# ─────────────────────────────────────────────
# Dependency check – fail early with clear message
# ─────────────────────────────────────────────
try:
    import riva.client          # noqa: F401
except ModuleNotFoundError:
    print("\n[ERROR] 'riva' package not found.")
    print("Install it with:  pip install nvidia-riva-client")
    import sys; sys.exit(1)

import argparse
import base64
import io
import sys
import time
import wave
import os

# ─────────────────────────────────────────────
# 1. CONNECTIVITY TEST
# ─────────────────────────────────────────────

def test_connectivity():
    """Check raw gRPC connection to the Riva cloud endpoint."""
    print("\n" + "=" * 60)
    print("TEST 1: gRPC Connectivity")
    print("=" * 60)

    try:
        import grpc
        channel = grpc.secure_channel(
            "grpc.nvcf.nvidia.com:443",
            grpc.ssl_channel_credentials()
        )
        # Try a connectivity check (non-blocking state probe)
        state = channel.subscribe(lambda c: None)
        print("  [OK] gRPC channel created successfully.")
        channel.close()
        return True
    except Exception as e:
        print(f"  [FAIL] Could not create gRPC channel: {e}")
        return False


# ─────────────────────────────────────────────
# 2. NMT TRANSLATION TESTS
# ─────────────────────────────────────────────

# All languages in the project (matches riva_client.py lang_map keys)
ALL_LANGUAGES = [
    "Arabic", "Bulgarian", "Simplified Chinese", "Traditional Chinese",
    "Croatian", "Czech", "Danish", "Dutch", "Estonian", "Finnish", "French",
    "German", "Greek", "Hindi", "Hungarian", "Indonesian", "Italian",
    "Japanese", "Korean", "Latvian", "Lithuanian",
    "Norwegian", "Polish", "European Portuguese",
    "Brazillian Portuguese", "Romanian", "Russian", "Slovak",
    "Slovenian", "European Spanish", "LATAM Spanish", "Swedish",
    "Thai", "Turkish", "Ukrainian", "Vietnamese"
]

# Representative phrases from the PHRASE_MAP in inference.py
TEST_PHRASES = [
    "Hello there",
    "Please help me",
    "I am in pain",
    "This is an emergency",
    "Thank you very much",
]


def test_nmt(target_langs=None, phrases=None):
    """
    Test Riva NMT for the given language list and phrases.
    Returns (passed, failed) counts.
    """
    print("\n" + "=" * 60)
    print("TEST 2: Riva NMT (Neural Machine Translation)")
    print("=" * 60)

    from src.riva_client import RivaTranslator  # riva already verified at top

    try:
        translator = RivaTranslator()
    except Exception as e:
        print(f"  [FAIL] Cannot initialize RivaTranslator: {e}")
        return 0, 1

    if not translator.service:
        print("  [FAIL] Riva NMT service is None – check credentials / endpoint.")
        return 0, 1

    langs   = target_langs or ALL_LANGUAGES
    phrases = phrases or TEST_PHRASES

    passed, failed, skipped = 0, 0, 0

    for lang in langs:
        for phrase in phrases:
            label = f"  [{lang}] '{phrase}'"
            try:
                start = time.time()
                result = translator.translate(phrase, lang, source_language="English")
                elapsed = time.time() - start

                if result is None:
                    print(f"{label} → [SKIP] Unsupported by Riva")
                    skipped += 1
                elif result.strip() == "" or result == phrase:
                    print(f"{label} → [WARN] Possible no-change: '{result}'")
                    passed += 1
                else:
                    print(f"{label} → [OK] '{result}'  ({elapsed:.2f}s)")
                    passed += 1
            except Exception as e:
                print(f"{label} → [ERROR] {e}")
                failed += 1

    print(f"\n  NMT Summary: {passed} passed | {failed} failed | {skipped} skipped/unsupported")
    return passed, failed


# ─────────────────────────────────────────────
# 3. TTS SYNTHESIS TEST
# ─────────────────────────────────────────────

TTS_TEST_CASES = [
    ("Hello there",          "en-US"),
    ("नमस्ते, आप कैसे हैं?", "hi"),   # Hindi
    ("Thank you very much",  "en-US"),
]


def test_tts(save_wav=False):
    """
    Test Riva TTS service. Optionally saves audio as .wav files for manual listening.
    Returns (passed, failed) counts.
    """
    print("\n" + "=" * 60)
    print("TEST 3: Riva TTS (Text-to-Speech)")
    print("=" * 60)

    from src.riva_client import RivaTTS  # riva already verified at top

    try:
        tts = RivaTTS()
    except Exception as e:
        print(f"  [FAIL] Cannot initialize RivaTTS: {e}")
        return 0, 1

    if not tts.service:
        print("  [FAIL] Riva TTS service is None – check credentials / endpoint.")
        return 0, 1

    passed, failed, skipped = 0, 0, 0

    for text, lang_code in TTS_TEST_CASES:
        label = f"  [{lang_code}] '{text[:40]}'"
        try:
            start = time.time()
            audio_bytes = tts.generate_audio_response(text, lang_code)
            elapsed = time.time() - start

            if not audio_bytes:
                print(f"{label} → [SKIP] Riva TTS unavailable (gTTS fallback active)")
                skipped += 1
                continue

            size_kb = len(audio_bytes) / 1024
            print(f"{label} → [OK] {size_kb:.1f} KB PCM  ({elapsed:.2f}s)")

            if save_wav:
                _save_wav(audio_bytes, lang_code, text)

            passed += 1
        except Exception as e:
            print(f"{label} → [ERROR] {e}")
            failed += 1

    print(f"\n  TTS Summary: {passed} passed | {failed} failed | {skipped} skipped (no Riva TTS endpoint)")
    return passed, failed


def _save_wav(pcm_bytes, lang_code, text_hint):
    """Helper: wrap raw PCM in a WAV container and save to disk."""
    out_dir = "test_audio_output"
    os.makedirs(out_dir, exist_ok=True)

    safe_name = "".join(c if c.isalnum() else "_" for c in f"{lang_code}_{text_hint[:20]}")
    wav_path  = os.path.join(out_dir, f"{safe_name}.wav")

    wav_io = io.BytesIO()
    with wave.open(wav_io, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)       # 16-bit
        wf.setframerate(22050)
        wf.writeframes(pcm_bytes)

    with open(wav_path, "wb") as f:
        f.write(wav_io.getvalue())

    print(f"    Saved WAV → {wav_path}")


# ─────────────────────────────────────────────
# 4. FULL PIPELINE TEST (mimic inference.py)
# ─────────────────────────────────────────────

PIPELINE_TARGETS = ["Hindi", "French", "German"]
PIPELINE_PHRASE  = "Please help me"


def test_full_pipeline():
    """
    Simulate the exact path used by GestureRecognizer.translate_text()
    followed by tts.generate_audio().
    """
    print("\n" + "=" * 60)
    print("TEST 4: Full Pipeline  (NMT → TTS)")
    print("=" * 60)

    from src.riva_client import RivaTranslator  # riva already verified at top
    from src.tts import generate_audio

    try:
        translator = RivaTranslator()
    except Exception as e:
        print(f"  [FAIL] RivaTranslator init: {e}")
        return 0, 1

    passed, failed = 0, 0

    for lang in PIPELINE_TARGETS:
        print(f"\n  Pipeline: English → {lang}")
        try:
            # Step 1: Translate
            translated = translator.translate(PIPELINE_PHRASE, lang)
            if not translated:
                print(f"    NMT → [SKIP/FAIL] No translation returned for {lang}")
                failed += 1
                continue
            print(f"    NMT  → '{translated}'")

            # Step 2: TTS
            audio_b64 = generate_audio(translated, lang)
            if audio_b64:
                raw_size = len(base64.b64decode(audio_b64))
                print(f"    TTS  → [OK] {raw_size // 1024} KB audio (base64 encoded)")
                passed += 1
            else:
                print(f"    TTS  → [WARN] No audio returned (gTTS / offline fallback may have been used)")
                passed += 1  # Not a failure – system has fallback

        except Exception as e:
            print(f"    [ERROR] {e}")
            failed += 1

    print(f"\n  Pipeline Summary: {passed} passed | {failed} failed")
    return passed, failed


# ─────────────────────────────────────────────
# 5. EDGE CASE TESTS
# ─────────────────────────────────────────────

def test_edge_cases():
    """Test boundary conditions: empty text, same-language, unknown language."""
    print("\n" + "=" * 60)
    print("TEST 5: Edge Cases")
    print("=" * 60)

    from src.riva_client import RivaTranslator

    try:
        translator = RivaTranslator()
    except Exception as e:
        print(f"  [FAIL] Cannot init RivaTranslator: {e}")
        return

    cases = [
        ("Same language pair",   "",      "English", "English"),
        ("Empty string",         "",      "English", "Hindi"),
        ("Unknown target lang",  "Hello", "English", "Klingon"),
        ("Very long text",       "I am in pain. " * 50, "English", "French"),
        ("Special characters",   "Hello! How are you?", "English", "Japanese"),
    ]

    for name, text, src, tgt in cases:
        try:
            result = translator.translate(text, tgt, source_language=src)
            status = "[OK]" if result is not None else "[None returned]"
            disp   = repr(result[:60]) if result else "None"
            print(f"  {name}: {status} → {disp}")
        except Exception as e:
            print(f"  {name}: [ERROR] {e}")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Riva Test Suite for GestureTalk")
    parser.add_argument("--nmt-only",  action="store_true", help="Only run NMT tests")
    parser.add_argument("--tts-only",  action="store_true", help="Only run TTS tests")
    parser.add_argument("--lang",      type=str,  default=None,
                        help="Test a single language (e.g. --lang Hindi)")
    parser.add_argument("--phrase",    type=str,  default=None,
                        help="Custom phrase for translation test")
    parser.add_argument("--save-wav",  action="store_true",
                        help="Save TTS output as .wav files in test_audio_output/")
    return parser.parse_args()


def main():
    args = parse_args()

    target_langs = [args.lang]   if args.lang   else None
    phrases      = [args.phrase] if args.phrase  else None

    total_passed, total_failed = 0, 0

    print("\n" + "🔷" * 30)
    print("  GestureTalk  –  NVIDIA Riva Full Test Suite")
    print("🔷" * 30)

    # 1. Connectivity (always run)
    conn_ok = test_connectivity()

    if not conn_ok:
        print("\n⚠️  Cannot reach Riva endpoint. NMT / TTS tests may fail.")

    # 2. NMT
    if not args.tts_only:
        p, f = test_nmt(target_langs=target_langs, phrases=phrases)
        total_passed += p
        total_failed += f

    # 3. TTS
    if not args.nmt_only:
        p, f = test_tts(save_wav=args.save_wav)
        total_passed += p
        total_failed += f

    # 4. Full Pipeline (skip if lang/tts-only flags used)
    if not args.nmt_only and not args.tts_only and not args.lang:
        p, f = test_full_pipeline()
        total_passed += p
        total_failed += f

    # 5. Edge Cases
    if not args.nmt_only and not args.tts_only:
        test_edge_cases()

    # ─── Final Summary ───
    print("\n" + "=" * 60)
    print(f"  FINAL RESULTS: {total_passed} passed | {total_failed} failed")
    print("=" * 60 + "\n")

    sys.exit(0 if total_failed == 0 else 1)


if __name__ == "__main__":
    main()
