"""
Text processing pipeline for manga OCR output.

This module handles the full text processing chain for converting raw OCR text
(typically from PaddleOCR) into clean, natural-sounding text suitable for TTS.

Pipeline stages:
  1. cleanup_text()           - Basic OCR noise removal, normalization, deduplication
  2. apply_manga_corrections() - Domain-specific manga/character name corrections
  3. reconstruct_ocr_text()   - Word segmentation, spell correction, merged-word splitting
  4. format_text_for_speech()  - Final formatting for natural TTS output

Public API:
  - cleanup_text(text) -> str
  - load_manga_corrections() -> None
  - apply_manga_corrections(text) -> str
  - reconstruct_ocr_text(text) -> str
  - format_text_for_speech(text) -> str
"""

import json
import logging
import os
import re
import importlib.resources

log = logging.getLogger(__name__)


# =============================================================================
# Module-level globals (word splitters, spell checker, manga corrections)
# =============================================================================

manga_corrections = None

# Word splitting: use wordninja (better at deeply merged words) as primary,
# wordsegment as fallback. Both are loaded if available.
_word_splitter = None
_have_wordninja = False
_have_wordsegment = False
try:
    import wordninja
    _have_wordninja = True
    _word_splitter = 'wordninja'
    log.info('wordninja loaded OK (primary word splitting)')
except ImportError:
    pass
try:
    import wordsegment
    wordsegment.load()
    _have_wordsegment = True
    if not _have_wordninja:
        _word_splitter = 'wordsegment'
    log.info('wordsegment loaded OK (secondary word splitting)')
except ImportError:
    pass
if not _word_splitter:
    log.warning('No word splitter installed! Run: pip install wordninja wordsegment')

# SymSpell for fast spell checking (1M words/sec)
_symspell = None
try:
    from symspellpy import SymSpell
    _symspell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
    dict_path = str(importlib.resources.files('symspellpy').joinpath('frequency_dictionary_en_82_765.txt'))
    _symspell.load_dictionary(dict_path, term_index=0, count_index=1)
    log.info(f'SymSpell loaded OK ({_symspell.word_count} words)')
except Exception as e:
    log.warning(f'SymSpell not available: {e}. Spell checking disabled.')

# Protected vocabulary — words that should never be spell-corrected
_protected_vocab = set()
try:
    from config import PROTECTED_VOCAB_PATH
    if os.path.isfile(PROTECTED_VOCAB_PATH):
        with open(PROTECTED_VOCAB_PATH) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    _protected_vocab.add(line.lower())
        log.info(f'Protected vocabulary loaded: {len(_protected_vocab)} terms')
except Exception as e:
    log.warning(f'Protected vocabulary not loaded: {e}')


# =============================================================================
# Manga corrections dictionary
# =============================================================================

def load_manga_corrections():
    """Load manga-specific OCR corrections from manga_corrections.json."""
    global manga_corrections
    corrections_path = os.path.join(os.path.dirname(__file__), 'manga_corrections.json')
    if os.path.isfile(corrections_path):
        with open(corrections_path) as f:
            manga_corrections = json.load(f)
        word_count = len(manga_corrections.get('word_corrections', {}))
        pattern_count = len(manga_corrections.get('pattern_corrections', []))
        log.info(f'Manga corrections loaded: {word_count} words, {pattern_count} patterns')
    else:
        log.info('No manga_corrections.json found, skipping corrections.')
        manga_corrections = {'word_corrections': {}, 'pattern_corrections': [], 'common_sfx': {}}


def _auto_spellcheck(text):
    """Fix OCR spelling errors using SymSpell dictionary.
    Only fixes words that are clearly wrong (not in dictionary) and have a close match.
    Skips SFX-like words (repeated chars, all consonants, etc.)."""
    if not _symspell or not text or not text.strip():
        return text

    words = text.split()
    fixed = []
    changes = 0
    for word in words:
        # Preserve punctuation
        stripped = re.sub(r'^[^a-zA-Z]+|[^a-zA-Z]+$', '', word)
        prefix = word[:word.index(stripped)] if stripped and stripped in word else ''
        suffix = word[len(prefix) + len(stripped):] if stripped else ''

        if not stripped or len(stripped) <= 2:
            fixed.append(word)
            continue

        core = stripped.lower()

        # Skip protected vocabulary (character names, manga terms, etc.)
        if core in _protected_vocab:
            fixed.append(word)
            continue

        # Skip SFX-like words: repeated chars (aaah, grrr, ooo), all consonants, etc.
        if re.match(r'^(.)\1{2,}$', core):  # "aaa", "ooo"
            fixed.append(word)
            continue
        if re.match(r'^[^aeiou]+$', core) and len(core) <= 5:  # all consonants like "grrr"
            fixed.append(word)
            continue
        has_repeated = any(core.count(c) >= 3 for c in set(core))
        if has_repeated and len(core) <= 6:  # "gyaaa", "oooa"
            fixed.append(word)
            continue

        # Skip if already a known word
        exact = _symspell.lookup(core, max_edit_distance=0, verbosity=0)
        if exact:
            fixed.append(word)
            continue

        # Try to fix with edit distance 1
        suggestions = _symspell.lookup(core, max_edit_distance=1, verbosity=0)
        if suggestions and suggestions[0].distance == 1:
            corrected = suggestions[0].term
            # Don't "correct" to a completely different word (e.g. gya -> gay)
            # Only allow corrections that keep at least 60% of original chars in order
            if len(core) >= 4 and corrected[0] != core[0]:
                fixed.append(word)
                continue
            # Preserve original casing
            if stripped.isupper():
                corrected = corrected.upper()
            elif stripped[0].isupper():
                corrected = corrected.capitalize()
            fixed.append(f'{prefix}{corrected}{suffix}')
            changes += 1
        else:
            fixed.append(word)

    result = ' '.join(fixed)
    if changes > 0:
        log.info(f'    [spellfix] {changes} corrections: "{text[:80]}" -> "{result[:80]}"')
    return result


def apply_manga_corrections(text):
    """Apply manga-specific OCR corrections from the dictionary."""
    if not manga_corrections or not text:
        return text

    # Word-level corrections (case-insensitive)
    word_map = manga_corrections.get('word_corrections', {})
    if word_map:
        words = text.split()
        corrected = []
        for word in words:
            # Strip punctuation for lookup, preserve it
            stripped = word.strip('.,!?;:\'"()[]{}')
            prefix = word[:len(word) - len(word.lstrip('.,!?;:\'"()[]{}'))]
            suffix = word[len(stripped) + len(prefix):]
            lookup = stripped.lower()
            if lookup in word_map:
                replacement = word_map[lookup]
                # Match original case style
                if stripped.isupper():
                    replacement = replacement.upper()
                elif stripped[0].isupper() if stripped else False:
                    replacement = replacement.capitalize()
                corrected.append(prefix + replacement + suffix)
            else:
                corrected.append(word)
        text = ' '.join(corrected)

    # Pattern corrections (regex)
    for rule in manga_corrections.get('pattern_corrections', []):
        pattern = rule.get('pattern', '')
        replacement = rule.get('replacement', '')
        if pattern:
            text = re.sub(pattern, replacement, text)

    # Sound effects normalization for TTS
    sfx_map = manga_corrections.get('common_sfx', {})
    if sfx_map:
        words = text.split()
        corrected = []
        for word in words:
            stripped = word.strip('.,!?;:')
            suffix = word[len(stripped):]
            if stripped.upper() in sfx_map:
                corrected.append(sfx_map[stripped.upper()] + suffix)
            else:
                corrected.append(word)
        text = ' '.join(corrected)

    return text


# =============================================================================
# Text cleanup helpers
# =============================================================================

def join_spaced_letters(text):
    """Join single spaced letters: 'H E L L O' -> 'HELLO'."""
    # Match sequences of single letters separated by spaces
    def replace_spaced(m):
        return m.group(0).replace(' ', '')
    # Pattern: single letter, space, single letter, (repeat)
    text = re.sub(r'\b([A-Za-z]) (?=[A-Za-z](?:\b| ))', replace_spaced, text)
    # Catch remaining pairs at end of string
    text = re.sub(r'\b([A-Za-z]) ([A-Za-z])\b', r'\1\2', text)
    return text

def join_dotted_letters(text):
    """Join dotted letters: 'H.E.L.L.O' -> 'HELLO'."""
    return re.sub(r'\b([A-Za-z])\.([A-Za-z])\.', lambda m: m.group(1) + m.group(2), text)

def fix_slash_as_l(text):
    """Fix OCR slash errors: 'ca/ed' -> 'called', 'contro/' -> 'control'.
    Also: '/ explained' -> 'I explained', '/'m' -> 'I'm'.
    Never touch numeric fractions like 1/2."""
    # Slash standing alone or at start = 'I' (very common manga OCR error)
    # "/ explained" -> "I explained", "/didnt" -> "Ididnt", "/'m" -> "I'm"
    text = re.sub(r"(?:^|(?<=\s))/(?=\s|[a-zA-Z'\u2019])", 'I', text)
    # Slash between letter and hyphen -> 'I': "ASSASS/-" -> "ASSASSI-"
    text = re.sub(r'([a-zA-Z])/-', r'\1I-', text)
    # Slash between letters -- in manga OCR, slash almost always = 'I' (the pronoun)
    # Strategy: check if both sides are complete English words -> slash = 'I' separator
    # If one side is a word fragment -> slash = 'l' (lowercase L) inside a word
    _COMMON_WORDS_SLASH = {
        'if', 'is', 'it', 'in', 'at', 'an', 'as', 'am', 'be', 'by', 'do', 'go',
        'he', 'me', 'my', 'no', 'of', 'oh', 'ok', 'on', 'or', 'so', 'to', 'up',
        'us', 'we', 'the', 'and', 'are', 'but', 'can', 'did', 'for', 'get', 'got',
        'had', 'has', 'her', 'him', 'his', 'how', 'its', 'let', 'may', 'not', 'now',
        'our', 'out', 'own', 'ran', 'run', 'say', 'she', 'the', 'too', 'two', 'was',
        'who', 'why', 'yes', 'yet', 'you', 'all', 'any', 'ask', 'big', 'end', 'far',
        'few', 'saw', 'tried', 'spent', 'took', 'went', 'came', 'knew', 'said',
        'made', 'gave', 'kept', 'left', 'lost', 'told', 'felt', 'met', 'sat',
        'cut', 'hit', 'put', 'set', 'read', 'ate', 'ran', 'drew', 'flew', 'grew',
        'whether', 'dont', 'didnt', 'wasnt', 'couldnt', 'shouldnt', 'wouldnt',
    }
    def _slash_to_l(m):
        before, after = m.group(1), m.group(2)
        bl, al = before.lower(), after.lower()
        # If BOTH sides look like complete words -> slash = 'I' (pronoun)
        before_is_word = bl in _COMMON_WORDS_SLASH or len(bl) >= 3
        after_is_word = al in _COMMON_WORDS_SLASH or len(al) >= 3
        if before_is_word and after_is_word:
            return before + ' I ' + after
        # If one side is very short (1-2 chars) -> probably 'l' inside a word
        # e.g. "ca/ed" -> "called", "contro/s" -> "controls"
        return before + 'l' + after
    text = re.sub(r'([a-zA-Z]+)/([a-zA-Z]+)', _slash_to_l, text)
    # Slash at end of word after letters -> 'l'
    text = re.sub(r'([a-zA-Z])/$', r'\1l', text)
    return text

def fix_mixed_alphanumeric(text):
    """Fix digits inside words: G0T->GOT, 0KAY->OKAY. Preserve standalone numbers."""
    digit_map = {'0': 'O', '1': 'I', '5': 'S', '8': 'B', '6': 'G'}

    def fix_word(m):
        word = m.group(0)
        # All digits = real number, keep as-is
        if word.isdigit():
            return word
        # Ordinals like 1ST, 2ND, 3RD
        if re.match(r'^\d+(ST|ND|RD|TH)$', word, re.IGNORECASE):
            return word
        # Mixed: replace digits surrounded by letters
        result = []
        for i, ch in enumerate(word):
            if ch.isdigit():
                before = (word[i - 1].isalpha() or word[i - 1] in "''\u2019") if i > 0 else False
                after = (word[i + 1].isalpha() or word[i + 1] in "''\u2019") if i < len(word) - 1 else False
                if before or after:
                    result.append(digit_map.get(ch, ch))
                else:
                    result.append(ch)
            else:
                result.append(ch)
        return ''.join(result)

    return re.sub(r'\b\S+\b', fix_word, text)

def remove_duplicate_words(text):
    """Remove immediately duplicated words: 'the the cat' -> 'the cat'."""
    return re.sub(r'\b(\w+)(\s+\1)+\b', r'\1', text, flags=re.IGNORECASE)

def remove_duplicate_sentences(text):
    """Remove duplicate sentences/lines."""
    lines = text.split('\n')
    seen = set()
    unique = []
    for line in lines:
        normalized = line.strip().lower()
        if normalized and normalized not in seen:
            seen.add(normalized)
            unique.append(line.strip())
        elif not normalized:
            unique.append('')
    return '\n'.join(unique)


_REAL_SHORT_WORDS = {
    'a', 'i', 'an', 'am', 'as', 'at', 'be', 'by', 'do', 'go', 'ha', 'he',
    'hi', 'if', 'in', 'is', 'it', 'me', 'mr', 'ms', 'my', 'no', 'of', 'oh',
    'ok', 'on', 'or', 'so', 'to', 'up', 'us', 'we', 'ah', 'eh', 'hm', 'uh',
    'um', 'ox', 'ax',
}

def _clean_noise_fragments(txt):
    """Remove stray 1-2 letter tokens that aren't real English words.
    e.g. 'ii', 'ff', 'qt', 'li', 'ex', 'fy', 'u', 'ny'"""
    words = txt.split()
    cleaned = []
    for w in words:
        core = re.sub(r'[^a-zA-Z]', '', w).lower()
        # Keep if: 3+ alpha chars, or a known short word, or purely punctuation/number
        if len(core) >= 3 or core in _REAL_SHORT_WORDS or len(core) == 0:
            cleaned.append(w)
        else:
            # Keep if it's a stutter with trailing punctuation (e.g. "H..." "B-")
            if re.match(r'^[A-Z][\.\!\?…\-]+$', w):
                cleaned.append(w)
            # else: drop silently
    return ' '.join(cleaned)


# =============================================================================
# Main cleanup pipeline
# =============================================================================

def cleanup_text(text):
    """Full text cleanup pipeline."""
    if not text:
        return ''
    # Normalize whitespace
    text = text.strip()
    text = re.sub(r'\r\n', '\n', text)

    # Strip "SFX:" / "SFX =" / "Sfx:" prefix (OCR reading manga SFX labels)
    text = re.sub(r'^SFX\s*[:=\-]\s*', '', text, flags=re.IGNORECASE).strip()
    text = re.sub(r'[ \t]+', ' ', text)

    # Normalize curly/smart quotes to straight quotes before anything else
    text = text.replace('\u2018', "'").replace('\u2019', "'")  # ' '
    text = text.replace('\u201C', '"').replace('\u201D', '"')  # " "

    # Fix spaced contractions: "phone' s" -> "phone's", "don' t" -> "don't"
    text = re.sub(r"(\w)'\s+(s|t|re|ll|ve|m|d)\b", r"\1'\2", text, flags=re.IGNORECASE)

    # Remove non-Latin characters (OCR noise from manga -- Hebrew, CJK, etc.)
    # Keep: Latin letters, digits, common punctuation, whitespace
    text = re.sub(r'[^\x00-\x7F\u00C0-\u024F]+', '', text)

    # Remove stray $ signs attached to words (OCR artifact)
    text = re.sub(r'\$\s*', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()

    # Fix lowercase 'l' misread as 'I' -- very common PaddleOCR error in manga
    # Standalone "l" between words -> "I": "so l didn't" -> "so I didn't"
    text = re.sub(r'\bl\b', 'I', text)
    # "l'" at word start -> "I'": "l'm" -> "I'm", "l've" -> "I've"
    text = re.sub(r"\bl'", "I'", text)
    # "lt" at word start when followed by "'s" or space -> "It": "lt's" -> "It's"
    text = re.sub(r'\blt\b', 'It', text)

    # Fix spaced hyphenations: "class i- fie d" -> "classi-fied"
    # Step 1: remove space before hyphen at end: "class i -" -> "classi-"
    text = re.sub(r'(\w)\s+-\s*$', r'\1-', text)
    text = re.sub(r'(\w)\s+-\s+', r'\1-', text)
    # Step 2: remove space after hyphen: "classi- fied" -> "classi-fied"
    text = re.sub(r'-\s+(\w)', r'-\1', text)

    # Join spaced/dotted letters
    text = join_spaced_letters(text)
    text = join_dotted_letters(text)

    # Fix OCR errors
    text = fix_slash_as_l(text)
    text = fix_mixed_alphanumeric(text)

    # PaddleOCR-specific: fix apostrophe-joined words: "I'MA" -> "I'M A"
    text = re.sub(r"\b(I'M|HE'S|SHE'S|IT'S|WE'RE|THEY'RE|YOU'RE|DON'T|CAN'T|WON'T|DIDN'T|ISN'T|WASN'T|AREN'T|WEREN'T|HAVEN'T|HASN'T|WOULDN'T|COULDN'T|SHOULDN'T)([A-Z])",
                  lambda m: m.group(1) + ' ' + m.group(2), text, flags=re.IGNORECASE)

    # Rejoin hyphenated words BEFORE word splitting (so "ASSASSI-NATION" -> "ASSASSINATION")
    # Also handles merged text: "ANDSHEIMMEDI-ATELY" -> "ANDSHEIMMEDIATELY" -> split later
    def _rejoin_hyphen_cleanup(m):
        left, right = m.group(1), m.group(2)
        joined = left + right
        # Short fragments -> always join (they're clearly one word split by line break)
        if len(left) < 4 or len(right) < 4:
            return joined
        if _word_splitter:
            parts = _split_words(joined)
            # If wordsegment can split it into real words, join them
            # (the word splitting step later will re-split properly)
            if len(parts) >= 1 and all(len(p) >= 2 for p in parts):
                return joined  # Always join -- let word splitting handle it later
        return joined  # Join by default for hyphen-split words
    text = re.sub(r'(\w+)-(\w+)', _rejoin_hyphen_cleanup, text)

    # Remove duplicates
    text = remove_duplicate_words(text)
    text = remove_duplicate_sentences(text)

    # Strip stray parentheses/brackets attached to words: "ha)" -> "ha", "(the" -> "the"
    text = re.sub(r'\((\w)', r'\1', text)  # leading (
    text = re.sub(r'(\w)\)', r'\1', text)  # trailing )
    text = re.sub(r'\[(\w)', r'\1', text)  # leading [
    text = re.sub(r'(\w)\]', r'\1', text)  # trailing ]

    # Remove OCR noise fragments (stray 1-2 letter non-words)
    text = _clean_noise_fragments(text)

    # Remove stray symbols attached to or near words: "=do" -> "do", "_the" -> "the"
    text = re.sub(r'[=_<>{}|\\^`~]+', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()

    # Remove stray quote+number combos: "4 or '4 (OCR reading visual artifacts as text)
    text = re.sub(r'["\'](?=\d\b)', '', text)

    # Strip isolated single digits (OCR reading panel numbers/art as text)
    # "Not yet!! 4." -> "Not yet!!", "A! A 4 5." -> "A! A."
    # But keep digits that are part of real numbers: "20 years" stays
    text = re.sub(r'\b\d\b\.?', '', text)
    text = re.sub(r'\s+', ' ', text).strip()

    # Strip leading digits stuck to words: "4however" -> "however", "15over" -> "over"
    text = re.sub(r'\b\d+([a-zA-Z]{3,})', r'\1', text)

    # Strip trailing digits stuck to words: "ino1" -> "ino", "monster2" -> "monster"
    text = re.sub(r'([a-zA-Z]{2,})\d+\b', r'\1', text)

    # Strip stray unbalanced quotes at start/end
    if text.startswith('"') and '"' not in text[1:]:
        text = text[1:].strip()
    if text.endswith('"') and '"' not in text[:-1]:
        text = text[:-1].strip()

    # Final trim
    text = text.strip()
    # Collapse multiple newlines
    text = re.sub(r'\n{3,}', '\n\n', text)
    # Replace remaining newlines with spaces for TTS (single bubble = one utterance)
    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()

    return text


# =============================================================================
# Word splitting (wordninja / wordsegment)
# =============================================================================

def _split_words(text_segment):
    """Split a merged word segment into individual words.
    Uses wordninja (better at deeply merged text) with validation:
    - If wordninja over-splits a compound word (e.g. "swordsmanship" -> ["swordsman","ship"]),
      check with SymSpell if the original is a known word and keep it whole.
    - Falls back to wordsegment if wordninja unavailable."""
    lower = text_segment.lower()

    # If the whole segment is a known word, don't split it
    if _symspell:
        results = _symspell.lookup(lower, max_edit_distance=0, verbosity=0)
        if results:
            return [lower]

    if _have_wordninja:
        parts = wordninja.split(lower)
        # Validate: if any part is very short (1-2 chars) and not a real word,
        # try wordsegment as second opinion
        if _have_wordsegment and len(parts) >= 2:
            has_bad = any(len(p) <= 2 and p not in ('a', 'i', 'an', 'am', 'as', 'at',
                'be', 'by', 'do', 'go', 'he', 'if', 'in', 'is', 'it', 'me', 'my',
                'no', 'of', 'oh', 'ok', 'on', 'or', 'so', 'to', 'up', 'us', 'we')
                for p in parts)
            if has_bad:
                ws_parts = wordsegment.segment(lower)
                ws_bad = any(len(p) <= 2 and p not in ('a', 'i') for p in ws_parts)
                if not ws_bad and len(ws_parts) <= len(parts):
                    return ws_parts
        return parts
    elif _have_wordsegment:
        return wordsegment.segment(lower)
    return [text_segment]

def _split_alpha_segment(segment):
    """Split a purely alphabetic segment using word splitter.
    Returns the split words as a list, preserving original case."""
    if len(segment) <= 2:
        return [segment]

    if _word_splitter is None:
        return [segment]

    parts = _split_words(segment)

    # If splitter returns only 1 part, no split needed
    if len(parts) <= 1:
        return [segment]

    # If the split produces mostly tiny fragments (all <= 3 chars) from a long word,
    # check if the unsplit word is close to a real word (OCR dropped a letter)
    # e.g. "attentne" -> ["at","tent","ne"] but should be "attentive"
    # BUT: if all split parts are real dictionary words, prefer the split.
    # e.g. "arevery" -> ["are","very"] is better than "every"
    if _symspell and len(segment) >= 7 and all(len(p) <= 4 for p in parts):
        # Skip protected vocabulary — don't split or correct protected terms
        if segment.lower() in _protected_vocab:
            return [segment]
        # First check: are ALL split parts real dictionary words?
        all_parts_real = all(
            _symspell.lookup(p, max_edit_distance=0, verbosity=0) or len(p) <= 1
            for p in parts
        )
        if not all_parts_real:
            suggestions = _symspell.lookup(segment.lower(), max_edit_distance=2, verbosity=0)
            if suggestions and suggestions[0].distance <= 2:
                fix = suggestions[0].term
                return [fix]

    # Rebuild with original case by walking through the original string
    result = []
    pos = 0
    for part in parts:
        orig_part = segment[pos:pos + len(part)]
        result.append(orig_part)
        pos += len(part)

    return result


# =============================================================================
# OCR text reconstruction helpers
# =============================================================================

def _split_on_punct_boundaries(text):
    """Split text like 'OKAY.I'M' into tokens respecting punctuation.
    Inserts spaces after sentence-ending punctuation (.!?) when followed
    by a letter, and after commas/semicolons when followed by a letter."""
    # Add space after .!? when directly followed by a letter or quote
    text = re.sub(r'([.!?])([A-Za-z"\'])', r'\1 \2', text)
    # Add space after , or ; when directly followed by a letter
    text = re.sub(r'([,;])([A-Za-z])', r'\1 \2', text)
    return text


def _fix_contraction_merges(text):
    """Fix merged contractions: SHE'SA -> SHE'S A, HE'SNOT -> HE'S NOT, etc.
    PaddleOCR often merges the word after a contraction into it."""
    if not text:
        return text

    # Pattern: known contraction endings followed directly by another word
    # e.g. SHE'SA -> SHE'S A, IT'SNOT -> IT'S NOT, DON'TKNOW -> DON'T KNOW
    contraction_endings = (
        r"(?:'S|'T|'RE|'LL|'VE|'M|'D"  # standard
        r"|'s|'t|'re|'ll|'ve|'m|'d)"    # lowercase
    )
    # Insert space between contraction ending and the next word
    text = re.sub(
        rf"(\w{contraction_endings})([A-Z][a-z]|[A-Z]{{2,}})",
        r'\1 \2', text
    )
    # Also handle lowercase: "she'sa" -> "she's a", "kid'sa" -> "kid's a"
    text = re.sub(
        rf"(\w{contraction_endings})([a-z]+)",
        lambda m: m.group(1) + ' ' + m.group(2),
        text
    )

    # Pre-contraction merges: "alli've" -> "all i've", "goti'm" -> "got i'm"
    # When text is merged BEFORE an I-contraction (I've, I'll, I'm, I'd)
    text = re.sub(r"\b(\w{2,})(i'(?:ve|ll|m|d))\b", r'\1 \2', text, flags=re.IGNORECASE)

    return text


def _fix_common_merged_words(text):
    """Fix very common OCR merge patterns that wordninja sometimes misses.
    These are high-confidence fixes based on common manga dialogue patterns."""
    if not text:
        return text

    # Fix "I" merged with next word: IMEAN -> I MEAN, IDON'T -> I DON'T, etc.
    # This is extremely common in manga OCR because "I" is a single character
    # that PaddleOCR joins to the next word.
    # Pattern: word-boundary "I" followed by a lowercase or uppercase word
    text = re.sub(r'\bI([A-Z][a-z]{2,})', r'I \1', text)  # IMean -> I Mean
    text = re.sub(r'\bI([A-Z]{2,})', r'I \1', text)        # IMEAN -> I MEAN
    # "I" + contraction: Idon't -> I don't, Ican't -> I can't
    text = re.sub(r"\bI(don't|can't|won't|didn't|haven't|wasn't|couldn't|shouldn't|wouldn't|isn't|aren't|weren't|hasn't|mustn't|needn't)",
                  r"I \1", text, flags=re.IGNORECASE)
    # "I" + common verbs: Ihave -> I have, Iwas -> I was, etc.
    text = re.sub(r"\bI(have|had|was|am|will|would|could|should|need|want|know|think|mean|feel|got|get|just|also|still|never|always|really)\b",
                  r"I \1", text, flags=re.IGNORECASE)

    # Reverse: "I" + fragment that should be joined back (OCR split a word starting with I/i)
    # "I mmediately" -> "immediately", "I mportant" -> "important", "I nstead" -> "instead"
    # Pattern: standalone "I" followed by a lowercase fragment starting with m/n/s/t/f
    def _rejoin_i_fragment(m):
        frag = m.group(1)
        joined = 'i' + frag
        # Known patterns (high confidence)
        known = {'mmediately': 'immediately', 'mportant': 'important', 'nstead': 'instead',
                 'mproving': 'improving', 'mpossible': 'impossible', 'ncredible': 'incredible',
                 'nteresting': 'interesting', 'magine': 'imagine', 'gnore': 'ignore',
                 'nvisible': 'invisible', 'mmense': 'immense', 'mmediate': 'immediate',
                 'mprove': 'improve', 'mpact': 'impact', 'nclude': 'include',
                 'ncrease': 'increase', 'ndeed': 'indeed', 'nside': 'inside',
                 'nnocent': 'innocent', 'ntense': 'intense', 'ntent': 'intent',
                 'nstinct': 'instinct', 'nstant': 'instant', 'nvite': 'invite'}
        if frag.lower() in known:
            return known[frag.lower()]
        if _word_splitter:
            splits = _split_words(joined)
            if len(splits) == 1:
                return splits[0]  # lowercase -- will be cased later by _to_sentence_case
        return m.group(0)  # no change
    text = re.sub(r'\bI ([a-zA-Z]{4,})\b', _rejoin_i_fragment, text)

    # Split contractions: "you re" -> "you're", "don t" -> "don't", "didn t" -> "didn't"
    text = re.sub(r"\b(you|we|they|I) (re|ve|ll|d)\b", r"\1'\2", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(he|she|it|that|what|there|who|how) (s|d|ll)\b", r"\1'\2", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(do|does|did|is|are|was|were|has|have|had|could|would|should|can|won|isn|aren|wasn|weren|hasn|haven|hadn|couldn|wouldn|shouldn|mustn|needn|ain) ?(t)\b", r"\1'\2", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(can) (not)\b", r"\1'\2", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(I) (m)\b", r"\1'\2", text, flags=re.IGNORECASE)
    # Missing apostrophe contractions: "didnt" -> "didn't", "dont" -> "don't", "wont" -> "won't"
    text = re.sub(r"\b(didn|don|doesn|won|wouldn|couldn|shouldn|can|isn|aren|wasn|weren|hasn|haven|hadn|mustn|needn|ain)t\b", r"\1't", text, flags=re.IGNORECASE)

    # "BEA" at word boundary when preceded by "BE" context -> "BE A"
    # Catches: "MUST BEA" -> "MUST BE A", "WILL BEA" -> "WILL BE A"
    text = re.sub(r'\b(BE)A\b', r'\1 A', text, flags=re.IGNORECASE)

    # "a" merged with next word: aweek -> a week, alot -> a lot
    text = re.sub(r'\ba(week|lot|while|bit|few|little|long|good|great|big|new|old)\b',
                  r'a \1', text, flags=re.IGNORECASE)

    # Common multi-word merges
    common_merges = [
        (r'\bTOBE\b', 'TO BE'), (r'\bINTHE\b', 'IN THE'),
        (r'\bOFTHE\b', 'OF THE'), (r'\bFORTHE\b', 'FOR THE'),
        (r'\bISTHE\b', 'IS THE'), (r'\bISTHAT\b', 'IS THAT'),
        (r'\bITIS\b', 'IT IS'),
        (r'\bTHATIS\b', 'THAT IS'), (r'\bWHATIS\b', 'WHAT IS'),
        (r'\bTHISIS\b', 'THIS IS'),
        (r'\bNOTGONNA\b', 'NOT GONNA'),
        (r'\bINFRONT\b', 'IN FRONT'), (r'\bINCHFROM\b', 'INCH FROM'),
        (r'\bATALL\b', 'AT ALL'), (r'\bOFMY\b', 'OF MY'),
        (r'\bANDTHE\b', 'AND THE'), (r'\bBUTTHE\b', 'BUT THE'),
        (r'\bFRONTOF\b', 'FRONT OF'), (r'\bOUTOF\b', 'OUT OF'),
        (r'\bONCEA\b', 'ONCE A'), (r'\bSUCHA\b', 'SUCH A'),
        # Multi-word fragments from OCR line breaks
        (r'\bat ten tine\b', 'attentive'), (r'\bat ten tive\b', 'attentive'),
        (r'\bim medi at ely\b', 'immediately'),
        (r'\bI m proving\b', 'improving'), (r'\bIm proving\b', 'improving'),
        (r'\bswords man ship\b', 'swordsmanship'),
        (r'\bswordsman ship\b', 'swordsmanship'),
        # "YOUGOTA" -> "YOU GOT A", "YOUGOT" -> "YOU GOT"
        (r'\bYOUGOTA\b', 'YOU GOT A'), (r'\bYOUGOT\b', 'YOU GOT'),
        (r'\bDAMNIT\b', 'DAMN IT'), (r'\bDOIT\b', 'DO IT'), (r'\bTHROWAWAY\b', 'THROW AWAY'),
        (r'\bGOTTA\b', 'GOTTA'),
        (r'\bGONNA\b', 'GONNA'), (r'\bWANNA\b', 'WANNA'),
        (r'\bGOTA\b', 'GOT A'), (r'\bLOTA\b', 'LOT A'),
        (r'\bKINDA\b', 'KINDA'), (r'\bSORTA\b', 'SORTA'),
        (r'\bMAYBEI\b', 'MAYBE I'), (r'\bMAYBE\b', 'MAYBE'),
    ]
    for pattern, replacement in common_merges:
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

    return text


def _rejoin_fragmented_words(text):
    """Rejoin words that PaddleOCR split across multiple detection lines.

    PaddleOCR detects each text line separately. When lines are joined with spaces,
    words that span line breaks get fragmented:
      'swords man ship' -> 'swordsmanship'
      'im medi at ely' -> 'immediately'
      'at ten tine' -> 'attentive'
      'E hi ca' -> removed (garbage)

    Strategy: try joining adjacent small fragments and check if the result
    is a real English word using wordninja's vocabulary."""
    if not text:
        return text

    if not _word_splitter:
        return text

    words = text.split()
    if len(words) <= 1:
        return text

    result = []
    i = 0
    while i < len(words):
        # Skip punctuation-only tokens
        if not any(c.isalpha() for c in words[i]):
            result.append(words[i])
            i += 1
            continue

        # Try joining 2, 3, 4, 5 consecutive words and check if they form a real word
        best_join = None
        best_len = 0

        # Only try fragment joining if the CURRENT word is short (<=6 chars)
        # and NOT a common standalone English word
        current_core = re.sub(r'[^a-zA-Z]', '', words[i])
        # Single letters are never fragment starts (except rarely)
        if len(current_core) <= 1:
            result.append(words[i])
            i += 1
            continue
        if len(current_core) > 6:
            result.append(words[i])
            i += 1
            continue
        # Skip common standalone words -- they're real words, not fragments
        _COMMON_STANDALONE = {
            'a', 'i',  # single letters that are real words
            'an', 'am', 'as', 'at', 'be', 'by', 'do', 'go',
            'he', 'if', 'in', 'is', 'it', 'me', 'my', 'no', 'of', 'oh',
            'ok', 'on', 'or', 'so', 'to', 'up', 'us', 'we',
            'and', 'are', 'but', 'can', 'did', 'for', 'get', 'got',
            'had', 'has', 'her', 'him', 'his', 'how', 'its', 'let',
            'may', 'new', 'nor', 'not', 'now', 'old', 'one', 'our',
            'out', 'own', 'ran', 'run', 'say', 'she', 'the', 'too',
            'two', 'was', 'way', 'who', 'why', 'yes', 'yet', 'you',
            'all', 'any', 'ask', 'big', 'end', 'far', 'few', 'saw',
            'also', 'been', 'both', 'come', 'does', 'done', 'down',
            'each', 'even', 'from', 'gave', 'gone', 'good', 'have',
            'here', 'into', 'just', 'keep', 'know', 'last', 'like',
            'long', 'look', 'made', 'make', 'many', 'more', 'much',
            'must', 'name', 'next', 'only', 'over', 'said', 'same',
            'seem', 'some', 'sure', 'take', 'tell', 'than', 'that',
            'them', 'then', 'they', 'this', 'time', 'took', 'very',
            'want', 'well', 'went', 'were', 'what', 'when', 'will',
            'with', 'word', 'work', 'year', 'your', 'meat', 'days',
        }
        if current_core.lower() in _COMMON_STANDALONE:
            result.append(words[i])
            i += 1
            continue

        for span in range(4, 1, -1):  # try 2-4 fragments
            if i + span > len(words):
                continue

            fragments = words[i:i+span]
            cores = []
            for f in fragments:
                core = re.sub(r'[^a-zA-Z]', '', f)
                cores.append(core)

            # Skip if any fragment is empty alpha
            if any(not c for c in cores):
                continue

            # At least 1 fragment must be short (<=4 chars) for 2-word joins,
            # at least 2 for 3+ word joins. "chal"+"lenge" should join to "challenge".
            short_count = sum(1 for c in cores if len(c) <= 4)
            min_short = 1 if span == 2 else 2
            if short_count < min_short:
                continue

            joined = ''.join(cores)
            if len(joined) < 5:
                continue

            # DON'T join if ALL fragments are already real dictionary words --
            # they're valid separate words, not OCR fragments.
            # e.g. "can't we slow down" should NOT become "cantweslowdown"
            if _symspell:
                all_real = True
                for c in cores:
                    cl = c.lower()
                    if cl in _COMMON_STANDALONE or len(cl) <= 1:
                        continue  # known real word
                    lookup = _symspell.lookup(cl, max_edit_distance=0, verbosity=0)
                    if not lookup:
                        all_real = False
                        break
                if all_real:
                    continue  # all fragments are real words -- don't merge

            # Check if joined text forms a recognized word
            # First try SymSpell: "challange" -> "challenge" (distance 1)
            if _symspell:
                sym_result = _symspell.lookup(joined.lower(), max_edit_distance=1, verbosity=0)
                if sym_result and sym_result[0].distance <= 1:
                    best_join = sym_result[0].term
                    best_len = span
                    break

            splits = _split_words(joined)

            if len(splits) == 1:
                # Perfect: single recognized word (e.g., 'immediately')
                best_join = joined
                best_len = span
                break
            elif len(splits) < span:
                # Fewer parts than fragments -- boundaries shifted = likely a real word
                cores_lower = [c.lower() for c in cores]
                splits_lower = [s.lower() for s in splits]
                if splits_lower != cores_lower:
                    # Check: if last split == last core >= 4 chars AND the last core
                    # is a common standalone word, don't include it (we'd eat a real word)
                    # But compound suffixes like -ship, -ment, -ness, -tion are OK
                    _COMPOUND_SUFFIXES = {'ship', 'ment', 'ness', 'tion', 'sion', 'ible',
                                         'able', 'ence', 'ance', 'ment', 'ling', 'ward',
                                         'like', 'wise', 'less', 'ful'}
                    last_core_l = cores_lower[-1]
                    if (splits_lower[-1] == last_core_l and len(last_core_l) >= 2
                            and last_core_l not in _COMPOUND_SUFFIXES):
                        continue
                    best_join = joined
                    best_len = span
                    break

        if best_join:
            # Preserve leading punctuation from first word
            lead = ''
            first = words[i]
            while first and not first[0].isalpha():
                lead += first[0]
                first = first[1:]
            # Preserve trailing punctuation from last word
            trail = ''
            last = words[i + best_len - 1]
            while last and not last[-1].isalpha():
                trail = last[-1] + trail
                last = last[:-1]

            result.append(lead + best_join + trail)
            i += best_len
        else:
            result.append(words[i])
            i += 1

    return ' '.join(result)


def _fix_ocr_letter_errors(text):
    """Fix common OCR character-level misreads.

    Common errors:
      'l' (lowercase L) misread as 'I' or vice versa
      'Ifl' -> 'If I'
      'l was' -> 'I was' (standalone lowercase L at word boundary)
    """
    if not text:
        return text

    # 'Ifl' -> 'If I' (very common OCR error -- l misread as I)
    text = re.sub(r'\b[Ii][Ff][Ll]\b', 'if I', text)

    # 'IfI' merged -- "IFIDIDN'T" -> "If I didn't"
    text = re.sub(r'\b[Ii][Ff][Ii](?=[a-z])', 'if I', text, flags=re.IGNORECASE)

    # 'I' merged with common verbs: "Ispent" -> "I spent", "Itried" -> "I tried"
    # After lowercasing these appear as "ispent", "itried", etc.
    _I_VERBS = (
        'spent|tried|tried|asked|began|came|can|could|couldn|continued|decided|did|didn|don|'
        'ended|felt|figured|found|gave|get|got|grew|had|have|heard|just|kept|knew|know|'
        'learned|left|let|looked|lost|made|managed|meant|met|might|must|need|never|noticed|'
        'only|put|ran|really|said|saw|seemed|should|spent|started|still|suppose|taught|'
        'think|thought|told|took|turned|understand|used|want|was|went|will|wish|woke|would'
    )
    text = re.sub(r'\bI(' + _I_VERBS + r')', r'I \1', text, flags=re.IGNORECASE)

    # Standalone 'l' that should be 'I' -- only when it's a standalone word
    # and surrounded by common English context
    text = re.sub(r'\bl\b(?=\s+(?:was|am|had|have|will|would|could|should|need|want|know|think|mean|feel|got|get|just|also|still|never|always|really|can|could|did|do|don|went|said|saw|came|told|thought|found|made|took|gave|let|tried|used|kept|called|heard|began|seemed|looked|turned|asked|moved|started|became|left|went|ran|came|put|set|took|read|wrote|ate|met|sat|cut|hit|bit|shut|hurt|lost|spent|paid|sent|built|taught|caught|brought|held|stood|spoke|chose|broke|wore|drove|rode|flew|grew|knew|drew|threw|gave|forgot|understood|woke))',
                  'I', text, flags=re.IGNORECASE)
    # 'l' before 'm, 'll, 've, 'd -- these are contractions with I
    text = re.sub(r"\bl'(m|ll|ve|d)\b", r"I'\1", text)

    # 'l' merged with verbs/contractions without space: "ldidn't" -> "I didn't", "ltried" -> "I tried"
    _L_VERBS = (
        'didn|don|couldn|wouldn|shouldn|wasn|isn|aren|haven|hadn|won|'
        'tried|spent|asked|came|could|continued|decided|did|ended|felt|'
        'found|gave|got|grew|had|have|heard|just|kept|knew|learned|left|'
        'looked|lost|made|managed|meant|met|need|never|noticed|put|ran|'
        'really|said|saw|seemed|should|started|still|think|thought|told|'
        'took|turned|used|want|was|went|will|wish|woke|would'
    )
    text = re.sub(r'\bl(' + _L_VERBS + r')', r'I \1', text, flags=re.IGNORECASE)

    return text


# =============================================================================
# OCR text reconstruction (main entry point)
# =============================================================================

def reconstruct_ocr_text(text):
    """Fix merged words and OCR spacing errors using wordninja.
    Runs after cleanup_text(), before format_text_for_speech().

    Examples:
      OKAY.I'M NOTGONNA -> OKAY. I'M NOT GONNA
      Intheeventyou findyourselfinMato -> In the event you find yourself in Mato
      unnecessaryexertion -> unnecessary exertion
      SHE'SA FAMOUS -> SHE'S A FAMOUS
      MUST BEA -> MUST BE A
      swords man ship -> swordsmanship
      Ifl -> If I
    """
    if not text:
        return text

    # CRITICAL: Normalize ALL CAPS to lowercase FIRST.
    # PaddleOCR returns ALL CAPS text. All our pattern matching and wordninja
    # work best on lowercase. We'll re-case to sentence case in format_text_for_speech().
    alpha_chars = [c for c in text if c.isalpha()]
    if alpha_chars:
        upper_ratio = sum(1 for c in alpha_chars if c.isupper()) / len(alpha_chars)
        if upper_ratio > 0.6:
            # Mostly uppercase -- convert to lowercase but preserve "I" as standalone word
            text = text.lower()
            # Restore standalone "I" (the pronoun)
            text = re.sub(r'\bi\b', 'I', text)
            # Restore "I'" contractions: "i'm" -> "I'm", "i'll" -> "I'll"
            text = re.sub(r"\bI'(m|ll|ve|d)\b", r"I'\1", text)

    # Step 0: Fix OCR letter-level misreads (l->I, Ifl->If I)
    text = _fix_ocr_letter_errors(text)

    # Step 0a: Fix contraction merges (SHE'SA -> SHE'S A) BEFORE general splitting
    text = _fix_contraction_merges(text)

    # Step 0b: Fix common merged word patterns
    text = _fix_common_merged_words(text)

    # Step 0c: Rejoin fragmented words EARLY -- before SymSpell corrects fragments
    # PaddleOCR splits "CHALLENGE" into "CHAL LENGE". If SymSpell runs first,
    # it "corrects" each fragment individually: "chal" -> "chat", "lenge" -> "ledge".
    # By rejoining first, we get "challenge" before SymSpell can mangle the fragments.
    text = _rejoin_fragmented_words(text)

    # Step 1: Fix punctuation-joined words (OKAY.I'M -> OKAY. I'M)
    text = _split_on_punct_boundaries(text)

    # Step 1b: SymSpell word-level correction BEFORE splitting
    # Fix OCR character substitution errors like ATTENTINE -> ATTENTIVE
    # Must run before wordsegment, otherwise "attentine" gets split to "attenti ne"
    if _symspell:
        pre_words = text.split()
        corrected_pre = []
        for w in pre_words:
            # Only correct pure alphabetic words >= 4 chars
            core = re.sub(r'[^a-zA-Z]', '', w)
            if len(core) >= 4 and core.isalpha() and "'" not in w:
                # Skip protected vocabulary
                if core.lower() in _protected_vocab:
                    corrected_pre.append(w)
                    continue
                # Try distance=1 first (safe)
                suggestions = _symspell.lookup(core.lower(), max_edit_distance=1, verbosity=0)
                if suggestions and suggestions[0].distance == 1:
                    fix = suggestions[0].term
                    # Rebuild word with original non-alpha chars
                    # Simple case: just replace the alpha content
                    if core == w:
                        w = fix if w.islower() else fix  # already lowercase from earlier
                    else:
                        # Has punctuation attached -- replace alpha portion
                        w = re.sub(r'[a-zA-Z]+', fix, w, count=1)
            corrected_pre.append(w)
        text = ' '.join(corrected_pre)

    # Step 2: Process each whitespace-delimited token
    words = text.split()
    result = []
    for word in words:
        # Separate leading and trailing punctuation
        lead = ''
        trail = ''
        core = word
        while core and not core[0].isalnum() and core[0] != "'":
            lead += core[0]
            core = core[1:]
        while core and not core[-1].isalnum() and core[-1] != "'":
            trail = core[-1] + trail
            core = core[:-1]

        if not core:
            result.append(word)
            continue

        # Skip very short words and numbers
        if len(core) <= 2 or core.isdigit():
            result.append(word)
            continue

        # Skip words with apostrophes -- contractions are fine as-is
        if "'" in core or "\u2019" in core:
            result.append(word)
            continue

        # Skip likely proper nouns / character names: if the word is unknown to dictionary
        # AND wordninja can't split it into all-real-word parts, it's probably a name
        core_lower = core.lower()
        has_camel = any(c.isupper() for c in core[1:]) if len(core) > 1 else False
        if not has_camel and len(core) >= 4 and _symspell:
            exact = _symspell.lookup(core_lower, max_edit_distance=0, verbosity=0)
            close = _symspell.lookup(core_lower, max_edit_distance=1, verbosity=0)
            if not exact and not close:
                # Unknown word -- try wordninja to see if it's merged words
                test_parts = _split_words(core_lower) if _word_splitter else [core_lower]
                if len(test_parts) >= 2:
                    # Check if ALL parts are real words -- if so, it's merged, not a name
                    all_real = all(
                        _symspell.lookup(p, max_edit_distance=0, verbosity=0)
                        for p in test_parts if len(p) >= 2
                    )
                    if not all_real:
                        result.append(word)
                        continue
                    # All parts are real words -- fall through to splitting below
                else:
                    # Can't split -- likely a proper noun, keep as-is
                    result.append(word)
                    continue

        # Try splitting with wordninja
        split_parts = _split_alpha_segment(core)

        if len(split_parts) == 1:
            # No split happened
            result.append(word)
        else:
            # Validate split: reject splits that produce mostly single-char fragments
            # (wordninja sometimes over-splits short words)
            single_chars = sum(1 for p in split_parts if len(p) <= 1)
            if single_chars > len(split_parts) * 0.4 and len(core) <= 8:
                result.append(word)
            # Also reject if split produces any 1-2 char fragments for short words
            # (e.g. "kunigami" -> "kuni gam i" has "i" fragment = bad split)
            elif any(len(p) <= 2 for p in split_parts) and len(core) <= 12:
                result.append(word)
            else:
                # Attach leading punct to first part, trailing to last
                split_parts[0] = lead + split_parts[0]
                split_parts[-1] = split_parts[-1] + trail
                result.extend(split_parts)

    text = ' '.join(result)

    # Final step: Rejoin fragmented words AFTER wordninja splitting
    # (must be last so wordninja doesn't re-split our joined words)
    text = _rejoin_fragmented_words(text)

    # Re-apply direct pattern fixes that wordninja may have undone
    text = _fix_common_merged_words(text)

    return text


# =============================================================================
# Speech formatting
# =============================================================================

# Words that should stay uppercase (proper nouns, acronyms, etc.)
_KEEP_UPPER = {
    'I', 'OK', 'TV', 'UK', 'DNA', 'FBI', 'CIA', 'USA', 'CEO',
    'AI', 'ID', 'HP', 'MP', 'XP', 'KO', 'OP', 'NPC', 'RPG',
}

# Words that should stay lowercase even at sentence start in dialogue
_LOWERCASE_WORDS = {
    'a', 'an', 'the', 'and', 'but', 'or', 'nor', 'for', 'yet', 'so',
    'in', 'on', 'at', 'to', 'of', 'by', 'up', 'as', 'if', 'is', 'it',
}


def _to_sentence_case(text):
    """Convert text to natural sentence case.
    Works on ALL CAPS text (lowercases then capitalizes sentence starts)
    AND on already-lowercase text (just capitalizes sentence starts + fixes 'I').
    Preserves acronyms, proper 'I', and numbers."""
    if not text:
        return text

    alpha_chars = [c for c in text if c.isalpha()]
    if not alpha_chars:
        return text

    upper_ratio = sum(1 for c in alpha_chars if c.isupper()) / len(alpha_chars)
    # For mostly uppercase: lowercase everything then re-case
    # For mostly lowercase: just fix sentence starts and 'I'
    needs_lowering = upper_ratio > 0.6

    words = text.split()
    result = []
    sentence_start = True

    for word in words:
        # Strip trailing punctuation for processing
        trail = ''
        core = word
        while core and core[-1] in '.,!?;:\u2026':
            trail = core[-1] + trail
            core = core[:-1]

        # Preserve numbers and number-letter combos (1ST, 2ND, 300)
        if core and (core.isdigit() or re.match(r'^\d+\w*$', core)):
            result.append(word)
            if trail and trail[-1] in '.!?':
                sentence_start = True
            else:
                sentence_start = False
            continue

        # Preserve known acronyms/abbreviations
        if core.upper() in _KEEP_UPPER:
            result.append(core.upper() + trail)
            sentence_start = trail and trail[-1] in '.!?'
            continue

        # Convert to lowercase if text was mostly uppercase,
        # OR if this individual word is all-caps and not a known acronym (e.g. "AS" -> "as")
        if needs_lowering:
            lower = core.lower()
        elif core.isupper() and len(core) >= 2 and core.upper() not in _KEEP_UPPER:
            lower = core.lower()
        else:
            lower = core

        # Capitalize first word of sentence
        if sentence_start and lower:
            lower = lower[0].upper() + lower[1:]

        # Always capitalize standalone "I"
        if lower == 'i':
            lower = 'I'
        # Fix "i'm", "i'll", "i've", "i'd"
        if lower.startswith("i'") or lower.startswith("i\u2019"):
            lower = 'I' + lower[1:]

        result.append(lower + trail)

        if trail and trail[-1] in '.!?':
            sentence_start = True
        else:
            sentence_start = False

    return ' '.join(result)


def format_text_for_speech(text):
    """Convert OCR output into natural spoken dialogue.
    Runs after cleanup_text() and apply_manga_corrections().
    Does not modify bubble coordinates.

    Goals:
    - Smooth, natural sentence flow for TTS
    - Proper pauses via punctuation (commas, periods, ellipses)
    - Clean up OCR artifacts without losing meaning
    - Preserve emotional markers (!, ?, ...) for TTS expression"""
    if not text:
        return ''

    # 0a. Strip leading stray punctuation (not ellipsis)
    text = re.sub(r'^[.,;:\-]+\s*', '', text)

    # 0. Strip leading garbage fragments -- short nonsense tokens before real text
    # e.g. "E hi ca and from there" -> "and from there"
    # Strategy: scan leading words. If we hit a conjunction/preposition that starts
    # a natural phrase and the words before it are short fragments, strip them.
    _REAL_SHORT = {'a', 'i', 'an', 'am', 'as', 'at', 'be', 'by', 'do', 'go',
                   'he', 'if', 'in', 'is', 'it', 'me', 'my', 'no', 'of', 'oh',
                   'ok', 'on', 'or', 'so', 'to', 'up', 'us', 'we',
                   'and', 'are', 'but', 'can', 'did', 'for', 'get', 'got',
                   'had', 'has', 'her', 'him', 'his', 'how', 'its', 'let',
                   'may', 'new', 'nor', 'not', 'now', 'old', 'one', 'our',
                   'out', 'own', 'ran', 'run', 'say', 'she', 'the', 'too',
                   'two', 'was', 'way', 'who', 'why', 'yes', 'yet', 'you',
                   'all', 'any', 'ask', 'big', 'end', 'far', 'few', 'saw',
                   'hey', 'huh', 'wow', 'hah', 'nah', 'yep', 'ugh', 'hmm'}
    # Words that commonly start real sentences after garbage
    _RESTART_WORDS = {'and', 'but', 'so', 'from', 'then', 'after', 'before',
                      'when', 'while', 'although', 'because', 'since', 'if',
                      'the', 'that', 'this', 'there', 'however', 'eventually',
                      'suddenly', 'finally', 'meanwhile', 'instead', 'anyway'}
    words = text.split()
    garbage_end = 0

    # Method 1: Pure garbage tokens (non-real short words)
    for wi, w in enumerate(words):
        core = re.sub(r'[^a-zA-Z]', '', w).lower()
        if not core:
            garbage_end = wi + 1
            continue
        # Contractions (I'll, I've, I'm, don't, etc.) are NEVER garbage
        if "'" in w:
            break
        # Words with ! or ? are dialogue exclamations -- NEVER garbage
        if '!' in w or '?' in w:
            break
        if len(core) <= 3 and core not in _REAL_SHORT and not core.isdigit():
            garbage_end = wi + 1
        else:
            break

    # Method 2: Short fragment + restart word pattern
    # "He cr and from there" -- "He" is real but "He cr" before "and" is garbage
    # "e her and from there" -- "e" stripped by Method 1, but "her" remains before "and"
    # Run Method 2 even if Method 1 found some garbage -- it might find MORE
    if len(words) >= 3:
        for wi in range(1, min(5, len(words))):
            core = re.sub(r'[^a-zA-Z]', '', words[wi]).lower()
            if core in _RESTART_WORDS:
                # Check if everything before is short fragments (all words <= 3 chars)
                prefix = words[:wi]
                all_short = all(len(re.sub(r'[^a-zA-Z]', '', w)) <= 3 for w in prefix)
                # Also check: at least one word before the restart is not a real word
                # Contractions (I'll, I've, etc.) are never garbage
                has_garbage = any(
                    len(re.sub(r'[^a-zA-Z]', '', w)) <= 3
                    and re.sub(r'[^a-zA-Z]', '', w).lower() not in _REAL_SHORT
                    and "'" not in w
                    for w in prefix
                )
                if all_short and has_garbage:
                    garbage_end = wi
                    break

    if 1 <= garbage_end <= 4 and garbage_end < len(words):
        stripped = ' '.join(words[:garbage_end])
        log.debug(f'  Stripped leading garbage: "{stripped}"')
        text = ' '.join(words[garbage_end:])

    # 1. Join broken lines into continuous sentences
    text = re.sub(r'\n+', ' ', text)

    # 1b. Rejoin hyphenated words: "COM-PETITION" -> "COMPETITION", "WONDER-FUL" -> "WONDERFUL"
    # Use wordninja to check if the joined form makes a real word
    def rejoin_hyphen(m):
        left, right = m.group(1), m.group(2)
        joined = left + right
        # Always join if either side is short (fragment)
        if len(left) < 4 or len(right) < 4:
            return joined
        # Check if joined form is a single recognized word
        if _word_splitter:
            parts = _split_words(joined)
            if len(parts) == 1:
                return joined  # "ASSASSINATION" = 1 word, join
        # Otherwise keep as separate words
        return left + ' ' + right
    text = re.sub(r'(\w+)\s*-\s*(\w+)', rejoin_hyphen, text)

    # 2. Join spaced single letters: "H E L L O" -> "HELLO"
    #    Only merge sequences of 3+ single letters to avoid merging real words like "I I" or "I A"
    def merge_spaced(m):
        return m.group(0).replace(' ', '')
    text = re.sub(r'(?<!\w)([A-Za-z]) ([A-Za-z])(?: ([A-Za-z]))+(?!\w)', merge_spaced, text)

    # 2b. Join single CONSONANT + short fragment: "H mph" -> "Hmph", "N no" -> "Nno"
    #     Skip vowels (A, I, O) and common single-letter words to avoid "a week" -> "aweek"
    #     Must NOT follow an apostrophe (avoid "didn't take" -> "didn'ttake")
    text = re.sub(r"(?<!['\u2019])\b([BCDFGHJKLMNPQRSTVWXYZbcdfghjklmnpqrstvwxyz]) ([a-z]{2,4})\b", r'\1\2', text)

    # 3. Fix slash-as-L errors (skip numeric fractions like 1/2)
    text = re.sub(r'([a-zA-Z])/([a-zA-Z])', r'\1l\2', text)
    text = re.sub(r'([a-zA-Z])/$', r'\1l', text)
    text = re.sub(r'([a-zA-Z])led\b', r'\1lled', text)

    # 4. Normalize repeated/conflicting punctuation
    text = re.sub(r'!{3,}', '!!', text)   # Keep double for emphasis, reduce triple+
    text = re.sub(r'\?{3,}', '??', text)
    text = re.sub(r'\.{4,}', '...', text)
    text = re.sub(r'[?!]{4,}', '?!', text)
    # Fix double dots -> ellipsis: "Mars.." -> "Mars..."
    text = re.sub(r'\.\.(?!\.)', '...', text)
    # Fix conflicting punctuation: ".?" -> "?", ".!" -> "!", ",." -> ".", ",," -> ","
    text = re.sub(r'[.,]+([?!])', r'\1', text)
    text = re.sub(r',\.', '.', text)
    text = re.sub(r'\.,', '.', text)
    text = re.sub(r',+', ',', text)
    # Trailing comma before end -> period
    text = re.sub(r',\s*$', '.', text)

    # 4c. Remove false periods mid-sentence: "always. watching" -> "always watching"
    # If a SINGLE period (not part of ..) is followed by a lowercase word, it's an OCR artifact
    text = re.sub(r'(?<!\.)\.(?!\.)\s+([a-z])', r' \1', text)

    # 5. Fix space before punctuation: "Hello ." -> "Hello.", "What ?" -> "What?"
    text = re.sub(r'\s+([.!?,;:\u2026])', r'\1', text)

    # 5b. Ensure space AFTER punctuation when followed by a word
    # "Hello.World" -> "Hello. World", "What?No" -> "What? No"
    text = re.sub(r'([.!?])([A-Za-z])', r'\1 \2', text)
    text = re.sub(r',([A-Za-z])', r', \1', text)

    # 6. Remove noise tokens (pure symbols, not meaningful punctuation)
    text = re.sub(r'(?<!\w)[=|_~#*]{2,}(?!\w)', '', text)
    text = re.sub(r'(?:^|\s)[^\w\s.!?,;:\u2026\'"]{3,}(?:\s|$)', ' ', text)

    # 6b. Remove orphaned punctuation (unmatched brackets, stray symbols)
    text = re.sub(r'(?<!\w)[)\]}]', '', text)  # closing without opener
    text = re.sub(r'[(\[{](?!\w)', '', text)    # opening without closer

    # 7. Convert ALL CAPS to natural sentence case
    text = _to_sentence_case(text)

    # 8. Remove duplicated adjacent words (after case conversion so "I I" works)
    text = re.sub(r'\b(\w+)(\s+\1)+\b', r'\1', text, flags=re.IGNORECASE)

    # 9. Ellipsis normalization for natural TTS pauses
    # Convert "..." to proper ellipsis character (TTS engines handle it better)
    text = text.replace('...', '\u2026')
    # Clean up ellipsis + stray punctuation: "...," -> "...", ",...," -> "..."
    text = re.sub(r'\u2026[.,]+', '\u2026', text)
    text = re.sub(r'[.,]+\u2026', '\u2026', text)

    # 10. Ensure sentence-ending punctuation for natural TTS pause
    text = text.strip()
    if text and text[-1] not in '.!?\u2026':
        text += '.'

    # 11. Ensure sentences have proper spacing for TTS breathing
    # Add comma before common conjunctions if there's no punctuation (helps TTS pacing)
    # But NOT after "even" (even though), "as" (as though), etc.
    text = re.sub(r'(?<!even )(?<!as )(\w{4,}) (but|yet|however|although) ', r'\1, \2 ',
                  text, flags=re.IGNORECASE)

    # Final whitespace cleanup
    text = re.sub(r'\s+', ' ', text).strip()

    # 12. SymSpell spell check -- fix remaining OCR typos
    # Only correct words that are clearly wrong (edit distance 1-2)
    # Skip short words, proper nouns, and words already in dictionary
    if _symspell and text:
        words = text.split()
        corrected = []
        for w in words:
            # Strip trailing punctuation for lookup
            trail = ''
            core = w
            while core and core[-1] in '.,!?;:\u2026\'"':
                trail = core[-1] + trail
                core = core[:-1]
            lead = ''
            while core and core[0] in '\'"':
                lead += core[0]
                core = core[1:]

            # Skip: empty, short words, numbers, contractions, capitalized words (likely names/places)
            if (not core or len(core) <= 2 or not core.isalpha()
                    or "'" in w or "\u2019" in w
                    or (core[0].isupper() and not core.isupper())):
                corrected.append(w)
                continue

            # Skip protected vocabulary (character names, honorifics, etc.)
            if core.lower() in _protected_vocab:
                corrected.append(w)
                continue

            # Check if word is in dictionary
            suggestions = _symspell.lookup(core.lower(), max_edit_distance=1, verbosity=0)
            if suggestions and suggestions[0].distance > 0:
                fix = suggestions[0].term
                # Preserve original capitalization
                if core[0].isupper():
                    fix = fix[0].upper() + fix[1:]
                if core.isupper():
                    fix = fix.upper()
                corrected.append(lead + fix + trail)
            else:
                corrected.append(w)
        text = ' '.join(corrected)

    return text
