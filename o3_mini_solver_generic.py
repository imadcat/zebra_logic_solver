#!/usr/bin/env python3
import re, sys
import spacy
from ortools.sat.python import cp_model

# Load spaCy’s English model.
nlp = spacy.load("en_core_web_trf")

# --- Helper Functions ---

def ordinal_word_to_index(word):
    """Convert an ordinal word (e.g., 'fifth') to a 0-indexed number."""
    ordinal_map = {
        "first": 1, "second": 2, "third": 3, "fourth": 4,
        "fifth": 5, "sixth": 6, "seventh": 7, "eighth": 8,
        "ninth": 9, "tenth": 10
    }
    return ordinal_map.get(word.lower(), None) - 1 if word.lower() in ordinal_map else None

def cardinal_word_to_number(word):
    """Convert a cardinal word (e.g., 'two') to a number."""
    cardinal_map = {
        "one": 1, "two": 2, "three": 3, "four": 4,
        "five": 5, "six": 6, "seven": 7, "eight": 8,
        "nine": 9, "ten": 10
    }
    return cardinal_map.get(word.lower(), None)

def split_puzzle_text(text):
    """
    Splits the full puzzle text into an attributes section and a clues section.
    It searches for a line containing “clues:” (case‑insensitive).
    """
    # Create a pattern that matches either of the two sentences.
    pattern = r"(Each house has a unique attribute for each of the following characteristics:)|(### Clues:)"
    parts = re.split(pattern, text)
    if len(parts) >= 2:
        attr_text = parts[3]
        clues_text = parts[6]
    else:
        exit("Error: could not find 'Each house has a unique attribute' or '### Clues:' in the text.")
    return attr_text, clues_text

def infer_attribute_key(text):
    """
    Infer an attribute name from the descriptive text (before the colon).
    Uses spaCy's noun_chunks to try to pick the most important noun in the sentence.
    If no noun_chunk is found, falls back to a cleaned, lower-cased version of the text.
    """
    doc = nlp(text)
    # Get a list of noun chunks.
    chunks = list(doc.noun_chunks)
    if chunks:
        # Use the root of the last noun chunk as the attribute key.
        attr_key = chunks[-1].root.lemma_.lower()
    else:
        attr_key = re.sub(r'\W+', '', text.lower())
    return attr_key

def extract_attributes(text):
    """
    Extracts attribute data from a multiline string.
    Each line is expected to have the format:
      [descriptive text]: value1, value2, value3, ...
    The attribute key is dynamically inferred from the descriptive text.
    Returns a dictionary where keys are the inferred attribute names and values are lists
    of attribute values.
    """
    attributes = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or ":" not in line:
            continue
        
        # Remove bullet markers if present.
        line = re.sub(r'^[\-*]\s*', '', line)
        key_part, val_part = line.split(":", 1)
        key_norm = infer_attribute_key(key_part)
        
        # Split the values by comma and strip any unnecessary spaces.
        values = [v.strip() for v in val_part.split(",") if v.strip()]
        
        # Optionally, use spaCy to "normalize" the tokens (optional).
        normalized_values = []
        for value in values:
            doc = nlp(value)
            tokens = [token.text for token in doc]
            normalized_value = " ".join(tokens)
            normalized_values.append(normalized_value)
            
        attributes[key_norm] = normalized_values
    return attributes

def get_index(lst, target):
    """Return the index of target in lst (ignoring case and extra whitespace)."""
    for i, val in enumerate(lst):
        if val.strip().lower() == target.strip().lower():
            return i
    raise ValueError(f"Value '{target}' not found in list: {lst}")

def build_normalized_attr(attr_lists):
    """
    Build a dictionary of normalized attribute values using spaCy’s lemmatization.
    Returns a structure: norm_attr[key][normalized_value] = original_value
    """
    norm_attr = {}
    for key, values in attr_lists.items():
        norm_attr[key] = {}
        for val in values:
            doc = nlp(val)
            if len(doc) > 0:
                norm = doc[0].lemma_.lower()
            else:
                norm = val.lower()
            norm_attr[key][norm] = val
    return norm_attr

# --- Remapping Helpers ---

def remap_operand(operand, attr_lists):
    """
    If an operand’s value does not appear in the list for its assigned attribute,
    scan other attribute lists and remap the operand to one that contains the value.
    If the operand's attribute is not found in attr_lists, return None.
    """
    if "attribute" in operand:
        orig_attr = operand["attribute"]
        val = operand["value"]
        if orig_attr not in attr_lists:
            print(f"Warning: attribute '{orig_attr}' not found in attr_lists. Dropping operand: {operand}", file=sys.stderr)
            return None
        try:
            _ = get_index(attr_lists[orig_attr], val)
            return operand
        except ValueError:
            for other_attr, values in attr_lists.items():
                if other_attr == orig_attr:
                    continue
                for v in values:
                    if v.strip().lower() == val.strip().lower():
                        return {"attribute": other_attr, "value": v}
            return None
    return operand

def filter_and_remap_operands(operands, attr_lists):
    """Remap any ambiguous operand and filter out invalid ones."""
    new_ops = []
    for op in operands:
        r = remap_operand(op, attr_lists)
        if r is not None and r not in new_ops:
            new_ops.append(r)
    return new_ops

# --- Custom Mapping for Frequent Phrases ---
def apply_custom_mappings(clue_lower, operands_found):
    """
    For certain common phrases, force-add the corresponding operand.
    The custom_mappings list contains tuples: (phrase, mapping-dict).
    """
    custom_mappings = [
        ("camping trips", {"attribute": "vacation", "value": "camping"}),
        ("mountain retreats", {"attribute": "vacation", "value": "mountain"}),
        ("city breaks", {"attribute": "vacation", "value": "city"}),
        ("craftsman-style", {"attribute": "style", "value": "craftsman"}),
        ("modern-style", {"attribute": "style", "value": "modern"}),
        ("colonial-style", {"attribute": "style", "value": "colonial"}),
        ("mediterranean-style", {"attribute": "style", "value": "mediterranean"}),
        ("victorian", {"attribute": "style", "value": "victorian"}),
        ("auburn hair", {"attribute": "hair", "value": "auburn"}),
        ("black hair", {"attribute": "hair", "value": "black"}),
        ("rose bouquet", {"attribute": "flower", "value": "roses"}),
        ("vase of tulips", {"attribute": "flower", "value": "tulips"}),
        ("carnations arrangement", {"attribute": "flower", "value": "carnations"}),
        ("very short", {"attribute": "height", "value": "very short"}),
        ("average", {"attribute": "height", "value": "average"}),
        ("tall", {"attribute": "height", "value": "tall"}),
        ("super tall", {"attribute": "height", "value": "super tall"}),
    ]
    for phrase, mapping in custom_mappings:
        if phrase in clue_lower and mapping not in operands_found:
            operands_found.append(mapping)
    return operands_found

# --- Clue Parsing Functions ---

def parse_clue(clue_text, attr_lists, norm_attr):
    """
    Parses a natural‑language clue into a dictionary with:
      "operator": one of "==", "!=", "<", ">", "+1==", "abs_eq", or "adjacent"
      "operands": a list of two operands (each is either {"attribute": key, "value": value} or {"constant": number}).
      Optionally, an "offset" is provided for abs_eq.
    It uses regex, spaCy tokenization, noun-chunk extraction, custom mappings, and a fallback split.
    """
    clue_orig = clue_text.strip()
    clue = re.sub(r"^\d+\.\s*", "", clue_orig)
    clue_lower = clue.lower()
    
    op = "=="
    offset = None

    # Determine the operator.
    if "directly left of" in clue_lower:
        op = "+1=="
    elif "directly right of" in clue_lower:
        op = "+1=="
    elif "next to" in clue_lower or "neighbors" in clue_lower or "adjacent" in clue_lower:
        op = "adjacent"
    elif "houses between" in clue_lower:
        op = "abs_eq"
        m = re.search(r"(\w+)\s+houses between", clue_lower)
        if m:
            num = cardinal_word_to_number(m.group(1))
            if num is not None:
                offset = num + 1
    elif "somewhere to the left of" in clue_lower:
        op = "<"
    elif "somewhere to the right of" in clue_lower:
        op = ">"
    elif "is not in" in clue_lower or "is not" in clue_lower:
        op = "!="

    operands_found = []

    # 1. Regex for ordinal expressions (e.g., "fifth house").
    ordinal_matches = re.findall(r"(first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth)\s+house", clue_lower)
    for word in ordinal_matches:
        idx = ordinal_word_to_index(word)
        if idx is not None and {"constant": idx} not in operands_found:
            operands_found.append({"constant": idx})

    # 2. Regex for direct attribute value matches.
    for key, values in attr_lists.items():
        for val in values:
            pattern = r"\b" + re.escape(val.lower().rstrip("s")) + r"(s)?\b"
            if re.search(pattern, clue_lower):
                operand = {"attribute": key, "value": val}
                if operand not in operands_found:
                    operands_found.append(operand)
    
    # 3. spaCy token-by-token analysis.
    doc = nlp(clue)
    for token in doc:
        token_norm = token.lemma_.lower()
        for key, norm_dict in norm_attr.items():
            if token_norm in norm_dict:
                operand = {"attribute": key, "value": norm_dict[token_norm]}
                if operand not in operands_found:
                    operands_found.append(operand)
    
    # 4. spaCy noun-chunk extraction.
    for chunk in doc.noun_chunks:
        chunk_norm = " ".join(tok.lemma_.lower() for tok in chunk if tok.pos_ in {"NOUN", "PROPN", "ADJ"})
        for key, norm_dict in norm_attr.items():
            for norm_val, orig_val in norm_dict.items():
                if norm_val in chunk_norm:
                    operand = {"attribute": key, "value": orig_val}
                    if operand not in operands_found:
                        operands_found.append(operand)
    
    # 5. Apply custom mappings.
    # operands_found = apply_custom_mappings(clue_lower, operands_found)
    
    # 6. Fallback: if fewer than 2 operands and the phrase " is the person who " is present.
    if len(operands_found) < 2 and " is the person who " in clue_lower:
        parts = clue_lower.split(" is the person who ")
        if len(parts) == 2:
            lhs, rhs = parts[0].strip(), parts[1].strip()
            for key, values in attr_lists.items():
                for val in values:
                    if val.lower() in lhs and {"attribute": key, "value": val} not in operands_found:
                        operands_found.append({"attribute": key, "value": val})
                    if val.lower() in rhs and {"attribute": key, "value": val} not in operands_found:
                        operands_found.append({"attribute": key, "value": val})
    
    # 7. Remap and filter invalid operands.
    operands_found = filter_and_remap_operands(operands_found, attr_lists)
    
    if len(operands_found) < 2:
        print(f"Warning: could not parse clue: {clue_orig}", file=sys.stderr)
        return None

    operands = operands_found[:2]
    if "directly right of" in clue_lower:
        operands = [operands[1], operands[0]]
    parsed = {"operator": op, "operands": operands}
    if op == "abs_eq":
        parsed["offset"] = offset if offset is not None else 1
    return parsed

def parse_clues(clues_text, attr_lists, norm_attr):
    def get_constant_index(clue):
        """
        Extracts a constant index from the clue text if present.
        For example, if the clue contains a pattern like 'house 3', it returns 2 (0-indexed).
        Adjust the regex pattern as needed for your puzzle's format.
        """
        pattern = r'house\s+(\d+)'  # Example pattern: "house 3"
        match = re.search(pattern, clue, re.IGNORECASE)
        if match:
            return int(match.group(1)) - 1  # Convert to 0-indexed
        return None

    # If clues_text is a list, join only non-None items into a single string.
    if isinstance(clues_text, list):
        clues_text = "\n".join([str(line) for line in clues_text if line is not None])

    operands_found = []
    for line in clues_text.splitlines():
        line = line.strip()
        if not line:
            continue
        clue_lower = line.lower()
        
        idx = get_constant_index(clue_lower)
        if idx is not None and {"constant": idx} not in operands_found:
            operands_found.append({"constant": idx})
        
        # Regex for direct attribute value matches.
        for key, values in attr_lists.items():
            for val in values:
                pattern = r"\b" + re.escape(val.lower().rstrip("s")) + r"(s)?\b"
                if re.search(pattern, clue_lower):
                    operand = {"attribute": key, "value": val}
                    if operand not in operands_found:
                        operands_found.append(operand)
        
        # spaCy token-by-token analysis.
        doc = nlp(line)
        for token in doc:
            token_norm = token.lemma_.lower()
            for key, norm_dict in norm_attr.items():
                if token_norm in norm_dict:
                    operand = {"attribute": key, "value": norm_dict[token_norm]}
                    if operand not in operands_found:
                        operands_found.append(operand)
        
        # spaCy noun-chunk extraction.
        for chunk in doc.noun_chunks:
            # ...existing code for noun chunk processing...
            pass

    return operands_found

# --- Constraint Adder ---

def add_generic_constraint(model, clue, model_vars, attr_lists):
    """
    Adds a constraint corresponding to a parsed clue.
    Operands may refer to an attribute reference or a constant.
    """
    op = clue["operator"]
    def eval_operand(operand):
        if "attribute" in operand:
            key = operand["attribute"]
            idx = get_index(attr_lists[key], operand["value"])
            return model_vars[key][idx]
        elif "constant" in operand:
            return operand["constant"]
        else:
            raise ValueError("Operand must have either 'attribute' or 'constant'.")
    expr1 = eval_operand(clue["operands"][0])
    expr2 = eval_operand(clue["operands"][1])
    
    if op == "==":
        model.Add(expr1 == expr2)
    elif op == "!=":
        model.Add(expr1 != expr2)
    elif op == "<":
        model.Add(expr1 < expr2)
    elif op == ">":
        model.Add(expr1 > expr2)
    elif op == "+1==":
        model.Add(expr1 + 1 == expr2)
    elif op in ("abs_eq", "adjacent"):
        offset = clue.get("offset", 1)
        diff = model.NewIntVar(0, len(next(iter(attr_lists.values()))), "diff_dummy")
        model.AddAbsEquality(diff, expr1 - expr2)
        model.Add(diff == offset)
    else:
        raise ValueError(f"Unknown operator: {op}")

# --- Main Program ---

def main():
    try:
        with open("full_puzzle.txt", "r") as f:
            full_text = f.read()
    except FileNotFoundError:
        sys.exit("File 'full_puzzle.txt' not found.")
    attr_text, clues_text = split_puzzle_text(full_text)
    attr_lists = extract_attributes(attr_text)
    
    # Optional: To see what keys were extracted, uncomment:
    for key, values in attr_lists.items():
        print(f"{key}: {', '.join(values)}")
    print('\n')
    
    if not attr_lists:
        sys.exit("No attribute lines were found in the puzzle text.")
    n_houses = None
    for key, values in attr_lists.items():
        if n_houses is None:
            n_houses = len(values)
        elif len(values) != n_houses:
            sys.exit(f"Attribute '{key}' has {len(values)} values; expected {n_houses}.")
    
    norm_attr = build_normalized_attr(attr_lists)
    clues = parse_clues(clues_text, attr_lists, norm_attr)
    for clue in clues:
        print(clue)

    if not clues:
        sys.exit("No clues could be parsed from the puzzle text.")
    
    model = cp_model.CpModel()
    model_vars = {}
    for key, values in attr_lists.items():
        model_vars[key] = [model.NewIntVar(0, n_houses - 1, f"{key}_{i}") for i in range(n_houses)]
        model.AddAllDifferent(model_vars[key])
    
    for clue in clues:
        try:
            add_generic_constraint(model, clue, model_vars, attr_lists)
        except Exception as e:
            print(f"Error adding constraint for clue {clue}: {e}", file=sys.stderr)
    
    solver = cp_model.CpSolver()
    status = solver.Solve(model)
    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        solution = { key: [None]*n_houses for key in attr_lists }
        for key, var_list in model_vars.items():
            for i in range(n_houses):
                for j, var in enumerate(var_list):
                    if solver.Value(var) == i:
                        solution[key][i] = attr_lists[key][j]
                        break
        header = "House".ljust(8)
        for key in attr_lists:
            header += key.capitalize().ljust(16)
        print("Solution Found!")
        print(header)
        for i in range(n_houses):
            row = f"House {i+1}".ljust(8)
            for key in attr_lists:
                row += str(solution[key][i]).ljust(16)
            print(row)
    else:
        print("No solution found.")


if __name__ == '__main__':
    main()