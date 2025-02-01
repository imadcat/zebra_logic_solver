import re
import streamlit as st
import pandas as pd
from ortools.sat.python import cp_model

# Try loading spaCy with the transformer-based model.
try:
    import spacy
    nlp = spacy.load("en_core_web_trf")
except Exception as e:
    st.warning("spaCy model could not be loaded; proceeding without it: " + str(e))
    nlp = None

# Mapping for ordinal words (assumes up to six houses).
ordinal_map = {
    "first": 1,
    "second": 2,
    "third": 3,
    "fourth": 4,
    "fifth": 5,
    "sixth": 6
}

# Mapping for number words used in between–clues.
word_to_num = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6
}

def sanitize_token(text):
    """
    Lowercase the text, remove common noise phrases,
    extra punctuation, and non-informative words like 'owner', 'lover', 'enthusiast'.
    """
    text = text.lower()
    noise_phrases = [
        r'\bthe person\b',
        r'\bperson\b',
        r'\bsmoker\b',
        r'\buses?\b',
        r'\bloves?\b',
        r'\bpartial to\b',
        r'\bbouquet of\b',
        r'\bboquet of\b',
        r'\bvase of\b',
        r'\bmany unique\b',
        r'\bbouquet\b',
        r'\bvase\b',
        r'\barrangement\b'
    ]
    for phrase in noise_phrases:
        text = re.sub(phrase, '', text, flags=re.IGNORECASE)
    text = re.sub(r'\b(a|an|the)\b', '', text)
    # Remove extra non-informative words.
    text = re.sub(r'\b(owner|lover|enthusiast)\b', '', text)
    text = re.sub(r'[^a-z0-9 ]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def normalize_token(token, candidate_key=None):
    """
    Normalize tokens when candidate_key indicates a special category.
    For "month": replace full month names with their abbreviated versions.
    For "nationalities": convert adjectives like "swedish" to "swede",
    "british" to "brit", and "danish" to "dane".
    """
    token_norm = token.lower()
    if candidate_key == "month":
        month_map = {
            "january": "jan",
            "february": "feb",
            "march": "mar",
            "april": "april",
            "may": "may",
            "june": "jun",
            "july": "jul",
            "august": "aug",
            "september": "sept",
            "october": "oct",
            "november": "nov",
            "december": "dec",
        }
        for full, abbr in month_map.items():
            if full in token_norm:
                token_norm = token_norm.replace(full, abbr)
    elif candidate_key == "nationalities":
        nat_map = {
            "swedish": "swede",
            "british": "brit",
            "danish": "dane"
        }
        for full, abbr in nat_map.items():
            if full in token_norm:
                token_norm = token_norm.replace(full, abbr)
    return token_norm

def lemmatize_text(text):
    """
    If spaCy is available, return a lemmatized version of the text.
    Otherwise, return the text unmodified.
    """
    if nlp is not None:
        doc = nlp(text)
        return " ".join([token.lemma_ for token in doc])
    return text

def get_category_key(category):
    """
    Determines a unique key for a category based on its description.
    For example, if the description mentions "name", "flower", "hobby", etc.,
    then the assigned keys will be "name", "flower", "hobby", etc.
    For "birthday" or "month", the key will be "month", and for "nationalities", the key is "nationalities".
    For pet/animal categories, if the description mentions either term, we return "animals".
    Otherwise, we default to the last word.
    """
    cat_lower = category.lower()
    if "favorite" in cat_lower and "color" in cat_lower:
        return "favorite_color"
    if "hair" in cat_lower:
        return "hair_color"
    if "name" in cat_lower:
        return "name"
    if "vacation" in cat_lower:
        return "vacation"
    if "occupation" in cat_lower:
        return "occupation"
    if "flower" in cat_lower:
        return "flower"
    if "lunch" in cat_lower:
        return "lunch"
    if "smoothie" in cat_lower:
        return "smoothie"
    if "hobby" in cat_lower:
        return "hobby"
    if "pet" in cat_lower or "animal" in cat_lower:
        return "animals"
    if "birthday" in cat_lower or "month" in cat_lower:
        return "month"
    if "nationalities" in cat_lower:
        return "nationalities"
    tokens = cat_lower.split()
    return tokens[-1] if tokens else cat_lower

def shorten_category(category):
    """
    Returns a short, distinct header for the category in the output table.
    """
    key = get_category_key(category)
    return key.replace('_', ' ')

class PuzzleSolver:
    def __init__(self, puzzle_text, debug=False):
        self.puzzle_text = puzzle_text
        self.num_houses = None
        # Categories: key = full bullet‐line text, value = list of attributes.
        self.categories = {}
        # Mapping from the full category description to a computed unique key.
        self.category_keys = {}
        self.clues = []  # List of clue strings.
        # CP variables: self.var[category][attribute] holds the house number.
        self.var = {}
        self.model = cp_model.CpModel()
        self.debug = debug

        # Extend the keywords with entries for various categories.
        # Both "pet" and "animals" keywords are provided.
        self.category_keywords = {
            "nationalities": ["swede", "norwegian", "german", "chinese", "dane", "brit", "danish", "swedish", "british"],
            "name": ["name"],
            "vacation": ["vacation", "trip", "break"],
            "occupation": ["occupation", "job"],
            "lunch": ["lunch", "soup", "stew", "grilled", "cheese", "spaghetti", "pizza", "stir"],
            "smoothie": ["smoothie", "cherry", "dragonfruit", "watermelon", "lime", "blueberry", "desert"],
            "models": ["phone", "model", "iphone", "pixel", "oneplus", "samsung", "xiaomi", "huawei"],
            "hair_color": ["hair"],
            "month": ["month", "birthday", "birth"],
            "hobby": ["photography", "cooking", "knitting", "woodworking", "paints", "painting", "gardening"],
            "pet": ["rabbit", "hamster", "fish", "cat", "bird", "dog"],
            "animals": ["rabbit", "dog", "horse", "fish", "bird", "cat"]
        }

    def parse_puzzle(self):
        # Determine number of houses.
        m = re.search(r"There are (\d+) houses", self.puzzle_text, re.IGNORECASE)
        self.num_houses = int(m.group(1)) if m else 6

        # Parse categories (expects bullet lines in the puzzle description)
        cat_pattern = re.compile(r"^[-*]\s*(.*?):\s*(.+)$")
        for line in self.puzzle_text.splitlines():
            line = line.strip()
            m = cat_pattern.match(line)
            if m:
                cat_label = m.group(1).strip()
                attr_line = m.group(2).strip()
                attrs = [x.strip() for x in attr_line.split(",") if x.strip()]
                self.categories[cat_label] = attrs
                self.category_keys[cat_label] = get_category_key(cat_label)
                if self.debug:
                    print(f"Parsed category: '{cat_label}' with attributes {attrs}")
                    print(f"Assigned key for category: {self.category_keys[cat_label]}")

        # Parse clues (all nonempty lines after "### Clues:")
        clues_section = False
        for line in self.puzzle_text.splitlines():
            if "### Clues:" in line:
                clues_section = True
                continue
            if clues_section:
                clean = line.strip()
                if clean:
                    self.clues.append(clean)
                    if self.debug:
                        print(f"Parsed clue: {clean}")

    def build_variables(self):
        for cat, attrs in self.categories.items():
            self.var[cat] = {}
            for attr in attrs:
                self.var[cat][attr] = self.model.NewIntVar(1, self.num_houses, f"{cat}_{attr}")
            self.model.AddAllDifferent(list(self.var[cat].values()))
            if self.debug:
                print(f"Added all-different constraint for category '{cat}'.")

    def find_attribute(self, token):
        """
        Try to match an attribute within the token.
        Uses the configured keywords and a longest-match heuristic.
        Also uses lemmatization, and—for hobby clues—maps any variant of 'paint' to 'painting'.
        """
        token_san = sanitize_token(token)
        candidate_key = None
        for key, kws in self.category_keywords.items():
            if any(kw in token_san for kw in kws):
                candidate_key = key
                if self.debug:
                    print(f"Debug: Token '{token}' suggests category key '{candidate_key}' based on keywords {kws}.")
                break

        # Unify candidate keys for animal-related clues:
        if candidate_key == "pet":
            candidate_key = "animals"

        token_lemmatized = lemmatize_text(token_san)
        if self.debug:
            print(f"Debug: Lemmatized token for '{token}': '{token_lemmatized}'")

        if candidate_key == "hobby":
            if "paint" in token_lemmatized:
                token_lemmatized = token_lemmatized.replace("paint", "painting")
                if self.debug:
                    print(f"Debug: Adjusted hobby token to '{token_lemmatized}' for proper matching.")

        if candidate_key in ["month", "nationalities"]:
            token_san = normalize_token(token_san, candidate_key)
            if self.debug:
                print(f"Debug: Normalized token for {candidate_key}: '{token_san}'")

        if candidate_key:
            categories_to_search = [(cat, attrs) for cat, attrs in self.categories.items()
                                      if self.category_keys.get(cat) == candidate_key]
            if self.debug:
                cats = [cat for cat, _ in categories_to_search]
                print(f"Debug: Restricted search to categories: {cats}")
        else:
            categories_to_search = self.categories.items()

        best = None
        best_len = 0
        for cat, attrs in categories_to_search:
            for attr in attrs:
                attr_san = sanitize_token(attr)
                if candidate_key in ["month", "nationalities"]:
                    attr_san = normalize_token(attr_san, candidate_key)
                pattern = rf'\b{re.escape(attr_san)}\b'
                if re.search(pattern, token_san) or re.search(pattern, token_lemmatized):
                    if len(attr_san) > best_len:
                        best = (cat, attr)
                        best_len = len(attr_san)
                else:
                    alt = attr_san[:-1] if attr_san.endswith('s') else attr_san + 's'
                    if re.search(rf'\b{re.escape(alt)}\b', token_san) or re.search(rf'\b{re.escape(alt)}\b', token_lemmatized):
                        if len(attr_san) > best_len:
                            best = (cat, attr)
                            best_len = len(attr_san)
        if best is None and candidate_key in ["month", "nationalities"]:
            if self.debug:
                print(f"Debug: Fallback for {candidate_key}: no match found in token '{token_san}'. Trying explicit substrings.")
            mapping = {}
            if candidate_key == "month":
                mapping = {"jan": "jan", "feb": "feb", "mar": "mar",
                           "april": "april", "may": "may", "jun": "jun",
                           "jul": "jul", "aug": "aug", "sept": "sept",
                           "oct": "oct", "nov": "nov", "dec": "dec"}
            elif candidate_key == "nationalities":
                mapping = {"swede": "swede", "norwegian": "norwegian", "german": "german",
                           "chinese": "chinese", "dane": "dane", "brit": "brit"}
            for key_abbr in mapping.values():
                if re.search(rf'\b{re.escape(key_abbr)}\b', token_san):
                    for cat, attrs in categories_to_search:
                        for attr in attrs:
                            attr_san = normalize_token(sanitize_token(attr), candidate_key)
                            if attr_san == key_abbr:
                                best = (cat, attr)
                                best_len = len(attr_san)
                                if self.debug:
                                    print(f"Debug: Found fallback match: '{attr_san}' in token '{token_san}'.")
                                break
                        if best is not None:
                            break
                if best is not None:
                    break
        if best is None and self.debug:
            print(f"DEBUG: No attribute found for token '{token}' (sanitized: '{token_san}', lemmatized: '{token_lemmatized}').")
        return best

    def find_all_attributes_in_text(self, text):
        found = []
        text_san = sanitize_token(text)
        for cat, attrs in self.categories.items():
            for attr in attrs:
                attr_san = sanitize_token(attr)
                if re.search(rf'\b{re.escape(attr_san)}\b', text_san):
                    found.append((cat, attr))
        unique = []
        seen = set()
        for pair in found:
            if pair not in seen:
                unique.append(pair)
                seen.add(pair)
        return unique

    def spacy_equality_extraction(self, text):
        """
        Use spaCy's dependency parse (and fallback to NER) to extract two attribute phrases.
        """
        if nlp is None:
            return None, None
        doc = nlp(text)
        for token in doc:
            if token.lemma_ == "be" and token.dep_ == "ROOT":
                subj = None
                attr = None
                for child in token.children:
                    if child.dep_ in ["nsubj", "nsubjpass"]:
                        subj = child
                    if child.dep_ in ["attr", "acomp"]:
                        attr = child
                if subj and attr:
                    subject_span = doc[subj.left_edge.i: subj.right_edge.i+1].text
                    attr_span = doc[attr.left_edge.i: attr.right_edge.i+1].text
                    return subject_span, attr_span
        ents = list(doc.ents)
        if len(ents) >= 2:
            return ents[0].text, ents[1].text
        return None, None

    def apply_constraint_equality(self, token1, token2):
        a1 = self.find_attribute(token1)
        a2 = self.find_attribute(token2)
        if a1 and a2:
            cat1, attr1 = a1
            cat2, attr2 = a2
            self.model.Add(self.var[cat1][attr1] == self.var[cat2][attr2])
            if self.debug:
                print(f"Added constraint: [{cat1}][{attr1}] == [{cat2}][{attr2}]")
        else:
            if self.debug:
                print(f"Warning: could not apply equality between '{token1}' and '{token2}'")

    def apply_constraint_inequality(self, token, house_number):
        a1 = self.find_attribute(token)
        if a1:
            cat, attr = a1
            self.model.Add(self.var[cat][attr] != house_number)
            if self.debug:
                print(f"Added constraint: [{cat}][{attr}] != {house_number}")
        else:
            if self.debug:
                print(f"Warning: could not apply inequality for '{token}' at house {house_number}")

    def apply_constraint_position(self, token1, op, token2):
        a1 = self.find_attribute(token1)
        a2 = self.find_attribute(token2)
        if a1 and a2:
            cat1, attr1 = a1
            cat2, attr2 = a2
            if op == "==":
                self.model.Add(self.var[cat1][attr1] == self.var[cat2][attr2])
                if self.debug:
                    print(f"Added constraint: [{cat1}][{attr1}] == [{cat2}][{attr2}]")
            elif op == "<":
                self.model.Add(self.var[cat1][attr1] < self.var[cat2][attr2])
                if self.debug:
                    print(f"Added constraint: [{cat1}][{attr1}] < [{cat2}][{attr2}]")
            elif op == ">":
                self.model.Add(self.var[cat1][attr1] > self.var[cat2][attr2])
                if self.debug:
                    print(f"Added constraint: [{cat1}][{attr1}] > [{cat2}][{attr2}]")
            elif op == "+1":
                self.model.Add(self.var[cat1][attr1] + 1 == self.var[cat2][attr2])
                if self.debug:
                    print(f"Added constraint: [{cat1}][{attr1}] + 1 == [{cat2}][{attr2}]")
            elif op == "-1":
                self.model.Add(self.var[cat1][attr1] - 1 == self.var[cat2][attr2])
                if self.debug:
                    print(f"Added constraint: [{cat1}][{attr1}] - 1 == [{cat2}][{attr2}]")
        else:
            if self.debug:
                print(f"Warning: could not apply position constraint between '{token1}' and '{token2}' with op '{op}'")

    def apply_constraint_next_to(self, token1, token2):
        a1 = self.find_attribute(token1)
        a2 = self.find_attribute(token2)
        if a1 and a2:
            cat1, attr1 = a1
            cat2, attr2 = a2
            diff = self.model.NewIntVar(0, self.num_houses, f"diff_{attr1}_{attr2}")
            self.model.AddAbsEquality(diff, self.var[cat1][attr1] - self.var[cat2][attr2])
            self.model.Add(diff == 1)
            if self.debug:
                print(f"Added next-to constraint: |[{cat1}][{attr1}] - [{cat2}][{attr2}]| == 1")
        else:
            if self.debug:
                print(f"Warning: could not apply next-to constraint between '{token1}' and '{token2}'")

    def apply_constraint_between(self, token1, token2, houses_between):
        a1 = self.find_attribute(token1)
        a2 = self.find_attribute(token2)
        if a1 and a2:
            cat1, attr1 = a1
            cat2, attr2 = a2
            diff = self.model.NewIntVar(0, self.num_houses, f"between_{attr1}_{attr2}")
            self.model.AddAbsEquality(diff, self.var[cat1][attr1] - self.var[cat2][attr2])
            self.model.Add(diff == houses_between + 1)
            if self.debug:
                print(f"Added between constraint: |[{cat1}][{attr1}] - [{cat2}][{attr2}]| == {houses_between + 1}")
        else:
            if self.debug:
                print(f"Warning: could not apply between constraint for '{token1}' and '{token2}' with {houses_between} houses in between")

    def apply_constraint_fixed(self, token, house_number):
        a1 = self.find_attribute(token)
        if a1:
            cat, attr = a1
            self.model.Add(self.var[cat][attr] == house_number)
            if self.debug:
                print(f"Added fixed constraint: [{cat}][{attr}] == {house_number}")
        else:
            if self.debug:
                print(f"Warning: could not apply fixed constraint for '{token}' at house {house_number}")

    def process_clue(self, clue):
        text = re.sub(r'^\d+\.\s*', '', clue).strip()
        if self.debug:
            print(f"Processing clue: {text}")
        ordinal_numbers = r"(?:\d+|first|second|third|fourth|fifth|sixth)"

        m_fixed = re.search(rf"(.+?) is in the ({ordinal_numbers}) house", text, re.IGNORECASE)
        if m_fixed:
            token = m_fixed.group(1).strip()
            num_str = m_fixed.group(2).strip().lower()
            house_num = int(num_str) if num_str.isdigit() else ordinal_map.get(num_str)
            if house_num is not None:
                self.apply_constraint_fixed(token, house_num)
                return

        m_not = re.search(rf"(.+?) is not in the ({ordinal_numbers}) house", text, re.IGNORECASE)
        if m_not:
            token = m_not.group(1).strip()
            num_str = m_not.group(2).strip().lower()
            house_num = int(num_str) if num_str.isdigit() else ordinal_map.get(num_str)
            if house_num is not None:
                self.apply_constraint_inequality(token, house_num)
                return

        m_left = re.search(r"(.+?) is directly left of (.+)", text, re.IGNORECASE)
        if m_left:
            token1 = m_left.group(1).strip()
            token2 = m_left.group(2).strip()
            self.apply_constraint_position(token1, "+1", token2)
            return

        m_right = re.search(r"(.+?) is directly right of (.+)", text, re.IGNORECASE)
        if m_right:
            token1 = m_right.group(1).strip()
            token2 = m_right.group(2).strip()
            self.apply_constraint_position(token1, "-1", token2)
            return

        m_sl = re.search(r"(.+?) is somewhere to the left of (.+)", text, re.IGNORECASE)
        if m_sl:
            token1 = m_sl.group(1).strip()
            token2 = m_sl.group(2).strip()
            self.apply_constraint_position(token1, "<", token2)
            return

        m_sr = re.search(r"(.+?) is somewhere to the right of (.+)", text, re.IGNORECASE)
        if m_sr:
            token1 = m_sr.group(1).strip()
            token2 = m_sr.group(2).strip()
            self.apply_constraint_position(token1, ">", token2)
            return

        m_next = re.search(r"(.+?) and (.+?) are next to each other", text, re.IGNORECASE)
        if m_next:
            token1 = m_next.group(1).strip()
            token2 = m_next.group(2).strip()
            self.apply_constraint_next_to(token1, token2)
            return

        m_between = re.search(rf"There (?:are|is) (\d+|one|two|three|four|five|six) house(?:s)? between (.+?) and (.+)", text, re.IGNORECASE)
        if m_between:
            num_str = m_between.group(1).strip().lower()
            houses_between = int(num_str) if num_str.isdigit() else word_to_num.get(num_str)
            token1 = m_between.group(2).strip()
            token2 = m_between.group(3).strip()
            self.apply_constraint_between(token1, token2, houses_between)
            return

        m_eq = re.search(r"(.+)\sis(?: the)?\s(.+)", text, re.IGNORECASE)
        if m_eq:
            token1 = m_eq.group(1).strip()
            token2 = m_eq.group(2).strip()
            token1 = re.sub(r"^(the person who\s+|who\s+)", "", token1, flags=re.IGNORECASE).strip()
            token2 = re.sub(r"^(a\s+|an\s+|the\s+)", "", token2, flags=re.IGNORECASE).strip()
            a1 = self.find_attribute(token1)
            a2 = self.find_attribute(token2)
            if a1 and a2:
                self.apply_constraint_equality(token1, token2)
                return
            else:
                if self.debug:
                    print("Equality regex failed to extract valid attributes using token cleaning.")
        if nlp is not None:
            left, right = self.spacy_equality_extraction(text)
            if left and right:
                if self.debug:
                    print(f"spaCy extracted equality: '{left}' == '{right}'")
                self.apply_constraint_equality(left, right)
                return

        if self.debug:
            print(f"Unprocessed clue: {text}")

    def process_all_clues(self):
        for clue in self.clues:
            self.process_clue(clue)

    def solve(self):
        solver = cp_model.CpSolver()
        status = solver.Solve(self.model)
        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            solution = {}
            for house in range(1, self.num_houses + 1):
                solution[house] = {}
                for cat, attr_dict in self.var.items():
                    for attr, var in attr_dict.items():
                        if solver.Value(var) == house:
                            solution[house][cat] = attr
            return solution
        else:
            if self.debug:
                print("No solution found. The clues may be contradictory or incomplete.")
            return None

    def print_solution(self, solution):
        # Create a table as a list of lists.
        if solution:
            headers = ["House"] + [shorten_category(cat) for cat in self.categories.keys()]
            table = []
            for house in sorted(solution.keys()):
                row = [str(house)]
                for cat in self.categories.keys():
                    row.append(solution[house].get(cat, ""))
                table.append(row)
            # Return as a DataFrame.
            df = pd.DataFrame(table, columns=headers)
            return df
        else:
            return None

# Streamlit App
st.title("Generic Logic Puzzle Solver")

st.markdown("""
Enter your puzzle description in the text area below.
The puzzle should include a list of category definitions (each on a new bullet line) and a list of clues (after a line containing '### Clues:').
""")

puzzle_text = st.text_area("Puzzle Input", height=300)

show_debug = st.checkbox("Show Debug Output", value=False)

if st.button("Solve Puzzle") and puzzle_text:
    solver_instance = PuzzleSolver(puzzle_text, debug=show_debug)
    solver_instance.parse_puzzle()
    solver_instance.build_variables()
    solver_instance.process_all_clues()

    # Display parsed attributes.
    st.subheader("Parsed Attributes (Categories & Their Attributes)")
    st.write(solver_instance.categories)

    # Display parsed clues.
    st.subheader("Parsed Clues")
    st.write(solver_instance.clues)

    solution = solver_instance.solve()
    st.subheader("Solution Table")
    df_solution = solver_instance.print_solution(solution)
    if df_solution is not None:
        st.table(df_solution)
    else:
        st.error("No solution found. The clues may be contradictory or incomplete.")