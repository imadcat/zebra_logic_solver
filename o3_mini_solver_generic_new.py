import re
import sys
import argparse
from ortools.sat.python import cp_model

# Try loading spaCy.
try:
    import spacy
    nlp = spacy.load("en_core_web_sm")
except Exception as e:
    print("spaCy model could not be loaded; proceeding without it:", e)
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
    Lowercase the text, remove common noise phrases and extra punctuation.
    This version includes additional noise phrases such as "bouquet", "vase", and "arrangement"
    so that phrases like "rose bouquet", "bouquet of daffodils", "vase of tulips", etc.
    are normalized to their underlying names.
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
    text = re.sub(r'[^a-z0-9 ]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def get_category_key(category):
    """
    Determines a unique key for a category based on its description.
    This mapping is configurable—you can adjust or add more rules as needed.
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
    # fallback: default to the last word
    tokens = cat_lower.split()
    return tokens[-1] if tokens else cat_lower

def shorten_category(category):
    """
    Returns a short, distinct header for the category in the output table.
    """
    key = get_category_key(category)
    # Optionally, you can define pretty–printed versions here.
    return key.replace('_', ' ')

class PuzzleSolver:
    def __init__(self, puzzle_text, debug=False):
        self.puzzle_text = puzzle_text
        self.num_houses = None
        # Categories: key = full bullet–line text, value = list of attributes.
        self.categories = {}
        # A mapping from full category description to its computed unique key.
        self.category_keys = {}
        self.clues = []  # List of clue strings.
        # CP variables: self.var[category][attribute] holds the house number for that attribute.
        self.var = {}
        self.model = cp_model.CpModel()
        self.debug = debug

        # A mapping from category keys to an extra set of keywords.
        # These keywords will be used to restrict search in find_attribute.
        self.category_keywords = {
            "favorite_color": ["favorite", "color"],
            "hair_color": ["hair"],
            "name": ["name"],
            "vacation": ["vacation", "trip", "break"],
            "occupation": ["occupation", "job"],
            "flower": ["flower", "roses", "lilies", "tulips", "iris", "daffodils", "carnations"]
        }

    def parse_puzzle(self):
        # Identify number of houses.
        m = re.search(r"There are (\d+) houses", self.puzzle_text, re.IGNORECASE)
        self.num_houses = int(m.group(1)) if m else 6

        # Parse categories.
        # Expected format for each category is a bullet line:
        # "- <Category text>: attribute1, attribute2, attribute3, ..."
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

        # Parse clues: all nonempty lines after a line containing "### Clues:".
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
        # For every category (keyed by the full bullet–line text)
        for cat, attrs in self.categories.items():
            self.var[cat] = {}
            for attr in attrs:
                self.var[cat][attr] = self.model.NewIntVar(1, self.num_houses, f"{cat}_{attr}")
            # All attributes in this category must occupy different houses.
            self.model.AddAllDifferent(list(self.var[cat].values()))
            if self.debug:
                print(f"Added all-different constraint for category '{cat}'.")

    def find_attribute(self, token):
        """
        Find an attribute whose sanitized text appears as a whole word in the token.
        First, determine (if possible) a candidate category key based on the token.
        Then restrict the search to those categories that have that key.
        Uses a longest-match heuristic with a simple singular/plural check.
        Returns (category, attribute) if found.
        """
        token_san = sanitize_token(token)
        candidate_key = None
        # Try to decide which category key we are dealing with by inspecting the token.
        for key, kws in self.category_keywords.items():
            if any(kw in token_san for kw in kws):
                candidate_key = key
                if self.debug:
                    print(f"Debug: Token '{token}' suggests category key '{candidate_key}' based on keywords {kws}.")
                break

        # Restrict search to categories that have the matching key, if determined.
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
                pattern = rf'\b{re.escape(attr_san)}\b'
                if re.search(pattern, token_san):
                    if len(attr_san) > best_len:
                        best = (cat, attr)
                        best_len = len(attr_san)
                else:
                    # Allow a simple singular/plural match.
                    if attr_san.endswith('s'):
                        alt = attr_san[:-1]
                    else:
                        alt = attr_san + 's'
                    if re.search(rf'\b{re.escape(alt)}\b', token_san):
                        if len(attr_san) > best_len:
                            best = (cat, attr)
                            best_len = len(attr_san)
        # Fallback: if nothing was found using the restricted search, look globally.
        if best is None and candidate_key is not None:
            if self.debug:
                print(f"Debug: No attribute found in candidate categories for token '{token}'; falling back to global search.")
            for cat, attrs in self.categories.items():
                for attr in attrs:
                    attr_san = sanitize_token(attr)
                    pattern = rf'\b{re.escape(attr_san)}\b'
                    if re.search(pattern, token_san):
                        if len(attr_san) > best_len:
                            best = (cat, attr)
                            best_len = len(attr_san)
                    else:
                        if attr_san.endswith('s'):
                            alt = attr_san[:-1]
                        else:
                            alt = attr_san + 's'
                        if re.search(rf'\b{re.escape(alt)}\b', token_san):
                            if len(attr_san) > best_len:
                                best = (cat, attr)
                                best_len = len(attr_san)
        if best is None and self.debug:
            print(f"DEBUG: No attribute found for token '{token}' (sanitized: '{token_san}').")
        return best

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
            elif op == "+1":  # directly left: X + 1 == Y.
                self.model.Add(self.var[cat1][attr1] + 1 == self.var[cat2][attr2])
                if self.debug:
                    print(f"Added constraint: [{cat1}][{attr1}] + 1 == [{cat2}][{attr2}]")
            elif op == "-1":  # directly right: X - 1 == Y.
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
            # There are houses_between houses in between, so difference equals houses_between + 1.
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
        # Remove any leading numbering (e.g., "1. ") and trim the text.
        text = re.sub(r'^\d+\.\s*', '', clue).strip()
        if self.debug:
            print(f"Processing clue: {text}")
        ordinal_numbers = r"(?:\d+|first|second|third|fourth|fifth|sixth)"

        # Fixed house pattern.
        m_fixed = re.search(rf"(.+?) is in the ({ordinal_numbers}) house", text, re.IGNORECASE)
        if m_fixed:
            token = m_fixed.group(1).strip()
            num_str = m_fixed.group(2).strip().lower()
            house_num = int(num_str) if num_str.isdigit() else ordinal_map.get(num_str)
            if house_num is not None:
                self.apply_constraint_fixed(token, house_num)
                return

        # Inequality fixed house.
        m_not = re.search(rf"(.+?) is not in the ({ordinal_numbers}) house", text, re.IGNORECASE)
        if m_not:
            token = m_not.group(1).strip()
            num_str = m_not.group(2).strip().lower()
            house_num = int(num_str) if num_str.isdigit() else ordinal_map.get(num_str)
            if house_num is not None:
                self.apply_constraint_inequality(token, house_num)
                return

        # Directly left.
        m_left = re.search(r"(.+?) is directly left of (.+)", text, re.IGNORECASE)
        if m_left:
            token1 = m_left.group(1).strip()
            token2 = m_left.group(2).strip()
            self.apply_constraint_position(token1, "+1", token2)
            return

        # Directly right.
        m_right = re.search(r"(.+?) is directly right of (.+)", text, re.IGNORECASE)
        if m_right:
            token1 = m_right.group(1).strip()
            token2 = m_right.group(2).strip()
            self.apply_constraint_position(token1, "-1", token2)
            return

        # Somewhere to the left.
        m_sl = re.search(r"(.+?) is somewhere to the left of (.+)", text, re.IGNORECASE)
        if m_sl:
            token1 = m_sl.group(1).strip()
            token2 = m_sl.group(2).strip()
            self.apply_constraint_position(token1, "<", token2)
            return

        # Somewhere to the right.
        m_sr = re.search(r"(.+?) is somewhere to the right of (.+)", text, re.IGNORECASE)
        if m_sr:
            token1 = m_sr.group(1).strip()
            token2 = m_sr.group(2).strip()
            self.apply_constraint_position(token1, ">", token2)
            return

        # Next-to.
        m_next = re.search(r"(.+?) and (.+?) are next to each other", text, re.IGNORECASE)
        if m_next:
            token1 = m_next.group(1).strip()
            token2 = m_next.group(2).strip()
            self.apply_constraint_next_to(token1, token2)
            return

        # Between pattern: accepts both "There is" and "There are".
        m_between = re.search(rf"There (?:are|is) (\d+|one|two|three|four|five|six) house(?:s)? between (.+?) and (.+)", text, re.IGNORECASE)
        if m_between:
            num_str = m_between.group(1).strip().lower()
            houses_between = int(num_str) if num_str.isdigit() else word_to_num.get(num_str)
            token1 = m_between.group(2).strip()
            token2 = m_between.group(3).strip()
            self.apply_constraint_between(token1, token2, houses_between)
            return

        # Equality pattern: now accepts both "is" and "is the".
        m_eq = re.search(r"(.+?) is(?: the)? (.+)", text, re.IGNORECASE)
        if m_eq:
            token1 = m_eq.group(1).strip()
            token2 = m_eq.group(2).strip()
            self.apply_constraint_equality(token1, token2)
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
        if solution:
            headers = ["House"] + [shorten_category(cat) for cat in self.categories.keys()]
            table = []
            for house in sorted(solution.keys()):
                row = [str(house)]
                for cat in self.categories.keys():
                    row.append(solution[house].get(cat, ""))
                table.append(row)
            # Determine column widths.
            col_widths = [max(len(str(row[i])) for row in ([headers] + table))
                          for i in range(len(headers))]
            header_line = " | ".join(str(headers[i]).ljust(col_widths[i]) for i in range(len(headers)))
            separator_line = "-+-".join("-" * col_widths[i] for i in range(len(headers)))
            print(header_line)
            print(separator_line)
            for row in table:
                print(" | ".join(str(row[i]).ljust(col_widths[i]) for i in range(len(headers))))
        else:
            print("No solution found.")

def main():
    parser = argparse.ArgumentParser(description="Robust Generic Logic Puzzle Solver with Configurable Category Keys")
    parser.add_argument("puzzle_file", help="Path to the puzzle description text file")
    parser.add_argument("--debug", action="store_true", help="Enable debug output")
    args = parser.parse_args()
    try:
        with open(args.puzzle_file, "r", encoding="utf-8") as f:
            puzzle_text = f.read()
    except Exception as e:
        print("Error reading file:", e)
        sys.exit(1)

    solver_instance = PuzzleSolver(puzzle_text, debug=args.debug)
    solver_instance.parse_puzzle()
    solver_instance.build_variables()
    solver_instance.process_all_clues()
    solution = solver_instance.solve()
    solver_instance.print_solution(solution)

if __name__ == "__main__":
    main()