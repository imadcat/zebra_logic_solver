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

# A mapping for ordinal words (assumes up to six houses).
ordinal_map = {
    "first": 1,
    "second": 2,
    "third": 3,
    "fourth": 4,
    "fifth": 5,
    "sixth": 6
}

def sanitize_token(text):
    """
    Lowercase and remove common noise phrases from the token.
    Also strips extra punctuation.
    """
    text = text.lower()
    # Remove common noise phrases.
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
        r'\bmany unique\b'
    ]
    for phrase in noise_phrases:
        text = re.sub(phrase, '', text, flags=re.IGNORECASE)
    # Remove articles (a, an, the) to lessen noise.
    text = re.sub(r'\b(a|an|the)\b', '', text)
    # Remove punctuation and extra whitespace.
    text = re.sub(r'[^a-z0-9 ]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

class PuzzleSolver:
    def __init__(self, puzzle_text):
        self.puzzle_text = puzzle_text
        self.num_houses = None
        # Categories: key = bullet line text before colon, value = list of attributes.
        self.categories = {}
        self.clues = []  # list of clues (strings)
        # CP variables: self.var[category][attribute] = variable holding the assigned house number.
        self.var = {}
        self.model = cp_model.CpModel()

    def parse_puzzle(self):
        # Identify the number of houses from text (default to 6 if not found)
        m = re.search(r"There are (\d+) houses", self.puzzle_text, re.IGNORECASE)
        if m:
            self.num_houses = int(m.group(1))
        else:
            self.num_houses = 6

        # Parse categories.
        # Expect bullet lines (starting with "-" or "*") of the form:
        # - <Category text>: attribute1, attribute2, attribute3, ...
        cat_pattern = re.compile(r"^[-*]\s*(.*?):\s*(.+)$")
        for line in self.puzzle_text.splitlines():
            line = line.strip()
            m = cat_pattern.match(line)
            if m:
                cat_label = m.group(1).strip()
                attr_line = m.group(2).strip()
                attrs = [x.strip() for x in attr_line.split(",") if x.strip()]
                self.categories[cat_label] = attrs

        # Parse clues.
        # Look for a line containing "### Clues:" then all subsequent non-empty lines are clues.
        clues_section = False
        for line in self.puzzle_text.splitlines():
            if "### Clues:" in line:
                clues_section = True
                continue
            if clues_section:
                clean = line.strip()
                if clean:
                    self.clues.append(clean)

    def build_variables(self):
        # For every category (use full bullet text as key) create an int var for each attribute.
        for cat, attrs in self.categories.items():
            self.var[cat] = {}
            for attr in attrs:
                self.var[cat][attr] = self.model.NewIntVar(1, self.num_houses, f"{cat}_{attr}")
            self.model.AddAllDifferent(list(self.var[cat].values()))

    def find_attribute(self, token):
        """
        Find an attribute whose sanitized text appears as a whole word in the sanitized token.
        Uses a longest–match heuristic and a simple singular/plural check. Returns (category, attribute) or None.
        """
        token_san = sanitize_token(token)
        best = None
        best_len = 0
        # Skip if token is too generic.
        if token_san in {"", "who", "whose"}:
            return None
        for cat, attrs in self.categories.items():
            for attr in attrs:
                attr_san = sanitize_token(attr)
                # Whole word match:
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

    def apply_constraint_equality(self, token1, token2):
        # Skip if tokens are too trivial.
        if sanitize_token(token1) in {"", "who", "whose"} or sanitize_token(token2) in {"", "who", "whose"}:
            return
        a1 = self.find_attribute(token1)
        a2 = self.find_attribute(token2)
        if a1 and a2:
            cat1, attr1 = a1
            cat2, attr2 = a2
            self.model.Add(self.var[cat1][attr1] == self.var[cat2][attr2])
        else:
            print(f"Warning: could not apply equality between '{token1}' and '{token2}'")

    def apply_constraint_inequality(self, token, house_number):
        a1 = self.find_attribute(token)
        if a1:
            cat, attr = a1
            self.model.Add(self.var[cat][attr] != house_number)
        else:
            print(f"Warning: could not apply inequality for '{token}' at house {house_number}")

    def apply_constraint_position(self, token1, op, token2):
        a1 = self.find_attribute(token1)
        a2 = self.find_attribute(token2)
        if a1 and a2:
            cat1, attr1 = a1
            cat2, attr2 = a2
            if op == "==":
                self.model.Add(self.var[cat1][attr1] == self.var[cat2][attr2])
            elif op == "<":
                self.model.Add(self.var[cat1][attr1] < self.var[cat2][attr2])
            elif op == ">":
                self.model.Add(self.var[cat1][attr1] > self.var[cat2][attr2])
            elif op == "+1":  # directly left: X + 1 == Y.
                self.model.Add(self.var[cat1][attr1] + 1 == self.var[cat2][attr2])
            elif op == "-1":  # directly right: X - 1 == Y.
                self.model.Add(self.var[cat1][attr1] - 1 == self.var[cat2][attr2])
        else:
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
        else:
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
        else:
            print(f"Warning: could not apply between constraint between '{token1}' and '{token2}' with {houses_between} houses in between")

    def apply_constraint_fixed(self, token, house_number):
        a1 = self.find_attribute(token)
        if a1:
            cat, attr = a1
            self.model.Add(self.var[cat][attr] == house_number)
        else:
            print(f"Warning: could not apply fixed constraint for '{token}' at house {house_number}")

    def process_clue(self, clue):
        # Remove any leading numbering (e.g. "1. ") and trim.
        text = re.sub(r'^\d+\.\s*', '', clue).strip()
        # First, try a "not in" fixed–house pattern.
        ordinal_numbers = r"(?:\d+|first|second|third|fourth|fifth|sixth)"
        m_not = re.search(rf"(.+?) is not in the ({ordinal_numbers}) house", text, re.IGNORECASE)
        if m_not:
            token = m_not.group(1).strip()
            num_str = m_not.group(2).strip().lower()
            house_num = int(num_str) if num_str.isdigit() else ordinal_map.get(num_str, None)
            if house_num is not None:
                self.apply_constraint_inequality(token, house_num)
                return

        # Fixed house pattern.
        m_fixed = re.search(rf"(.+?) is in the ({ordinal_numbers}) house", text, re.IGNORECASE)
        if m_fixed:
            token = m_fixed.group(1).strip()
            num_str = m_fixed.group(2).strip().lower()
            house_num = int(num_str) if num_str.isdigit() else ordinal_map.get(num_str, None)
            if house_num is not None:
                self.apply_constraint_fixed(token, house_num)
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

        # Somewhere to the left / right.
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

        # Next to.
        m_next = re.search(r"(.+?) and (.+?) are next to each other", text, re.IGNORECASE)
        if m_next:
            token1 = m_next.group(1).strip()
            token2 = m_next.group(2).strip()
            self.apply_constraint_next_to(token1, token2)
            return

        # Between pattern.
        m_between = re.search(r"There are (\d+) houses? between (.+?) and (.+)", text, re.IGNORECASE)
        if m_between:
            houses_between = int(m_between.group(1))
            token1 = m_between.group(2).strip()
            token2 = m_between.group(3).strip()
            self.apply_constraint_between(token1, token2, houses_between)
            return

        # Equality pattern: "X is the Y"
        m_eq = re.search(r"(.+?) is the (.+)", text, re.IGNORECASE)
        if m_eq:
            token1 = m_eq.group(1).strip()
            token2 = m_eq.group(2).strip()
            self.apply_constraint_equality(token1, token2)
            return

        # Fallback: if exactly two attributes are mentioned (using our attribute finder), equate them.
        found = self.find_all_attributes_in_text(text)
        if len(found) == 2:
            (cat1, attr1), (cat2, attr2) = found
            self.model.Add(self.var[cat1][attr1] == self.var[cat2][attr2])
            return

        # Optionally, use spaCy's noun-chunk extraction as a last resort.
        if nlp:
            doc = nlp(text)
            chunks = [chunk.text.strip() for chunk in doc.noun_chunks if chunk.text.strip()]
            if len(chunks) >= 2:
                self.apply_constraint_equality(chunks[0], chunks[1])
                return

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
            return None

    def print_solution(self, solution):
        if solution:
            print("Solution:")
            for house in sorted(solution.keys()):
                print(f"House {house}:")
                for cat, attr in solution[house].items():
                    print(f"  {cat}: {attr}")
        else:
            print("No solution found.")

def main():
    parser = argparse.ArgumentParser(description="Refined Generic Logic Puzzle Solver")
    parser.add_argument("puzzle_file", help="Path to the puzzle description text file")
    args = parser.parse_args()
    try:
        with open(args.puzzle_file, "r", encoding="utf-8") as f:
            puzzle_text = f.read()
    except Exception as e:
        print("Error reading file:", e)
        sys.exit(1)

    solver_instance = PuzzleSolver(puzzle_text)
    solver_instance.parse_puzzle()
    solver_instance.build_variables()
    solver_instance.process_all_clues()
    solution = solver_instance.solve()
    solver_instance.print_solution(solution)

if __name__ == "__main__":
    main()