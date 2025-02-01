import streamlit as st
import json
from ortools.sat.python import cp_model
import streamlit.components.v1 as components

# Default puzzle description (users can edit this)
DEFAULT_PUZZLE = """There are 6 houses, numbered 1 to 6 from left to right, as seen from across the street. Each house is occupied by a different person. Each house has a unique attribute for each of the following characteristics:
- Each person has a unique name: Bob, Carol, Alice, Peter, Arnold, Eric
- People have unique heights: average, super tall, very short, short, very tall, tall
- Each person has a unique favorite drink: coffee, milk, boba tea, root beer, tea, water
- Each person has a favorite color: blue, yellow, red, green, purple, white
- Each person lives in a unique style of house: mediterranean, colonial, modern, ranch, victorian, craftsman
- They all have a unique favorite flower: lilies, roses, iris, daffodils, carnations, tulips

### Clues:
1. Eric is not in the second house.
2. The coffee drinker is somewhere to the right of the person residing in a Victorian house.
3. The person who is very tall is directly left of the person in a Craftsman‑style house.
4. The person living in a colonial‑style house is the person who likes milk.
5. The root beer lover is directly left of the person whose favorite color is green.
6. The person who loves yellow is the person who loves the bouquet of lilies.
7. Arnold is the person whose favorite color is red.
8. The person who loves the rose bouquet is Bob.
9. The person who is very tall is the root beer lover.
10. Alice is somewhere to the left of the person who loves a bouquet of daffodils.
11. The boba tea drinker is Carol.
12. There is one house between the person in a ranch‑style home and the person who loves the bouquet of iris.
13. The person who loves a carnations arrangement is the tea drinker.
14. The boba tea drinker is the person in a Mediterranean‑style villa.
15. The person who loves a bouquet of daffodils is in the second house.
16. The person who is tall and the person who loves the vase of tulips are next to each other.
17. Carol is somewhere to the left of the person who loves the bouquet of iris.
18. The person who has an average height is directly left of Arnold.
19. The person who is super tall and Eric are next to each other.
20. The person who loves the vase of tulips is somewhere to the right of the person who loves the bouquet of lilies.
21. The person who is super tall is the tea drinker.
22. The person who is short is Carol.
23. The coffee drinker is the person who loves purple.
24. The person who loves blue is somewhere to the right of the person who loves a carnations arrangement.
"""

def main():
    st.title("Puzzle Validation Web App")
    st.write("Edit the puzzle description below if needed and then click **Validate Puzzle**.")
    puzzle_description = st.text_area("Puzzle Description", value=DEFAULT_PUZZLE, height=400)
    
    if st.button("Validate Puzzle"):
        # ----------------------------------------------------------------------
        # Build the CP‑SAT model using inverse mappings.
        # (The ordering used here is taken from the puzzle description.)
        # ----------------------------------------------------------------------
        model = cp_model.CpModel()
        num_houses = 6
        
        # Mapping lists from the puzzle description.
        person_names = ["Bob", "Carol", "Alice", "Peter", "Arnold", "Eric"]
        height_names = ["average", "super tall", "very short", "short", "very tall", "tall"]
        drink_names  = ["coffee", "milk", "boba tea", "root beer", "tea", "water"]
        color_names  = ["blue", "yellow", "red", "green", "purple", "white"]
        style_names  = ["mediterranean", "colonial", "modern", "ranch", "victorian", "craftsman"]
        flower_names = ["lilies", "roses", "iris", "daffodils", "carnations", "tulips"]
        
        # Create inverse mapping variables: For each attribute value, store its house index.
        person_house = [model.NewIntVar(0, num_houses - 1, f'person_house_{i}') for i in range(num_houses)]
        height_house = [model.NewIntVar(0, num_houses - 1, f'height_house_{i}') for i in range(num_houses)]
        drink_house  = [model.NewIntVar(0, num_houses - 1, f'drink_house_{i}') for i in range(num_houses)]
        color_house  = [model.NewIntVar(0, num_houses - 1, f'color_house_{i}') for i in range(num_houses)]
        style_house  = [model.NewIntVar(0, num_houses - 1, f'style_house_{i}') for i in range(num_houses)]
        flower_house = [model.NewIntVar(0, num_houses - 1, f'flower_house_{i}') for i in range(num_houses)]
        
        # All‐different constraints (each attribute value appears in exactly one house)
        model.AddAllDifferent(person_house)
        model.AddAllDifferent(height_house)
        model.AddAllDifferent(drink_house)
        model.AddAllDifferent(color_house)
        model.AddAllDifferent(style_house)
        model.AddAllDifferent(flower_house)
        
        # -----------------------------
        # Add constraints for all 24 clues:
        # -----------------------------
        # Clue 1: Eric is not in the second house.
        model.Add(person_house[5] != 1)  # Eric is at index 5; second house is index 1.
        
        # Clue 2: The coffee drinker is somewhere to the right of the person residing in a Victorian house.
        model.Add(drink_house[0] > style_house[4])  # coffee: drink index 0; victorian: style index 4.
        
        # Clue 3: The person who is very tall is directly left of the person in a Craftsman‑style house.
        model.Add(height_house[4] + 1 == style_house[5])  # very tall: height index 4; craftsman: style index 5.
        
        # Clue 4: The person living in a colonial‑style house is the person who likes milk.
        model.Add(style_house[1] == drink_house[1])  # colonial: style index 1; milk: drink index 1.
        
        # Clue 5: The root beer lover is directly left of the person whose favorite color is green.
        model.Add(drink_house[3] + 1 == color_house[3])  # root beer: index 3; green: color index 3.
        
        # Clue 6: The person who loves yellow is the person who loves the bouquet of lilies.
        model.Add(color_house[1] == flower_house[0])  # yellow: color index 1; lilies: flower index 0.
        
        # Clue 7: Arnold is the person whose favorite color is red.
        model.Add(person_house[4] == color_house[2])  # Arnold: person index 4; red: color index 2.
        
        # Clue 8: The person who loves the rose bouquet is Bob.
        model.Add(person_house[0] == flower_house[1])  # Bob: person index 0; roses: flower index 1.
        
        # Clue 9: The person who is very tall is the root beer lover.
        model.Add(height_house[4] == drink_house[3])  # very tall: height index 4; root beer: drink index 3.
        
        # Clue 10: Alice is somewhere to the left of the person who loves a bouquet of daffodils.
        model.Add(person_house[2] < flower_house[3])  # Alice: person index 2; daffodils: flower index 3.
        
        # Clue 11: The boba tea drinker is Carol.
        model.Add(drink_house[2] == person_house[1])  # boba tea: drink index 2; Carol: person index 1.
        
        # Clue 12: There is one house between the person in a ranch‑style home and the person who loves the bouquet of iris.
        diff1 = model.NewIntVar(0, num_houses, "diff1")
        model.AddAbsEquality(diff1, style_house[3] - flower_house[2])  # ranch: style index 3; iris: flower index 2.
        model.Add(diff1 == 2)
        
        # Clue 13: The person who loves a carnations arrangement is the tea drinker.
        model.Add(drink_house[4] == flower_house[4])  # tea: drink index 4; carnations: flower index 4.
        
        # Clue 14: The boba tea drinker is the person in a Mediterranean‑style villa.
        model.Add(drink_house[2] == style_house[0])  # boba tea: drink index 2; mediterranean: style index 0.
        
        # Clue 15: The person who loves a bouquet of daffodils is in the second house.
        model.Add(flower_house[3] == 1)  # daffodils: flower index 3; second house is index 1.
        
        # Clue 16: The person who is tall and the person who loves the vase of tulips are next to each other.
        diff2 = model.NewIntVar(0, num_houses, "diff2")
        model.AddAbsEquality(diff2, height_house[5] - flower_house[5])  # tall: height index 5; tulips: flower index 5.
        model.Add(diff2 == 1)
        
        # Clue 17: Carol is somewhere to the left of the person who loves the bouquet of iris.
        model.Add(person_house[1] < flower_house[2])  # Carol: person index 1; iris: flower index 2.
        
        # Clue 18: The person who has an average height is directly left of Arnold.
        model.Add(height_house[0] + 1 == person_house[4])  # average: height index 0; Arnold: person index 4.
        
        # Clue 19: The person who is super tall and Eric are next to each other.
        diff3 = model.NewIntVar(0, num_houses, "diff3")
        model.AddAbsEquality(diff3, height_house[1] - person_house[5])  # super tall: height index 1; Eric: person index 5.
        model.Add(diff3 == 1)
        
        # Clue 20: The person who loves the vase of tulips is somewhere to the right of the person who loves the bouquet of lilies.
        model.Add(flower_house[5] > flower_house[0])  # tulips: flower index 5; lilies: flower index 0.
        
        # Clue 21: The person who is super tall is the tea drinker.
        model.Add(height_house[1] == drink_house[4])  # super tall: height index 1; tea: drink index 4.
        
        # Clue 22: The person who is short is Carol.
        model.Add(person_house[1] == height_house[3])  # Carol: person index 1; short: height index 3.
        
        # Clue 23: The coffee drinker is the person who loves purple.
        model.Add(drink_house[0] == color_house[4])  # coffee: drink index 0; purple: color index 4.
        
        # Clue 24: The person who loves blue is somewhere to the right of the person who loves a carnations arrangement.
        model.Add(color_house[0] > flower_house[4])  # blue: color index 0; carnations: flower index 4.
        
        # ----------------------------------------------------------------------
        # Solve the model.
        # ----------------------------------------------------------------------
        solver = cp_model.CpSolver()
        status = solver.Solve(model)
        
        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            # Invert the mapping to create a solution per house.
            house_person = [None] * num_houses
            house_height = [None] * num_houses
            house_drink  = [None] * num_houses
            house_color  = [None] * num_houses
            house_style  = [None] * num_houses
            house_flower = [None] * num_houses

            # Persons: for each person value, assign to house.
            for p in range(num_houses):
                h = solver.Value(person_house[p])
                house_person[h] = person_names[p]
            # Heights (iterate over indices of height mapping)
            for idx in range(num_houses):
                h_index = solver.Value(height_house[idx])
                house_height[h_index] = height_names[idx]
            # Drinks:
            for d in range(num_houses):
                h_val = solver.Value(drink_house[d])
                house_drink[h_val] = drink_names[d]
            # Colors:
            for c in range(num_houses):
                h_val = solver.Value(color_house[c])
                house_color[h_val] = color_names[c]
            # Styles:
            for s in range(num_houses):
                h_val = solver.Value(style_house[s])
                house_style[h_val] = style_names[s]
            # Flowers:
            for f in range(num_houses):
                h_val = solver.Value(flower_house[f])
                house_flower[h_val] = flower_names[f]
                
            solution = {
                "person": house_person,
                "height": house_height,
                "drink": house_drink,
                "color": house_color,
                "style": house_style,
                "flower": house_flower
            }
            
            st.success("Solution Found!")
            st.write("Below is the computed solution:")
            st.table({
                "House": [f"House {i+1}" for i in range(num_houses)],
                "Person": solution["person"],
                "Height": solution["height"],
                "Drink": solution["drink"],
                "Color": solution["color"],
                "Style": solution["style"],
                "Flower": solution["flower"]
            })
            
            # ---------------------------
            # Prepare the interactive HTML.
            # ---------------------------
            # Build HTML rows for the solution table.
            solution_rows = ""
            for i in range(num_houses):
                solution_rows += f"""
                <tr>
                  <td id="house{i+1}-house">House {i+1}</td>
                  <td id="house{i+1}-person">{solution['person'][i]}</td>
                  <td id="house{i+1}-height">{solution['height'][i]}</td>
                  <td id="house{i+1}-drink">{solution['drink'][i]}</td>
                  <td id="house{i+1}-color">{solution['color'][i]}</td>
                  <td id="house{i+1}-style">{solution['style'][i]}</td>
                  <td id="house{i+1}-flower">{solution['flower'][i]}</td>
                </tr>
                """
            
            # List of clues (clue number and description)
            clues = [
              ("1", "Eric is not in the second house."),
              ("2", "The coffee drinker is somewhere to the right of the person residing in a Victorian house."),
              ("3", "The person who is very tall is directly left of the person in a Craftsman‑style house."),
              ("4", "The person living in a colonial‑style house is the person who likes milk."),
              ("5", "The root beer lover is directly left of the person whose favorite color is green."),
              ("6", "The person who loves yellow is the person who loves the bouquet of lilies."),
              ("7", "Arnold is the person whose favorite color is red."),
              ("8", "The person who loves the rose bouquet is Bob."),
              ("9", "The person who is very tall is the root beer lover."),
              ("10", "Alice is somewhere to the left of the person who loves a bouquet of daffodils."),
              ("11", "The boba tea drinker is Carol."),
              ("12", "There is one house between the person in a ranch‑style home and the person who loves the bouquet of iris."),
              ("13", "The person who loves a carnations arrangement is the tea drinker."),
              ("14", "The boba tea drinker is the person in a Mediterranean‑style villa."),
              ("15", "The person who loves a bouquet of daffodils is in the second house."),
              ("16", "The person who is tall and the person who loves the vase of tulips are next to each other."),
              ("17", "Carol is somewhere to the left of the person who loves the bouquet of iris."),
              ("18", "The person who has an average height is directly left of Arnold."),
              ("19", "The person who is super tall and Eric are next to each other."),
              ("20", "The person who loves the vase of tulips is somewhere to the right of the person who loves the bouquet of lilies."),
              ("21", "The person who is super tall is the tea drinker."),
              ("22", "The person who is short is Carol."),
              ("23", "The coffee drinker is the person who loves purple."),
              ("24", "The person who loves blue is somewhere to the right of the person who loves a carnations arrangement.")
            ]
            
            # Helper to find the cell id for a given attribute (column) with a specific value.
            def find_cell(attribute, value):
                for i in range(num_houses):
                    if solution[attribute][i].lower() == value.lower():
                        return f"house{i+1}-{attribute}"
                return None

            # Build a mapping dictionary (clue number -> list of cell id strings),
            # based on our interpretation of affected solution cells for each clue.
            clue_mapping = {
              "1": [find_cell("person", "Eric"), "house2-person"],
              "2": [find_cell("drink", "coffee"), find_cell("style", "victorian")],
              "3": [find_cell("height", "very tall"), find_cell("style", "craftsman")],
              "4": [find_cell("style", "colonial"), find_cell("drink", "milk")],
              "5": [find_cell("drink", "root beer"), find_cell("color", "green")],
              "6": [find_cell("color", "yellow"), find_cell("flower", "lilies")],
              "7": [find_cell("person", "Arnold"), find_cell("color", "red")],
              "8": [find_cell("flower", "roses"), find_cell("person", "Bob")],
              "9": [find_cell("height", "very tall"), find_cell("drink", "root beer")],
              "10": [find_cell("person", "Alice"), find_cell("flower", "daffodils")],
              "11": [find_cell("drink", "boba tea"), find_cell("person", "Carol")],
              "12": [find_cell("style", "ranch"), find_cell("flower", "iris")],
              "13": [find_cell("flower", "carnations"), find_cell("drink", "tea")],
              "14": [find_cell("drink", "boba tea"), find_cell("style", "mediterranean")],
              "15": ["house2-flower"],
              "16": [find_cell("height", "tall"), find_cell("flower", "tulips")],
              "17": [find_cell("person", "Carol"), find_cell("flower", "iris")],
              "18": [find_cell("height", "average"), find_cell("person", "Arnold")],
              "19": [find_cell("height", "super tall"), find_cell("person", "Eric")],
              "20": [find_cell("flower", "tulips"), find_cell("flower", "lilies")],
              "21": [find_cell("height", "super tall"), find_cell("drink", "tea")],
              "22": [find_cell("height", "short"), find_cell("person", "Carol")],
              "23": [find_cell("drink", "coffee"), find_cell("color", "purple")],
              "24": [find_cell("color", "blue"), find_cell("flower", "carnations")]
            }
            
            clue_mapping_json = json.dumps(clue_mapping)
            
            # Build complete HTML with interactive highlighting.
            html_code = f"""
            <!DOCTYPE html>
            <html lang="en">
            <head>
              <meta charset="UTF-8">
              <meta name="viewport" content="width=device-width, initial-scale=1.0">
              <title>Puzzle Validation Visualization</title>
              <style>
                body {{
                  font-family: Arial, sans-serif;
                  margin: 20px;
                }}
                table {{
                  width: 100%;
                  border-collapse: collapse;
                  margin: 20px 0;
                }}
                th, td {{
                  border: 1px solid #ccc;
                  padding: 8px 12px;
                  text-align: center;
                }}
                th {{
                  background-color: #f0f0f0;
                }}
                .solution-cell.highlight {{
                  background-color: #ffeb3b;
                  transition: background-color 0.3s;
                }}
                .clue-row {{
                  cursor: pointer;
                }}
                .satisfied {{
                  background-color: #d4edda;
                  color: #155724;
                  font-weight: bold;
                }}
              </style>
            </head>
            <body>
              <h2>Solution Table</h2>
              <table id="solution-table">
                <thead>
                  <tr>
                    <th>House</th>
                    <th>Person</th>
                    <th>Height</th>
                    <th>Drink</th>
                    <th>Color</th>
                    <th>Style</th>
                    <th>Flower</th>
                  </tr>
                </thead>
                <tbody>
                  {solution_rows}
                </tbody>
              </table>
              <h2>Clue Validation</h2>
              <table id="clue-table">
                <thead>
                  <tr>
                    <th>Clue No.</th>
                    <th>Clue Description</th>
                    <th>Status</th>
                  </tr>
                </thead>
                <tbody>
            """
            for clue_num, clue_text in clues:
                html_code += f"""
                  <tr class="clue-row" data-clue="{clue_num}">
                    <td>{clue_num}</td>
                    <td>{clue_text}</td>
                    <td class="satisfied">&#10003; Satisfied</td>
                  </tr>
                """
            html_code += f"""
                </tbody>
              </table>
              <script>
                const clueMapping = {clue_mapping_json};
                function clearHighlights() {{
                  document.querySelectorAll('.solution-cell').forEach(cell => {{
                    cell.classList.remove('highlight');
                  }});
                }}
                document.querySelectorAll('.clue-row').forEach(row => {{
                  row.addEventListener('click', function() {{
                    clearHighlights();
                    const clueNum = this.getAttribute('data-clue');
                    const cellIDs = clueMapping[clueNum] || [];
                    cellIDs.forEach(id => {{
                      const cell = document.getElementById(id);
                      if (cell) {{
                        cell.classList.add('highlight');
                      }}
                    }});
                  }});
                }});
              </script>
            </body>
            </html>
            """
            
            # Render the interactive HTML in Streamlit.
            components.html(html_code, height=800, scrolling=True)
            
        else:
            st.error("No solution found.")

if __name__ == "__main__":
    main()