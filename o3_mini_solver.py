#!/usr/bin/env python3
from ortools.sat.python import cp_model

def main():
    model = cp_model.CpModel()
    num_houses = 6

    # Mapping lists for each attribute.
    person_names = ["Alice", "Peter", "Bob", "Eric", "Carol", "Arnold"]
    height_names = ["Very tall", "Tall", "Average", "Short", "Super tall", "Very short"]
    drink_names  = ["Root beer", "Water", "Milk", "Coffee", "Boba tea", "Tea"]
    color_names  = ["Yellow", "Green", "White", "Purple", "Red", "Blue"]
    style_names  = ["Victorian", "Craftsman", "Colonial", "Mediterranean", "Ranch", "Modern"]
    flower_names = ["Lilies", "Daffodils", "Roses", "Iris", "Tulips", "Carnations"]

    # For each attribute, create a permutation variable where X_house[v] is the house
    # (0-indexed) where that attribute value (with index v) is located.
    person_house = [model.NewIntVar(0, num_houses - 1, f'person_house_{i}') for i in range(num_houses)]
    height_house = [model.NewIntVar(0, num_houses - 1, f'height_house_{i}') for i in range(num_houses)]
    drink_house  = [model.NewIntVar(0, num_houses - 1, f'drink_house_{i}') for i in range(num_houses)]
    color_house  = [model.NewIntVar(0, num_houses - 1, f'color_house_{i}') for i in range(num_houses)]
    style_house  = [model.NewIntVar(0, num_houses - 1, f'style_house_{i}') for i in range(num_houses)]
    flower_house = [model.NewIntVar(0, num_houses - 1, f'flower_house_{i}') for i in range(num_houses)]

    # Each attribute must be assigned to exactly one unique house.
    model.AddAllDifferent(person_house)
    model.AddAllDifferent(height_house)
    model.AddAllDifferent(drink_house)
    model.AddAllDifferent(color_house)
    model.AddAllDifferent(style_house)
    model.AddAllDifferent(flower_house)

    # ------ Clues as Constraints ------

    # 1. Eric is not in House 2.
    # "Eric" is person_names index 3; House 2 is index 1.
    model.Add(person_house[3] != 1)

    # 2. The coffee drinker is somewhere right of the Victorian house.
    # "Coffee" is drink index 3; "Victorian" is style index 0.
    model.Add(drink_house[3] > style_house[0])

    # 3. The very tall person is immediately left of the Craftsman house.
    # "Very tall" is height index 0; "Craftsman" is style index 1.
    model.Add(height_house[0] + 1 == style_house[1])

    # 4. The Colonial house is paired with the milk drinker.
    # "Colonial" is style index 2; "Milk" is drink index 2.
    model.Add(style_house[2] == drink_house[2])

    # 5. The root beer lover is immediately left of the green-lover.
    # "Root beer" is drink index 0; "Green" is color index 1.
    model.Add(drink_house[0] + 1 == color_house[1])

    # 6. The yellow lover also loves lilies.
    # "Yellow" is color index 0; "Lilies" is flower index 0.
    model.Add(color_house[0] == flower_house[0])

    # 7. Arnold’s favorite color is red.
    # "Arnold" is person index 5; "Red" is color index 4.
    model.Add(person_house[5] == color_house[4])

    # 8. Bob loves roses.
    # "Bob" is person index 2; "Roses" is flower index 2.
    model.Add(person_house[2] == flower_house[2])

    # 9. The very tall person is the root beer drinker.
    # "Very tall" is height index 0; "Root beer" is drink index 0.
    model.Add(height_house[0] == drink_house[0])

    # 10. Alice is left of the daffodils lover.
    # "Alice" is person index 0; "Daffodils" is flower index 1.
    model.Add(person_house[0] < flower_house[1])

    # 11. The boba tea drinker is Carol.
    # "Boba tea" is drink index 4; "Carol" is person index 4.
    model.Add(drink_house[4] == person_house[4])

    # 12. One house between the Ranch house and the iris-lover.
    # "Ranch" is style index 4; "Iris" is flower index 3.
    diff1 = model.NewIntVar(0, num_houses, 'diff1')
    model.AddAbsEquality(diff1, style_house[4] - flower_house[3])
    model.Add(diff1 == 2)

    # 13. The tea drinker is the super tall person who loves carnations.
    # "Tea" is drink index 5; "Super tall" is height index 4; "Carnations" is flower index 5.
    model.Add(drink_house[5] == height_house[4])
    model.Add(drink_house[5] == flower_house[5])

    # 14. Carol lives in a Mediterranean-style house.
    # "Carol" is person index 4; "Mediterranean" is style index 3.
    model.Add(person_house[4] == style_house[3])

    # 15. The daffodils are in House 2.
    # Daffodils (flower index 1) must be in house index 1.
    model.Add(flower_house[1] == 1)

    # 16. The tall person and the tulips lover are neighbors.
    # "Tall" is height index 1; "Tulips" is flower index 4.
    diff2 = model.NewIntVar(0, num_houses, 'diff2')
    model.AddAbsEquality(diff2, height_house[1] - flower_house[4])
    model.Add(diff2 == 1)

    # 17. Carol is left of the iris-lover.
    # "Carol" is person index 4; "Iris" is flower index 3.
    model.Add(person_house[4] < flower_house[3])

    # 18. The average height person is immediately left of Arnold.
    # "Average" is height index 2; "Arnold" is person index 5.
    model.Add(height_house[2] + 1 == person_house[5])

    # 19. The super tall person and Eric are neighbors.
    # "Super tall" is height index 4; "Eric" is person index 3.
    diff3 = model.NewIntVar(0, num_houses, 'diff3')
    model.AddAbsEquality(diff3, height_house[4] - person_house[3])
    model.Add(diff3 == 1)

    # 20. Tulips are right of the lilies.
    # "Tulips" is flower index 4; "Lilies" is flower index 0.
    model.Add(flower_house[4] > flower_house[0])

    # 21. The super tall person is the tea drinker.
    # (This is the same as part of clue 13.)
    model.Add(height_house[4] == drink_house[5])

    # 22. Carol is short.
    # "Carol" is person index 4; "Short" is height index 3.
    model.Add(person_house[4] == height_house[3])

    # 23. The coffee drinker loves purple.
    # "Coffee" is drink index 3; "Purple" is color index 3.
    model.Add(drink_house[3] == color_house[3])

    # ------ Solve the Model ------

    solver = cp_model.CpSolver()
    status = solver.Solve(model)

    if status in (cp_model.FEASIBLE, cp_model.OPTIMAL):
        # Now invert the inverse mapping: for each house, find which attribute value is assigned.
        house_person = [None] * num_houses
        house_height = [None] * num_houses
        house_drink  = [None] * num_houses
        house_color  = [None] * num_houses
        house_style  = [None] * num_houses
        house_flower = [None] * num_houses

        # For persons:
        for p in range(num_houses):
            h = solver.Value(person_house[p])
            house_person[h] = person_names[p]
        # Heights:
        for h_val in range(num_houses):
            for s in range(num_houses):
                if solver.Value(height_house[s]) == h_val:
                    house_height[h_val] = height_names[s]
                    break
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

        print("Solution Found!")
        header = "{:<6}\t{:<8}\t{:<8}\t{:<8}\t{:<8}\t{:<16}\t{:<8}".format("House", "Person", "Height", "Drink", "Color", "Style", "Flower")
        print(header)
        for i in range(num_houses):
            print("{:<6}\t{:<8}\t{:<8}\t{:<8}\t{:<8}\t{:<16}\t{:<8}".format(f"House {i+1}", house_person[i], house_height[i], house_drink[i], house_color[i], house_style[i], house_flower[i]))
    else:
        print("No solution found.")

if __name__ == '__main__':
    main()