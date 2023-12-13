# Importing the libraries
from pulp import *
from scipy.stats import kendalltau, spearmanr
from itertools import combinations

print("\n############################################################################################################################")
print("Exercise 1:-")

# Creating the linear problem
prob = LpProblem("Toy_Manufacturing_Unit", LpMaximize)

# Creating problem variables
x = LpVariable("Toy_A", 0, None, LpInteger)
y = LpVariable("Toy_B", 0, None, LpInteger)

# Adding the objective function
prob += 25 * x + 20 * y ,"Profit_to_be_maximized"

# Adding the other constraints
prob += 20 * x + 12 * y <= 2000, "Resource_constraint"
prob += 5 * x + 5 * y <= 540, "Time_constraint"

# Writing the problem data to .lp file
prob. writeLP("Max_profit.lp")

# Solving the problem using PuLP's solver
prob.solve()

# Printing the status of the problem
print("Status: ", LpStatus[prob.status])

# Printing each of the variables, optimium value
print('\nOptimal Solution:---------------------')
for v in prob.variables():
    print(f"{v.name} = {v.varValue} units.\n")

# Printing the final objective value
print('\nThe maximum profit is achieved to be:', str(int(value(prob.objective))) + '€')







print("\n############################################################################################################################")
print("Exercise 2:-")

# Defining the rankings of the form: duration, price and appreciation
data = {
    'Arc_de_Triomphe': [1, 9.5, 3], 
    'Avenue_des_Champs_Elysees': [1.5, 0, 5], 
    'Basilique_du_Sacre_Coeur': [2, 8, 4], 
    'Catacombes': [2, 10, 4], 
    'Cathedrale_Notre_Dame': [2, 5, 5], 
    'Centre_Pompidou': [2.5, 10, 1], 
    'Eiffel_Tower': [4.5, 15.5, 5], 
    'Jardin_Tuileries': [1.5, 0, 3], 
    'Museum_Louvre': [3, 12, 4], 
    'Museum_Orsay': [2, 11, 2], 
    'Place_de_la_Concorde': [0.75, 0, 3], 
    'Sainte_Chapelle': [1.5, 8.5, 1], 
    'Tour_Montparnasse': [2, 15, 2]
}

def maximize_visit_paris(data, prob):
    """
    Returns the sites to be visited for a particular problem along with the total duration and price.
    """

    ListVisit = []
    for v in prob.variables():
        if v.varValue == 1 and v.name in data.keys():
            ListVisit.append(v.name)

    print("Total number of sites visited by Mr. Doe are: ", int(value(prob.objective)))
    print("The sites that Mr. Doe should visit are: ", ", ".join([f"{x}" for x in ListVisit]))
    print("Total price of Mr. Doe's visit is: " + str(sum([data[v][1] for v in ListVisit])) + " €.")
    print("Total duration of Mr. Doe's visit is: " + str(sum([data[v][0] for v in ListVisit])) + " hours.")
    
    return ListVisit

# Helping function
def is_identical(list1, list2):
    """
    Performs a check on two lists if they are identical
    """
    return sorted(list1) == sorted(list2)

# Defining attractions such that they can be visited either once or none at all
TE = LpVariable("Eiffel_Tower", 0, 1, LpBinary)
ML = LpVariable("Museum_Louvre", 0, 1, LpBinary)
AT = LpVariable("Arc_de_Triomphe", 0, 1, LpBinary)
MO = LpVariable("Museum_Orsay", 0, 1, LpBinary)
JT = LpVariable("Jardin_Tuileries", 0, 1, LpBinary)
CA = LpVariable("Catacombes", 0, 1, LpBinary)
CP = LpVariable("Centre_Pompidou", 0, 1, LpBinary)
CN = LpVariable("Cathedrale_Notre_Dame", 0, 1, LpBinary)
BS = LpVariable("Basilique_du_Sacre_Coeur", 0, 1, LpBinary)
SC = LpVariable("Sainte_Chapelle", 0, 1, LpBinary)
PC = LpVariable("Place_de_la_Concorde", 0, 1, LpBinary)
TM = LpVariable("Tour_Montparnasse", 0, 1, LpBinary)
AC = LpVariable("Avenue_des_Champs_Elysees", 0, 1, LpBinary)

def create_linear_problem(prob_name = "Mr_Doe_Visits_Paris"):
    """
    Creates the linear problem, adding generic objective function along with time and price constraints
    """
    prob = LpProblem(prob_name, LpMaximize)

    # Adding the objective function
    prob += TE + ML + AT + MO + JT + CA + CP + CN + BS + SC + PC + TM + AC # the sum of all visits of sites
 
    # Adding the constraints
    
    # a) Time constraint
    prob += 4.5 * TE + 3 * ML + 1 * AT + 2 * MO + 1.5 * JT + 2 * CA + 2.5 * CP + 2 * CN + 2 * BS + 1.5 * SC + 0.75 * PC + 2 * TM + 1.5 * AC <= 12

    # b) Price constraint
    prob += 15.5 * TE + 12 * ML + 9.5 * AT + 11 * MO + 0 * JT + 10 * CA + 10 * CP + 5 * CN + 8 * BS + 8.5 * SC + 0 * PC + 15 * TM + 0 * AC <= 65
    
    return prob

prob = create_linear_problem("ListVisit1")
prob.solve(PULP_CBC_CMD(msg=False))
print("Status: ", LpStatus[prob.status])
ListVisit1 = maximize_visit_paris(data, prob)

print("\n#########################################")
print("Preference 1:-")
print("If two sites are geographically very close (within a radius of 1 km of walking), he will prefer to visit these two sites instead of visiting only one.\n")

prob = create_linear_problem("ListVisit2_Preference1")

# Creating variables for places which are within the distance of 1 km
MO_JT = pulp.LpVariable("MO_JT", 0, 1, LpBinary)
CP_CN = pulp.LpVariable("CP_CN", 0, 1, LpBinary) 
ML_SC = pulp.LpVariable("ML_SC", 0, 1, LpBinary)
SC_CP = pulp.LpVariable("SC_CP", 0, 1, LpBinary)
CN_SC = pulp.LpVariable("CN_SC", 0, 1, LpBinary)
PC_MO = pulp.LpVariable("PC_MO", 0, 1, LpBinary)
PC_JT = pulp.LpVariable("PC_JT", 0, 1, LpBinary)
AT_AC = pulp.LpVariable("AT_AC", 0, 1, LpBinary)

# Logic encoded: if visiting either of the places, then joint variable should be 2.
# If visiting neither of the places, then joint variables should be 0.
# Thus, twice the joint variable is the sum of each places.

# If MO + JT is 0, ML_CP must be 0. If ML + CP is 2, ML_CP must be 1.
preference1 = [
    MO + JT == 2 * MO_JT, 
    CP + CN == 2 * CP_CN, 
    ML + SC == 2 * ML_SC, 
    SC + CP == 2 * SC_CP, 
    CN + SC == 2 * CN_SC, 
    PC + MO == 2 * PC_MO, 
    PC + JT == 2 * PC_JT, 
    AT + AC == 2 * AT_AC
    ]

# Adding constraint for this preference
for constraint in preference1:
    prob += constraint

# Solving the preference and saving the result
prob.solve(PULP_CBC_CMD(msg = False))
print("Status: ", LpStatus[prob.status])

ListVisit2_Preference1 = maximize_visit_paris(data, prob)
print("\nIs Preference1 same as ListVisit1: ", is_identical(ListVisit1, ListVisit2_Preference1))

print("\n#########################################")
print("Preference 2:-")
print("He absolutely wants to visit the Eiffel Tower (TE) and Catacombes (CA).\n")

prob = create_linear_problem("ListVisit2_Preference2")

# Logic encoded: Both TE and CA to be visited, hence, sum of variables will be 2.
preference2 = [TE + CA == 2]

# Adding constraint for this preference
for constraint in preference2:
    prob += constraint

prob.solve(PULP_CBC_CMD(msg=False))
print("Status: ", LpStatus[prob.status])

ListVisit2_Preference2 = maximize_visit_paris(data, prob)
print("\nIs Preference2 same as ListVisit1: ", is_identical(ListVisit1, ListVisit2_Preference2))

print("\n#########################################")
print("Preference 3:-")
print("If he visits Notre Dame Cathedral (CN) then he will not visit the Sainte Chapelle (SC).\n")

prob = create_linear_problem("ListVisit2_Preference3")

# Logic encoded: If CN visited, then SC shouldnot be visited, hence sum of both <= 1.
preference3 = [CN + SC <= 1]

# Adding constraint for this preference
for constraint in preference3:
    prob += constraint

prob.solve(PULP_CBC_CMD(msg=False))
print("Status: ", LpStatus[prob.status])

ListVisit2_Preference3 = maximize_visit_paris(data, prob)
print("\nIs Preference3 same as ListVisit1: ", is_identical(ListVisit1, ListVisit2_Preference3))

print("\n#########################################")
print("Preference 4:-")
print("He absolutely wants to visit Tour Montparnasse (TM).\n")

prob = create_linear_problem("ListVisit2_Preference4")

# Logic encoded: TM should be visited, hence the value of variable is 1.
preference4 = [TM == 1]

# Adding constraint for this preference
for constraint in preference4:
    prob += constraint

prob.solve(PULP_CBC_CMD(msg=False))
print("Status: ", LpStatus[prob.status])

ListVisit2_Preference4 = maximize_visit_paris(data, prob)
print("\nIs Preference4 same as ListVisit1: ", is_identical(ListVisit1, ListVisit2_Preference4))

print("\n#########################################")
print("Preference 5:-")
print("If he visits the Louvre (ML) Museum then he must visit the Pompidou Center (CP).\n")

prob = create_linear_problem("ListVisit2_Preference5")

# Logic encoded: if visiting either of the places, then joint variable should be 2.
# If visiting neither of the places, then joint variables should be 0.
# Thus, twice the joint variable is the sum of each places.

ML_CP = pulp.LpVariable("ML_CP", 0, 1, LpBinary)

preference5 = [ML + CP == 2 * ML_CP] 

# Adding constraint for this preference
for constraint in preference5:
    prob += constraint

prob.solve(PULP_CBC_CMD(msg=False))
print("Status: ", LpStatus[prob.status])

ListVisit2_Preference5 = maximize_visit_paris(data, prob)
print("\nIs Preference5 same as ListVisit1: ", is_identical(ListVisit1, ListVisit2_Preference5))

print("\n#########################################")
print("Preference 1 and Preference 2:-")

prob = create_linear_problem("ListVisit2_b")

# Adding constraint for this preference
for constraint in (preference1 + preference2):
    prob += constraint

prob.solve(PULP_CBC_CMD(msg=False))
print("Status: ", LpStatus[prob.status])

ListVisit2_b = maximize_visit_paris(data, prob)

print("\n#########################################")
print("Preference 1 and Preference 3:-")

prob = create_linear_problem("ListVisit2_c")

# Adding constraint for this preference
for constraint in (preference1 + preference3):
    prob += constraint

prob.solve(PULP_CBC_CMD(msg=False))
print("Status: ", LpStatus[prob.status])

ListVisit2_c = maximize_visit_paris(data, prob)

print("\n#########################################")
print("Preference 1 and Preference 4:-")

prob = create_linear_problem("ListVisit2_d")

# Adding constraint for this preference
for constraint in (preference1 + preference4):
    prob += constraint

prob.solve(PULP_CBC_CMD(msg=False))
print("Status: ", LpStatus[prob.status])

ListVisit2_d = maximize_visit_paris(data, prob)

print("\n#########################################")
print("Preference 2 and Preference 5:-")

prob = create_linear_problem("ListVisit2_e")

# Adding constraint for this preference
for constraint in (preference2 + preference5):
    prob += constraint

prob.solve(PULP_CBC_CMD(msg=False))
print("Status: ", LpStatus[prob.status])

ListVisit2_e = maximize_visit_paris(data, prob)

print("\n#########################################")
print("Preference 3 and Preference 4:-")

prob = create_linear_problem("ListVisit2_f")

# Adding constraint for this preference
for constraint in (preference3 + preference4):
    prob += constraint

prob.solve(PULP_CBC_CMD(msg=False))
print("Status: ", LpStatus[prob.status])

ListVisit2_f = maximize_visit_paris(data, prob)

print("\n#########################################")
print("Preference 4 and Preference 5:-")

prob = create_linear_problem("ListVisit2_g")

# Adding constraint for this preference
for constraint in (preference4 + preference5):
    prob += constraint

prob.solve(PULP_CBC_CMD(msg=False))
print("Status: ", LpStatus[prob.status])

ListVisit2_g = maximize_visit_paris(data, prob)

print("\n#########################################")
print("Preference 1, Preference 2 and Preference 4:-")

prob = create_linear_problem("ListVisit2_h")

# Adding constraint for this preference
for constraint in (preference1 + preference2 + preference4):
    prob += constraint

prob.solve(PULP_CBC_CMD(msg=False))
print("Status: ", LpStatus[prob.status])

ListVisit2_h = maximize_visit_paris(data, prob)

print("\n#########################################")
print("Preference 2, Preference 3 and Preference 5:-")

prob = create_linear_problem("ListVisit2_i")

# Adding constraint for this preference
for constraint in (preference2 + preference3 + preference5):
    prob += constraint

prob.solve(PULP_CBC_CMD(msg=False))
print("Status: ", LpStatus[prob.status])

ListVisit2_i = maximize_visit_paris(data, prob)

print("\n#########################################")
print("Preference 2, Preference 3, Preference 4 and Preference 5:-")

prob = create_linear_problem("ListVisit2_j")

# Adding constraint for this preference
for constraint in (preference2 + preference3 + preference4 + preference5):
    prob += constraint

prob.solve(PULP_CBC_CMD(msg=False))
print("Status: ", LpStatus[prob.status])

ListVisit2_j = maximize_visit_paris(data, prob)

print("\n#########################################")
print("Preference 1, Preference 2, Preference 4 and Preference 5:-")

prob = create_linear_problem("ListVisit2_k")

# Adding constraint for this preference
for constraint in (preference1 + preference2 + preference4 + preference5):
    prob += constraint

prob.solve(PULP_CBC_CMD(msg=False))
print("Status: ", LpStatus[prob.status])

ListVisit2_k = maximize_visit_paris(data, prob)

print("\n#########################################")
print("Preference 1, Preference 2, Preference 3, Preference 4 and Preference 5:-")

prob = create_linear_problem("ListVisit2_l")

# Adding constraint for this preference
for constraint in (preference1 + preference2 + preference3 + preference4 + preference5):
    prob += constraint

prob.solve(PULP_CBC_CMD(msg=False))
print("Status: ", LpStatus[prob.status])

ListVisit2_l = maximize_visit_paris(data, prob)

print("\n#########################################")
print("Checking for identical with ListVisit1:-\n")

for i, x in enumerate(list('bcdefghijkl')):
    print(f"Are 'ListVisit 1' and 'ListVisit2_{x}' same ? {is_identical(ListVisit1, [f'ListVisit2_{x}'])}\n")

print("\n#########################################")
print("Rankings:-\n")

# Extracting columns from data
# 1) Duration: extracted as it is
# 2) Price: extracted as it is
# 3) Appreciation: negated the value, as higher the better

duration_ranking = [values[0] for values in data.values()]
price_ranking = [values[1] for values in data.values()]
appreciation_ranking = [(-1)*values[2] for values in data.values()]

ranking_combinations = list(combinations(['duration', 'price', 'appreciation'], 2))

for pair in ranking_combinations:
  ranking1 = eval(f"{pair[0]}_ranking")
  ranking2 = eval(f"{pair[1]}_ranking")
  print(f"Kendall Tau between '{pair[0]}' and '{pair[1]}': {round(kendalltau(ranking1, ranking2).correlation, 3)}\n")
  print(f"Spearman between '{pair[0]}' and '{pair[1]}': {round(spearmanr(ranking1, ranking2).correlation, 3)}\n")
  print()