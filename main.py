import json
import numpy as np
from flask import Flask, jsonify, request

# Find out MwhPrice for each powerplant and sort them from cheapest to the more expensive.
# Find out the pmin and pmax of each windturbines.
# Count how many powerplants are needed to cover the load.
def createMeritOrderedList(payloadString):
    payload = payloadString
    
    load = payload["load"]
    fuels = payload["fuels"]
    powerplants = payload["powerplants"]

    for powerplant in powerplants:

        if powerplant["type"] == "gasfired":
            powerplant["MwhPrice"] = round(fuels["gas(euro/MWh)"] / powerplant["efficiency"], 3)

        elif powerplant["type"] == "turbojet":
            powerplant["MwhPrice"] = round(fuels["kerosine(euro/MWh)"] / powerplant["efficiency"], 3)

        elif powerplant["type"] == "windturbine":
            powerplant["MwhPrice"] = 0
            powerplant["pmin"] = powerplant["pmin"] * fuels["wind(%)"]
            powerplant["pmax"] = powerplant["pmax"] * fuels["wind(%)"]

        else:
            print("Issue in powerplant type selection.")
            print(powerplant["type"])

    meritOrderedList = sorted(powerplants, key = lambda i: i['MwhPrice'])
    cumpmin = 0
    cumpmax = 0
    countNeededPlant = 1
    
    for j in meritOrderedList:
        j["cumPmin"] = cumpmin + j["pmin"]
        cumpmin = j["cumPmin"]
        j["cumPmax"] = cumpmax + j["pmax"]
        cumpmax = j["cumPmax"]
        if not(load > j["cumPmin"]) & (load < j["cumPmax"]):
            countNeededPlant += 1
    
    return load, meritOrderedList, countNeededPlant
  
  
# Generate a numpy array with enough rows for each constraint 
# plus the objective function and enough columns for the variables, 
# slack variables, M (max/min) and the corresponding value.
def gen_matrix(var,cons):    
    tab = np.zeros((cons+1, var+cons+2))    
    return tab


# Check to see if 1+ pivots are required due to a negative element 
# in the furthest right column, excluding the bottom value, of course.
def next_round_r(table):    
    m = min(table[:-1,-1])    
    if m>= 0:        
        return False    
    else:        
        return True

# Check to see if 1+ pivots are required due to a negative element in the bottom row, 
# excluding the final value.
def next_round(table):    
    lr = len(table[:,0])   
    m = min(table[lr-1,:-1])    
    if m>=0:
        return False
    else:
        return True

# Find negative elements in the furthest right column.
def find_neg_r(table):
    lc = len(table[0,:])
    m = min(table[:-1,lc-1])
    if m<=0:        
        n = np.where(table[:-1,lc-1] == m)[0][0]
    else:
        n = None
    return n

# Find negative elements in the bottom row
def find_neg(table):
    lr = len(table[:,0])
    m = min(table[lr-1,:-1])
    if m<=0:
        n = np.where(table[lr-1,:-1] == m)[0][0]
    else:
        n = None
    return n

#  Identify the column and row indexes, respectively, for negative elements in the last column, last row. 
#  Find the pivot element corresponding to these values.
def loc_piv_r(table):
    total = []        
    r = find_neg_r(table)
    row = table[r,:-1]
    m = min(row)
    c = np.where(row == m)[0][0]
    col = table[:-1,c]
    for i, b in zip(col,table[:-1,-1]):
        if i**2>0 and b/i>0:
            total.append(b/i)
        else:                
            total.append(10000)
    index = total.index(min(total))        
    return [index,c]

# Find a pivot element corresponding to a negative element in the bottom row.
def loc_piv(table):
    if next_round(table):
        total = []
        n = find_neg(table)
        for i,b in zip(table[:-1,n],table[:-1,-1]):
            if b/i >0 and i**2>0:
                total.append(b/i)
            else:
                total.append(10000)
        index = total.index(min(total))
        return [index,n]

# Pivot about an element to remove the negative entry in the final column or row and return the updated table.
def pivot(row,col,table):
    lr = len(table[:,0])
    lc = len(table[0,:])
    t = np.zeros((lr,lc))
    pr = table[row,:]
    if table[row,col]**2>0:
        e = 1/table[row,col]
        r = pr*e
        for i in range(len(table[:,col])):
            k = table[i,:]
            c = table[i,col]
            if list(k) == list(pr):
                continue
            else:
                t[i,:] = list(k-r*c)
        t[row,:] = list(r)
        return t
    else:
        print('Cannot pivot on this element.')

# Convert user inputs into float variables. Function will receive inputs such as (‘1,3,L,5’); 
# this means 1(x1) + 3(x2) ≤ 5. Alternatively, ‘G’ could be used to mean a ≥ inequality.
def convert(eq):
    eq = eq.split(',')
    if 'G' in eq:
        g = eq.index('G')
        del eq[g]
        eq = [float(i)*-1 for i in eq]
        return eq
    if 'L' in eq:
        l = eq.index('L')
        del eq[l]
        eq = [float(i) for i in eq]
        return eq

# Convert maximization problem into minimization problem
def convert_min(table):
    table[-1,:-2] = [-1*i for i in table[-1,:-2]]
    table[-1,-1] = -1*table[-1,-1]    
    return table

# Generate only the required number of variables x1, x2,…xn.
def gen_var(table):
    lc = len(table[0,:])
    lr = len(table[:,0])
    var = lc - lr -1
    v = []
    for i in range(var):
        v.append('x'+str(i+1))
    return v

# Check if 1+ constraints can be added to the matrix, meaning there are at least two rows of all 0 elements.
def add_cons(table):
    lr = len(table[:,0])
    empty = []
    for i in range(lr):
        total = 0
        for j in table[i,:]:                       
            total += j**2
        if total == 0: 
            empty.append(total)
    if len(empty)>1:
        return True
    else:
        return False

# Add the constraints to the problem.
def constrain(table,eq):
    if add_cons(table) == True:
        lc = len(table[0,:])
        lr = len(table[:,0])
        var = lc - lr -1      
        j = 0
        while j < lr:            
            row_check = table[j,:]
            total = 0
            for i in row_check:
                total += float(i**2)
            if total == 0:                
                row = row_check
                break
            j +=1
        eq = convert(eq)
        i = 0
        while i<len(eq)-1:
            row[i] = eq[i]
            i +=1        
        row[-1] = eq[-1]
        row[var+j] = 1    
    else:
        print('Cannot add another constraint.')

# Check if the objective function can be added.
def add_obj(table):
    lr = len(table[:,0])
    empty = []
    for i in range(lr):
        total = 0        
        for j in table[i,:]:
            total += j**2
        if total == 0:
            empty.append(total)    
    if len(empty)==1:
        return True
    else:
        return False

# Add the objective function to the tableau, given that it satisfies add_obj().
def obj(table,eq):
    if add_obj(table)==True:
        eq = [float(i) for i in eq.split(',')]
        lr = len(table[:,0])
        row = table[lr-1,:]
        i = 0        
        while i<len(eq)-1:
            row[i] = eq[i]*-1
            i +=1
        row[-2] = 1
        row[-1] = eq[-1]
    else:
        print('You must finish adding constraints before the objective function can be added.')

# Definition of Maximization function
def maxz(table):
    while next_round_r(table)==True:
        table = pivot(loc_piv_r(table)[0],loc_piv_r(table)[1],table)
    while next_round(table)==True:
        table = pivot(loc_piv(table)[0],loc_piv(table)[1],table)        
    lc = len(table[0,:])
    lr = len(table[:,0])
    var = lc - lr -1
    i = 0
    val = {}
    for i in range(var):
        col = table[:,i]
        s = sum(col)
        m = max(col)
        if float(s) == float(m):
            loc = np.where(col == m)[0][0]            
            val[gen_var(table)[i]] = table[loc,-1]
        else:
            val[gen_var(table)[i]] = 0
    val['max'] = table[-1,-1]
    return val

# Definition of Minimization function
def minz(table):
    table = convert_min(table)
    while next_round_r(table)==True:
        table = pivot(loc_piv_r(table)[0],loc_piv_r(table)[1],table)    
    while next_round(table)==True:
        table = pivot(loc_piv(table)[0],loc_piv(table)[1],table)       
    lc = len(table[0,:])
    lr = len(table[:,0])
    var = lc - lr -1
    i = 0
    val = {}
    for i in range(var):
        col = table[:,i]
        s = sum(col)
        m = max(col)
        if float(s) == float(m):
            loc = np.where(col == m)[0][0]             
            val[gen_var(table)[i]] = table[loc,-1]
        else:
            val[gen_var(table)[i]] = 0 
            val['min'] = table[-1,-1]*-1
    return val
  
  
# REST API
app = Flask(__name__)
@app.route('/productionplan', methods=['POST'])
def prodPlan():
    payload = request.json
    load, meritOrderedList, countNeededPlant = createMeritOrderedList(payload)
    
    # Simplex Algorithm

    # gen_matrix(nbr_var, nbr_constraints)
    # nbr_var = nbr de powerplants = len(powerplants)
    # nbr_constraints = load_constraint + pmin_constaint + pmax_constraint = 2 + len(powerplants)*2
    m = gen_matrix(countNeededPlant, 2 + countNeededPlant*2)

    # 2 load constraints
    c = ['1,' for a in range(countNeededPlant)]
    k = ''.join(c)
    constrain(m, k + 'G,' + str(load))
    constrain(m, k + 'L,' + str(load))

    # pmin constrain and pmax constrain
    for i in range(countNeededPlant):
        c = ['0' for a in range(countNeededPlant)]
        c[i] = '1'
        k = ','.join(c)
        constrain(m, k + ',G,' + str(meritOrderedList[i]['pmin']))
        constrain(m, k + ',L,' + str(meritOrderedList[i]['pmax']))

    c = [str(meritOrderedList[a]['MwhPrice']) for a in range(countNeededPlant)]
    k = ','.join(c)
    obj(m, k +',0')

    # Create the Response.json
    r = minz(m)
    resp = []
    for i in range(countNeededPlant):
        resp.append({"name": meritOrderedList[i]["name"], "p": list(r.values())[i]})
    
    return jsonify(resp)

app.run()
