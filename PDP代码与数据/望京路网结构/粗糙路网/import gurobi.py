import pandas as pd
import numpy as np
import random 
from gurobipy import *
if __name__ == '__main__':
    data = pd.read_csv('wangjing-newdis-simplify.csv').values
    vehicleNum = int (input("the number of vehicles available"))
    N = 10
    requirement = random.choice(range (data.shape[0]),2*N+2) 
    departure = []
    destination= []
    window =[]
    start = requirement[0]
    end = requirement[2*N+1]
    e = random.random()
    for i in  range (1,N+1):
        departure.append[requirement[i]]
    for i in range (1,N+1):
        destination.append[requirement[N+i]]
    dis = np.zeros(shape=[2*N+2,2*N+2])
    for i in range(2*N+2):
        for j in range(2*N+2):
                dis[i, j] = data[requirement[i],requirement[j]]
    for i in range (1,2*N+1):
        window .append= 5*random.random()+2
    model = Model("PDPTW")
    z = model.addVars(2*N+2,2*N+2,vehicleNum,vtype =GRB.BINARY,name = "z")
    a = model.addVars(2*N+2,vtype = GRB.CONTINUOUS,name="a")
    obj=0 
    for i in range (2*N+2):
        for j in range (2*N+2):
            for k in range (vehicleNum):
                if i != j:
                    obj+= z[i,j,k]*dis[i,j] 
    for j in range (1,2*N+1):
        if a[j]> window [j]:
            obj += e*(a[j]-window [j])
    for i in range(1, N+1):
        expr1 = LinExpr(0)
        for j in range(data.shape[0]):
            for k in range(vehicleNum):
                if i != j:
                    expr1.addTerms(1, z[i, j, k])
        model.addConstr(expr1 == 1, name='cons1')
    for i in range (1,N+1):
        expr2= LinExpr(0)
        for k in range (vehicleNum):
            for j in range(1,2*N+1):
                if i !=j:
                    expr2.addTerms(1,z[i,j,k]-z[N+i,j,k])
            expr2. addTerms(1,z[i,2*N+1,k]-z[i,2*N+1,k])
        model.addConstr(expr2 == 0, name ="cons2" )
    for k in range (vehicleNum):
        expr3= LinExpr(0)
        for j in range (1,2*N+2):
            expr3.addTerms(1,z[0,j,k])
        model.addConstr(expr3==1, name ="cons3")
    for i in range(1,2*N+1):
        for k in  range (vehicleNum):
            expr4 =LinExpr(0)
            for j in range(2*N+2):
                    expr4.addTerms(1,z[i,j,k]-z[j,i,k])
                    model.addConstr(expr4==0, name ="cons4")
    for k in range (vehicleNum):
        expr5= LinExpr(0)
        for i in range (2*N+1):
            expr3.addTerms(1,z[i,2*N+1,k])
        model.addConstr(expr5==1, name ="cons5")
    for i in range (2*N+2):
        for j in range(2+N+2):
            for k in range (vehicleNum):
                if  i !=j:
                    model.addConstr(a[j]+(1-z[i,j,k])*1000>=(a[i]+dis[i,j])*z[i,j,k],name='cons6' + '_' + str(i) + '_' + str(j) + '_' + str(k))
    for i  in range (1,N+1):
        model.addConstr(a[N+i]>=a[i]+dis[i,N+i],name="constr7")
    
    model.setObjective(obj),GRB.MINIMIZE
    model.write("ans.lp")
    model.optimize
    for key in z.keys():
        if(z[key].z > 0):
            print(z[key].Varname + ' =', z[key].z)