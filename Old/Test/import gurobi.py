import pandas as pd
import os
os.chdir('/Users/apple/Desktop/Study/Traffic/Test')
import numpy as np
import random 
from gurobipy import *
if __name__ == '__main__':
    data = pd.read_csv('wangjing-newdis-simplify.csv').values
    vehicleNum =3
    #int (input("the number of vehicles available"))
    N = 10
    #requirement = random.sample(range (data.shape[0]),2*N+2) 
    requirement = [31, 16, 25, 44, 61, 56, 58, 62, 8, 71, 29, 24, 69, 23, 27, 52, 0, 21, 26, 20, 34, 35]
    demand= [10, 20, 30, 20, 30, 30, 30, 30, 20, 10, 10, 20, 30, 20, 30, 30, 30, 30, 20, 10]
    e=3.1415926
    departure = []
    destination= []
    window=[1000000,1000000, 1000000, 1000000, 1000000, 1000000, 1000000, 1000000, 1000000, 1000000, 1000000, 4.836524177878462, 6.477717374902587, 4.346499155632049, 6.030109860947896, 5.314449151747034, 6.894066036835748, 6.881163582682019, 5.261821795271478, 3.30478847960814, 6.891180500033741,1000000]
    start = requirement[0]
    end = requirement[2*N+1]
    for i in  range (1,N+1):
        departure.append(requirement[i])
    for i in range (1,N+1):
        destination.append(requirement[N+i])
    dis = np.zeros(shape=[2*N+2,2*N+2])
    for i in range(2*N+2):
        for j in range(2*N+2):
                dis[i, j] = data[requirement[i],requirement[j]+1]

    model = Model()
    z = model.addVars(2*N+2,2*N+2,vehicleNum,vtype =GRB.BINARY,name = "z")
    a = model.addVars(2*N+2,vtype = GRB.CONTINUOUS,name="a")
    y = model.addVars(2*N+2,vtype = GRB.CONTINUOUS,name="y")
    Q = model.addVars(2*N+2,vtype=GRB.CONTINUOUS,name="q" )

    obj=LinExpr(0) 
    for i in range (2*N+2):
        for j in range (2*N+2):
            for k in range (vehicleNum):
                if i != j:
                    obj+= z[i,j,k]*dis[i,j] 
    for j in range (1,2*N+1):
        obj += e*y[j]
    for i in range(1,2*N+1):
        expr1 = LinExpr(0)
        for j in range(2*N+2):
            for k in range(vehicleNum):
                if i != j:
                    expr1.addTerms(1, z[i, j, k])
        model.addConstr(expr1 == 1, name='cons1')
    for i in range (1,N+1):
        for k in range (vehicleNum):
            expr2= LinExpr(0)
            for j in range(1,2*N+1):
                if i !=j:
                    expr2.addTerms(1,z[i,j,k])
                    expr2.addTerms(-1,z[N+i,j,k])
            expr2. addTerms(1,z[i,2*N+1,k])
            expr2.addTerms(-1,z[i,2*N+1,k])
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
                    expr4.addTerms(1,z[i,j,k])
                    expr4.addTerms(-1,z[j,i,k])
            model.addConstr(expr4==0, name ="cons4")
    for k in range (vehicleNum):
        expr5= LinExpr(0)
        for i in range (2*N+1):
            expr5.addTerms(1,z[i,2*N+1,k])
        model.addConstr(expr5==1, name ="cons5")
    for i in range (2*N+2):
        for j in range(2+N+2):
            for k in range (vehicleNum):
                if  i !=j:
                    model.addConstr(a[j]+(1-z[i,j,k])*1000>=(a[i]+dis[i,j])*z[i,j,k],name='cons6' + '_' + str(i) + '_' + str(j) + '_' + str(k))
    for i  in range (1,N+1):
        model.addConstr(a[N+i]>=a[i]+dis[i,N+i],name="cons7")
    for j in range(1,2*N+1):
        model.addConstr(y[j]>=0,name="cons8")
    for j in range (1,2*N+1):
        model.addConstr(a[j]>=0)
    model.addConstr(a[0]==0)
    for j in range(N+1,2*N+1):
        model.addConstr(y[j]>=a[j]-window[j],name="constr9")
    for k in range (vehicleNum):
        for i in range(2*N+2):
            for j in range (1,N+1):
                if  i != j :
                    model.addConstr((z[i,j,k]==1)>>(Q[i]+demand[j-1]==Q[j]))
    for k in range (vehicleNum):
        for i in range(2*N+2):
            for j in range (N+1,2*N+1):
                if  i != j :
                    model.addConstr((z[i,j,k]==1)>>(Q[i]-demand[j-1]==Q[j]))
    for k in range (vehicleNum):
            for j in range (1,N+1):
                model.addConstr((z[0,j,k]==1)>>(Q[j]==Q[0]+demand[j-1]))
                model.addConstr(Q[0]==0)
    for j in range (2*N+2):
        model.addConstr(Q[j]<=100)           
    model.setObjective(obj),GRB.MINIMIZE
    model.optimize()
