import random
import pandas as pd
import os
os.chdir('/Users/apple/Desktop/Study/Traffic/Test')
if __name__ == '__main__':
    data = pd.read_csv('wangjing-newdis-simplify.csv').values
    N=10
    requirement = random.sample(range (data.shape[0]),2*N+2) 
    window=[]
    for i in range (1,2*N+1):
        window.append( 5*random.random()+2)
    print (requirement)
    print(window)