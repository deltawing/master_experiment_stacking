# -*- coding: utf-8 -*-
"""
Created on Mon Apr 17 05:59:43 2017

@author: 罗骏
"""
import csv
import os
import math
import pywt
import numpy as np
from mlxtend.classifier import StackingClassifier
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier,ExtraTreesClassifier,AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.svm import SVC
S=0
opposite_path=os.path.abspath('')
def transfer_feature(inputlist):
    feature_hash={'A':[],'C':[],'D':[],'E':[],'F':[],'G':[],'H':[],'I':[],
    'K':[],'L':[],'M':[],'N':[],'P':[],'Q':[],'R':[],'S':[],'T':[],'V':[],
    'W':[],'Y':[]}
    for row in inputlist:
        feature_hash['A']+=[float(row[0])]
        feature_hash['C']+=[float(row[1])]
        feature_hash['D']+=[float(row[2])]
        feature_hash['E']+=[float(row[3])]
        feature_hash['F']+=[float(row[4])]
        feature_hash['G']+=[float(row[5])]
        feature_hash['H']+=[float(row[6])]
        feature_hash['I']+=[float(row[7])]
        feature_hash['K']+=[float(row[8])]
        feature_hash['L']+=[float(row[9])]
        feature_hash['M']+=[float(row[10])]
        feature_hash['N']+=[float(row[11])]
        feature_hash['P']+=[float(row[12])]
        feature_hash['Q']+=[float(row[13])]
        feature_hash['R']+=[float(row[14])]
        feature_hash['S']+=[float(row[15])]
        feature_hash['T']+=[float(row[16])]
        feature_hash['V']+=[float(row[17])]
        feature_hash['W']+=[float(row[18])]
        feature_hash['Y']+=[float(row[19])]
    return feature_hash   

def transfer_fasta(fasta):
    fasta_hash={}
    dip_number_row=[]
    protein_array_row=['']
    for row in fasta:
        row=row.strip(" \n")
        if (len(row)>0 and row.find('>')>=0):
            if (len(dip_number_row)>0 and len(protein_array_row[0])>0):
                fasta_hash[dip_number_row]=protein_array_row[0]
                protein_array_row=['']
            dip_number_row=row
        elif (len(row)>0 and row.find('>')<0):
            protein_array_row[0]+=row
    return fasta_hash     
    
def construct_array(dip_matrix,locate_feature,locate_fasta,label):
    temp_interact_hash={}
    #temp_interact_array_list=[]
    #transformed_array=[]
    #PseAAC_array=[]
    #AC_array=[]
    #LD_array=[]
    #CT_array=[]
    DWT_array=[]
    for row in dip_matrix:
        if (len(row)<=0 ):
            continue
        if (row[0].find('-')<0 or len(row[0])<=0 ):
            continue
        row=row[0].split('-')
        dip_proteinA,dip_proteinB=row[0],row[1]
        short_itemA=dip_proteinA[:]
        short_itemB=dip_proteinB[:]
        #print(short_itemA,short_itemB,'\t',dip_proteinA,dip_proteinB)
        if (short_itemA>short_itemB):
            short_itemA,short_itemB=short_itemB,short_itemA
        temp_interact_hash[str(short_itemA)+'M'+str(short_itemB)]=0
    #print(temp_interact_hash)
        
    for temp_interact_row in temp_interact_hash:
        interact_proteinA=temp_interact_row[0:temp_interact_row.find('M')]
        interact_proteinB=temp_interact_row[temp_interact_row.find('M')+1:]
        dip_proteinA=search_match(interact_proteinA,locate_fasta)
        dip_proteinB=search_match(interact_proteinB,locate_fasta)
        if (len(dip_proteinA)<64 or len(dip_proteinB)<64):
            continue      
        #print(dip_proteinA,'\t',dip_proteinB,'\n')
        #PseAAC_featured_proteinA=PseAAC(dip_proteinA,locate_feature)
        #PseAAC_featured_proteinB=PseAAC(dip_proteinB,locate_feature)
        #AC_featured_proteinA=Auto_Covariance(dip_proteinA,locate_feature)
        #AC_featured_proteinB=Auto_Covariance(dip_proteinB,locate_feature)
        #LD_featured_proteinA=Local_descriptors(dip_proteinA)
        #LD_featured_proteinB=Local_descriptors(dip_proteinB)
        #CT_featured_proteinA=Conjoint_triad(dip_proteinA)
        #CT_featured_proteinB=Conjoint_triad(dip_proteinA)
        DWT_featured_proteinA=Discrete_wavelet_transform(dip_proteinA,locate_feature)
        DWT_featured_proteinB=Discrete_wavelet_transform(dip_proteinB,locate_feature)
        #PseAAC_array+=[PseAAC_featured_proteinA+['PSE']+PseAAC_featured_proteinB]
        #AC_array+=[AC_featured_proteinA+['AC']+AC_featured_proteinB+['1']]
        #LD_array+=[LD_featured_proteinA+['LD']+LD_featured_proteinB]
        #CT_array+=[CT_featured_proteinA+['CT']+CT_featured_proteinB]
        DWT_array+=[DWT_featured_proteinA+['DWT']+DWT_featured_proteinB+[label]]
    return DWT_array 
            #AC_array ,PseAAC_array, LD_array, , CT_array, 
        #print(PseAAC_featured_proteinA,'\t',PseAAC_featured_proteinB,0)
        #transformed_array+=[PseAAC_featured_proteinA+['PSE']+PseAAC_featured_proteinB+['The other']+AC_featured_proteinA+['AC']+AC_featured_proteinB]
    #return transformed_array

def Discrete_wavelet_transform(protein_array,locate_feature):
    return_DWT_array=[]
    for q in range(7):
        test_DWT_array=[]
        for i in range(len(protein_array)):
            if (protein_array[i]=='X' or protein_array[i]=='U' or protein_array[i]==' '):
                continue
            test_DWT_array+=[locate_feature[protein_array[i]][q]]
        cA4, cD4, cD3, cD2, cD1 = pywt.wavedec(test_DWT_array, 'db2',level=4)
        cA4array=Take_Obvious_item(cA4)
        cD4array=Take_Obvious_item(cD4)
        cD3array=Take_Obvious_item(cD3)
        cD2array=Take_Obvious_item(cD2)
        cD1array=np.array(cD1).astype(np.float32)
        cD1array=[cD1array.mean(),cD1array.var()]
        return_DWT_array+=['The',q+1,'th']+cA4array+cD4array+cD3array+cD2array+cD1array
    return return_DWT_array

def Take_Obvious_item(X):
    temp_X=np.array(X).astype(np.float32)
    Q=np.array(X).astype(np.float32)
    Q=Q**2
    Y=[]
    count=4
    max_number=np.max(Q)**0.5
    me_mean=Q.mean()
    me_var=Q.var()
    for i in range(count):
        Y+=[temp_X[np.argmax(Q)]]
        Y+=[(np.argmax(Q)+1)*max_number/Q.size]
        temp_X[np.argmax(Q)]=0
        Q[np.argmax(Q)]=0  
    Y+=[me_mean,me_var]
    return Y
    
def Conjoint_triad(protein_array):
    local_operate_array=[]
    for i in range(len(protein_array)):
        if (protein_array[i]=='A' or protein_array[i]=='G' or protein_array[i]=='V'):
            local_operate_array+=[1]
        elif (protein_array[i]=='I' or protein_array[i]=='L' or protein_array[i]=='F' or protein_array[i]=='P'):
            local_operate_array+=[2]
        elif (protein_array[i]=='Y' or protein_array[i]=='M' or protein_array[i]=='T' or protein_array[i]=='S'):
            local_operate_array+=[3]
        elif (protein_array[i]=='H' or protein_array[i]=='N' or protein_array[i]=='Q' or protein_array[i]=='W'):
            local_operate_array+=[4]
        elif (protein_array[i]=='R' or protein_array[i]=='K'):
            local_operate_array+=[5]
        elif (protein_array[i]=='D' or protein_array[i]=='E'):
            local_operate_array+=[6]
        elif (protein_array[i]=='C'):
            local_operate_array+=[7]
        else :
            local_operate_array+=[7]
    #print(local_operate_array)
    vector_3_matrix=[]
    for a in range(7):
        for b in range(7):
            for c in range(7):
                vector_3_matrix+=[[a+1,b+1,c+1,0]]
    for m in range(len(local_operate_array)-2):
        vector_3_matrix[(local_operate_array[m]-1)*49+(local_operate_array[m+1]-1)*7+(local_operate_array[m+2]-1)][3]+=1
    CT_array=[]
    for q in range(343):
        CT_array+=[vector_3_matrix[q][3]]
    return CT_array
    
def Local_descriptors(protein_array):
    local_operate_array=[]
    for i in range(len(protein_array)):
        if (protein_array[i]=='A' or protein_array[i]=='G' or protein_array[i]=='V'):
            local_operate_array+=[1]
        elif (protein_array[i]=='I' or protein_array[i]=='L' or protein_array[i]=='F' or protein_array[i]=='P'):
            local_operate_array+=[2]
        elif (protein_array[i]=='Y' or protein_array[i]=='M' or protein_array[i]=='T' or protein_array[i]=='S'):
            local_operate_array+=[3]
        elif (protein_array[i]=='H' or protein_array[i]=='N' or protein_array[i]=='Q' or protein_array[i]=='W'):
            local_operate_array+=[4]
        elif (protein_array[i]=='R' or protein_array[i]=='K'):
            local_operate_array+=[5]
        elif (protein_array[i]=='D' or protein_array[i]=='E'):
            local_operate_array+=[6]
        elif (protein_array[i]=='C'):
            local_operate_array+=[7]
        else :
            local_operate_array+=[7]
    #print(local_operate_array)
    A_point=math.floor(len(protein_array)/4-1)
    B_point=math.floor(len(protein_array)/2-1)
    C_point=math.floor(len(protein_array)/4*3-1)
    part_vector=[]
    part_vector+=Construct_63_vector(local_operate_array[0:A_point])
    part_vector+=Construct_63_vector(local_operate_array[A_point:B_point])
    part_vector+=Construct_63_vector(local_operate_array[B_point:C_point])
    part_vector+=Construct_63_vector(local_operate_array[C_point:-1])
    part_vector+=Construct_63_vector(local_operate_array[0:B_point])
    part_vector+=Construct_63_vector(local_operate_array[B_point:-1])
    part_vector+=Construct_63_vector(local_operate_array[A_point:C_point])
    part_vector+=Construct_63_vector(local_operate_array[0:C_point])
    part_vector+=Construct_63_vector(local_operate_array[A_point:-1])
    part_vector+=Construct_63_vector(local_operate_array[math.floor(A_point/2):math.floor(C_point/2)])
    #print (part_vector) 
    return part_vector
    
def Construct_63_vector(part_array):
    simple_7=[0 for i in range(7)]
    marix_7_7=[[0 for i in range(7)] for j in range(7)]
    simple_21=[0 for i in range(21)]
    simple_35=[0 for i in range(35)]
    for i in range(len(part_array)):
        simple_7[part_array[i]-1]+=1
        if (i<(len(part_array)-1) and part_array[i]!=part_array[i+1]):
            if(part_array[i]>part_array[i+1]):
                j,k=part_array[i+1],part_array[i]
            else:
                j,k=part_array[i],part_array[i+1]
            marix_7_7[j-1][k-1]+=1
    i=0
    for j in range(7):
        for k in range(j+1,7):
            simple_21[i]=marix_7_7[j][k]
            i+=1
    residue_count=[0,0,0,0,0,0,0]
    for q in range(len(part_array)):
        residue_count[part_array[q]-1]+=1
        if (residue_count[part_array[q]-1]==1):
            simple_35[5*(part_array[q]-1)]=q+1
        elif(residue_count[part_array[q]-1]==math.floor(simple_7[part_array[q]-1]/4)):
            simple_35[5*(part_array[q]-1)+1]=q+1
        elif(residue_count[part_array[q]-1]==math.floor(simple_7[part_array[q]-1]/2)):
            simple_35[5*(part_array[q]-1)+2]=q+1
        elif(residue_count[part_array[q]-1]==math.floor(simple_7[part_array[q]-1]*0.75)):
            simple_35[5*(part_array[q]-1)+3]=q+1
        elif(residue_count[part_array[q]-1]==simple_7[part_array[q]-1]):
            simple_35[5*(part_array[q]-1)+4]=q+1
    for o in range(7):
        simple_7[o]/=len(part_array)
    for p in range(21):       
        simple_21[p]/=len(part_array)
    for m in range(35):       
        simple_35[m]/=len(part_array)
    simple_63_vector=simple_7+simple_21+simple_35
    return simple_63_vector
    
def PseAAC(protein_array,locate_feature):
    nambda=15   #
    omega=0.05  #
    AA_frequency={'A':[0],'C':[0],'D':[0],'E':[0],'F':[0],'G':[0],'H':[0],'I':[0],
    'K':[0],'L':[0],'M':[0],'N':[0],'P':[0],'Q':[0],'R':[0],'S':[0],'T':[0],'V':[0],
    'W':[0],'Y':[0]}
    A_class_feature=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    B_class_feature=[]
    sum_frequency=0
    sum_occurrence_frequency=0
    for i in range(len(protein_array)):
        if (protein_array[i]=='X' or protein_array[i]=='U'):
            continue
        AA_frequency[protein_array[i]][0]+=1
    for j in AA_frequency:
        sum_frequency+=AA_frequency[j][0]
    for m in AA_frequency:
        if (sum_frequency==0):
            #print(protein_array)
            s=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
            return s
        else:    
            AA_frequency[m][0]/=sum_frequency
    for o in AA_frequency:
        sum_occurrence_frequency+=AA_frequency[o][0]
    
    for k in range(nambda):
        B_class_feature+=[THET(protein_array,locate_feature,k+1)]
    Pu_under=sum_occurrence_frequency+omega*sum(B_class_feature)
    for l in range(nambda):
        B_class_feature[l]=(B_class_feature[l]*omega/Pu_under)*100
    
    A_class_feature[0]=AA_frequency['A'][0]/Pu_under*100
    A_class_feature[1]=AA_frequency['C'][0]/Pu_under*100
    A_class_feature[2]=AA_frequency['D'][0]/Pu_under*100
    A_class_feature[3]=AA_frequency['E'][0]/Pu_under*100
    A_class_feature[4]=AA_frequency['F'][0]/Pu_under*100
    A_class_feature[5]=AA_frequency['G'][0]/Pu_under*100
    A_class_feature[6]=AA_frequency['H'][0]/Pu_under*100
    A_class_feature[7]=AA_frequency['I'][0]/Pu_under*100
    A_class_feature[8]=AA_frequency['K'][0]/Pu_under*100
    A_class_feature[9]=AA_frequency['L'][0]/Pu_under*100
    A_class_feature[10]=AA_frequency['M'][0]/Pu_under*100
    A_class_feature[11]=AA_frequency['N'][0]/Pu_under*100
    A_class_feature[12]=AA_frequency['P'][0]/Pu_under*100
    A_class_feature[13]=AA_frequency['Q'][0]/Pu_under*100
    A_class_feature[14]=AA_frequency['R'][0]/Pu_under*100
    A_class_feature[15]=AA_frequency['S'][0]/Pu_under*100
    A_class_feature[16]=AA_frequency['T'][0]/Pu_under*100
    A_class_feature[17]=AA_frequency['V'][0]/Pu_under*100
    A_class_feature[18]=AA_frequency['W'][0]/Pu_under*100
    A_class_feature[19]=AA_frequency['Y'][0]/Pu_under*100
    class_feature=A_class_feature+B_class_feature
    return class_feature

def Auto_Covariance(protein_array,locate_feature):
    lg=30 #will affect 'ac_array' down below
    AC_array=[[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
              [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
              [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
              [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
              [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
              [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
              [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]]
    mean_feature=[0,0,0,0,0,0,0]
    for j in range(len(mean_feature)):
        for i in range(len(protein_array)):
            if (protein_array[i]=='X' or protein_array[i]=='U' or protein_array[i]==' '):
                continue
            mean_feature[j]+=locate_feature[protein_array[i]][j]
    for k in range(len(mean_feature)):
        mean_feature[k]/=len(protein_array)
    for lag in range(lg):
        for ac_fea in range(len(mean_feature)):
            AC_array[ac_fea][lag]=AcSUM(protein_array,lag,mean_feature,ac_fea) 
    Auto_Covariance_feature=[]
    for o in range(len(AC_array)):
        for p in range(len(AC_array[0])):
            Auto_Covariance_feature+=[AC_array[o][p]]
    return Auto_Covariance_feature

def AcSUM(protein_array,lag,mean_feature,ac_fea):
    phychem_sum=0
    for i in range (len(protein_array)-lag):
        if(protein_array[i]=='X' or protein_array[i+lag]=='X' or protein_array[i]=='U' or protein_array[i+lag]=='U' or protein_array[i]==' ' or protein_array[i+lag]==' '):
            continue
        phychem_sum+=(locate_feature[protein_array[i]][ac_fea]-mean_feature[ac_fea])*(locate_feature[protein_array[i+lag]][ac_fea]-mean_feature[ac_fea])
    phychem_sum/=(len(protein_array)-lag)
    return phychem_sum
    
def THET(protein_array,locate_feature,t):
    sum_COMP=0
    for i in range(len(protein_array)-t):
        sum_COMP+=COMP(protein_array[i],protein_array[i+t],locate_feature)
    sum_COMP/=(len(protein_array)-t)
    return sum_COMP
    
def COMP(Ri,Rj,locate_feature):
    theth=0
    if (Ri=='X' or Rj=='X' or Ri=='U' or Rj=='U'):
        return 0
    else:
        theth+=pow(locate_feature[Ri][0]-locate_feature[Rj][0],2)
        theth+=pow(locate_feature[Ri][1]-locate_feature[Rj][1],2)
        theth+=pow(locate_feature[Ri][2]-locate_feature[Rj][2],2)
        theth=theth/3
        return theth
    
def search_match(protein_number,locate_fasta):
    target_protein=''
    #print('DIP-'+protein_number+'N|')
    for protein_array in locate_fasta:  
        if (protein_array==('>'+protein_number)):
            target_protein+=locate_fasta[protein_array]
            break
    return target_protein
        
with open(opposite_path+"/normalized_feature.csv") as C:
    normalized_feature=csv.reader(C)
    locate_feature=transfer_feature(normalized_feature)
with open(opposite_path+"/fasta.txt") as B:
    locate_fasta=transfer_fasta(B)
with open(opposite_path+"/Positive_H.pylori.csv") as A:
    dip_matrix=csv.reader(A)
    #PseAAC_feature_rows, AC_feature_rows, LD_feature_rows, CT_feature_rows,
    DWT_Positive_feature_rows=construct_array(dip_matrix,locate_feature,locate_fasta,'1')
with open(opposite_path+"/Negative_H.pylori.csv") as A:
    dip_matrix=csv.reader(A)
    DWT_Negative_feature_rows=construct_array(dip_matrix,locate_feature,locate_fasta,'0')
    DWT_feature=DWT_Negative_feature_rows+DWT_Positive_feature_rows
    #with open ("C:/Strawberry/fortestdip/iPPI-Esml_method/features/PSE_Negative_S.cerevisiae.csv",'w') as W1: 
        #featured_array1=csv.writer(W1)
        #featured_array1.writerows(PseAAC_feature_rows)
    #with open ("D:/dip_protein/iPPI-Esml_method/features/AC_Positive_S.cerevisiae.csv",'w') as W2: 
        #featured_array2=csv.writer(W2)
        #featured_array2.writerows(AC_feature_rows)
    #with open ("C:/Strawberry/fortestdip/iPPI-Esml_method/features/LD_Negative_S.cerevisiae.csv",'w') as W3: 
        #featured_array3=csv.writer(W3)
        #featured_array3.writerows(LD_feature_rows)
    #with open ("C:/Strawberry/fortestdip/iPPI-Esml_method/features/CT_Negative_S.cerevisiae.csv",'w') as W4: 
        #featured_array4=csv.writer(W4)
        #featured_array4.writerows(CT_feature_rows)
    #with open ("D:/dip_protein/iPPI-Esml_method/features/AC_Positive_H.pylori.csv",'w',newline='') as W5: 
        #featured_array5=csv.writer(W5)
        #featured_array5.writerows(DWT_feature_rows)
    S=len(DWT_feature)
    row_number=-1
    for rows in DWT_feature:
        row_number+=1 
        if (row_number==0):
            data=np.zeros((S,len(rows[3:45]+rows[48:90]+rows[93:135]+rows[138:180]+rows[183:225]+rows[228:270]+rows[273:315]+rows[319:361]+rows[364:406]+rows[409:451]+rows[454:496]+rows[499:541]+rows[544:586]+rows[589:631]+[rows[-1]])))
            data[row_number,:]=np.array([rows[3:45]+rows[48:90]+rows[93:135]+rows[138:180]+rows[183:225]+rows[228:270]+rows[273:315]+rows[319:361]+rows[364:406]+rows[409:451]+rows[454:496]+rows[499:541]+rows[544:586]+rows[589:631]+[rows[-1]]]).astype(np.float32)
            continue
        data[row_number,:]=np.array([rows[3:45]+rows[48:90]+rows[93:135]+rows[138:180]+rows[183:225]+rows[228:270]+rows[273:315]+rows[319:361]+rows[364:406]+rows[409:451]+rows[454:496]+rows[499:541]+rows[544:586]+rows[589:631]+[rows[-1]]])
featureList=data[:,0:-1]
labelList=data[:,-1] 
if __name__=='__main__':   
    clf1=GradientBoostingClassifier(learning_rate=0.3,n_estimators=150,max_depth=7)
    clf2=ExtraTreesClassifier(n_estimators=200)
    clf3=KNeighborsClassifier(weights='distance')
    clf4=QuadraticDiscriminantAnalysis()
    clf5=RandomForestClassifier(criterion='entropy',n_estimators=100, random_state=1)
    clf6=SVC()
    lr = LogisticRegression()
    sclf = StackingClassifier(classifiers=[clf1, clf2, clf3, clf4], meta_classifier=lr)
    predicted = cross_val_predict(sclf,featureList,labelList,cv=5,n_jobs=2 )
    #predicted = cross_val_predict(clf3,featureList,labelList,cv=5,n_jobs=2 )
    score1 = metrics.confusion_matrix(labelList,predicted)
    score2 = metrics.f1_score(labelList,predicted)
    score3 = metrics.auc(labelList,predicted)
    score4 = metrics.matthews_corrcoef(labelList,predicted)
    print ("KNeighborsClassifier:\n",score1,"\n",np.average(score2),"\n",np.average(score3),"\n",np.average(score4),"\n")