#!/usr/bin/env python
# coding: utf-8

# In this notebook we going to read our data and create new dataset for training

# In[1]:


import os
import pandas as pd
import numpy as np
import random


# In[2]:


pathf = os.path.dirname(os.path.dirname(os.path.abspath(__name__)))


# In[3]:


pathd=os.path.join(pathf ,'Datasets')


# In[4]:


fname = open(os.path.join(pathd,'fnames.txt')).read()
fname = fname.split("\n")
print(random.sample(fname,20),'\n',len(fname))


# In[5]:


mname = open(os.path.join(pathd,'mnames.txt')).read()
mname = mname.split("\n")
print(random.sample(mname,20),'\n',len(mname))


# In[6]:


allname = pd.read_csv(os.path.join(pathd,'Arabic_names.csv'))


# In[7]:


allname.sample(10)


# In[8]:


allname[allname['Gender'] == 'M'].sample(10)


# In[9]:


allname['Gender'].value_counts()


# In[10]:


male = allname[allname['Gender'] == 'M']


# In[11]:


def real_names(samples):
    """This function generate real names and the result in Nname.csv"""
    if not os.path.exists(os.path.join(pathd,'Nname.csv')):
        f = open(os.path.join(pathd,'Nname.csv'), "w")
        f.write('Name'+','+'Real'+'\n')
        f.close()
    f = open(os.path.join(pathd,'Nname.csv'), "a")
    for i in range(samples):
        result = ' '.join(random.choice(allname['Name'].values ) for _ in range(1))
        result1= ' '.join(random.choice(allname[allname['Gender'] == 'M']['Name'].values ) for _ in range(2))
        
        f.write(result+' '+result1+','+'1'+'\n')
    f.close()

real_names(30000)


# In[12]:


class Name_modification():
    ar_chars = list(' أبجد هوز حطي كلمن سعفص قرشت ثخذ ضظغ'.replace(" ", ""))
    def add_char(self,name):
        """Repeat the last char in the given name ex:- احمد -> احمدد"""
        self.name = list(name)
        self.name.append(self.name[-1])
        return ''.join(self.name)
    
    def remove_char(self,name):
        """Remove the middle char in the given name ex:- السيد -> اليد"""
        self.name = list(name)
        mid = len(self.name)//2
        self.name.pop(mid)
        return ''.join(self.name)
    
    def change_char(self,name):
        """Change the middle char with random arabic char ex:- محمود -> محهود"""
        self.name = list(name)
        mid = len(self.name)//2
        self.name[mid] = random.sample(self.ar_chars,1)[0]
        return ''.join(self.name)
        


# In[13]:


def fake_names(samples):
    """This function generate fake names and the result in Nname.csv"""
    faker = Name_modification()
    operation =[faker.add_char,faker.remove_char,faker.change_char] #Contain all Name_modification operations
    if not os.path.exists(os.path.join(pathd,'Nname.csv')):
        f = open(os.path.join(pathd,'Nname.csv'), "w")
        f.write('Name'+','+'Real'+'\n')
        f.close()
    f = open(os.path.join(pathd,'Nname.csv'), "a")
    for i in range(samples):
        result = random.choices(allname['Name'].values ,k = 3) 
        
        output = [op(result[ind]) for ind,op in enumerate(random.choices(operation, k=3)) ]
        
       
        f.write(' '.join(output)+','+'0'+'\n')
    f.close()
fake_names(30000)


# In[14]:


new_name = pd.read_csv(os.path.join(pathd,'Nname.csv'))
new_name.sample(10)


# In[16]:


new_name['Real'].value_counts()


# In[ ]:




