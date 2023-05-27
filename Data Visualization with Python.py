#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[104]:


import matplotlib.pyplot as plt


# In[7]:


x = [1, 2, 3]
y = [1, 4, 9]
plt.plot(x, y)
plt.title("test plot")
plt.xlabel("x")
plt.ylabel("y")
plt.show()


# In[9]:


x = [1, 2, 3]
y = [1, 4, 9]
z = [10, 5, 0]
plt.plot(x, y)
plt.plot(x, z)
plt.title("test plot")
plt.xlabel("x")
plt.ylabel("y and z")
plt.legend(["this is y", "this is z"])
plt.show()


# In[10]:


sample_data = pd.read_csv('C:/Users/ADERONKE/Downloads/sample_data.csv')


# In[11]:


sample_data


# In[12]:


type(sample_data)


# In[13]:


sample_data.column_c


# In[14]:


type(sample_data.column_c)


# In[17]:


sample_data.column_c.iloc[4]


# In[23]:


plt.plot(sample_data.column_a, sample_data.column_b, 'o')
plt.plot(sample_data.column_a, sample_data.column_c)
plt.show()


# In[24]:


data = pd.read_csv("C:/Users/ADERONKE/Downloads/countries.csv")


# In[25]:


data


# In[26]:


# Compare the population growth in US and China


# In[29]:


data.country == 'United States'


# In[27]:


us = data[data.country == 'United States']


# In[28]:


us


# In[30]:


data[data.country == 'United States']


# In[31]:


China = data[data.country == 'China']


# In[32]:


China


# In[38]:


plt.plot(us.year, us.population / 10**6)
plt.plot(Chplt.plot(us.year, us.population / 10**6)
plt.plot(China.year, China.population / 10**6)
plt.legend(['United State', 'China'])
plt.xlabel('year')
plt.ylabel('population')
plt.show()ina.year, China.population / 10**6)
plt.legend(['United State', 'China'])
plt.xlabel('year')
plt.ylabel('population')
plt.show()


# In[40]:


us.population


# In[41]:


# To retrieve the first year population


# In[42]:


us.population.iloc[0]


# In[43]:


us.population / us.population.iloc[0]


# In[44]:


# Showing the result in percentage


# In[45]:


us.population / us.population.iloc[0] * 100


# In[46]:


plt.plot(us.year, us.population / us.population.iloc[0] * 100)
plt.plot(China.year, China.population / China.population.iloc[0] * 100)
plt.legend(['United State', 'China'])
plt.xlabel('year')
plt.ylabel('population growth (first year = 100)')
plt.show()


# In[47]:


# Tutorial


# In[48]:


from matplotlib import pyplot as plt


# In[49]:


plt.plot([1,2,3],[4,5,1])


# In[50]:


plt.show()


# In[51]:


x = [5,8,10]
y = [12,16,6]

plt.plot(x,y)

plt.title("info")
plt.ylabel("Y axis")
plt.xlabel("X axis")

plt.show()


# In[52]:


# Adding Style to the Graph


# In[53]:


from  matplotlib import pyplot as plt
from  matplotlib import style

style.use("ggplot")

X = [5,8,10]
Y = [12,16,6]

X2 = [6,9,11]
Y2 = [6,15,7]

plt.plot(x,y,'g',label='LineOne', linewidth=5)
plt.plot(X2,Y2,'c',label="LineTwo",linewidth=5)

plt.title("info")
plt.ylabel("Y-axis")
plt.xlabel("X-axis")

plt.show()


# In[59]:


# Bar Graph


# In[98]:


import matplotlib.pyplot as plt


Venue = [1, 2, 3, 5,  7, 9, 6]
Street = ['Arch', 'Earling', 'Bow', 'Westam', 'Barking', 'Eastham', 'Whipcross']


plt.bar(Venue, Street)



plt.xlabel('Venue')
plt.ylabel('Street')

plt.title('Venue Vs Street')

plt.grid(True)

plt.show()




# In[64]:


import matplotlib.pyplot as plt

population_ages = [22,55,62,45,21,22,34,42,42,4,99,102,110,120,121,122,130,111,115,112,80,75,65,54,44,43,42,48]

bins = [0,10,20,30,40,50,60,70,80,90,100,110,120,130]

plt.hist(population_ages, bins, histype='bar',rwidth=0.8)

plt.xlabel('x')
plt.ylabel('y')
plt.title('Histogram')
plt.legend()
plt.show()


# In[80]:


# Bar Graph


# In[97]:


import matplotlib.pyplot as plt
   
country = ['A', 'B', 'C', 'D', 'E']
gdp_per_capita = [45000, 42000, 52000, 49000, 47000]

colors = ['green', 'blue', 'purple', 'brown', 'teal']
plt.bar(country, gdp_per_capita, color=colors)
plt.title('Country Vs GDP Per Capita', fontsize=14)
plt.xlabel('Country', fontsize=14)
plt.ylabel('GDP Per Capita', fontsize=12)
plt.grid(True)
plt.show()


# In[76]:


# Scatter Plot


# In[79]:


import matplotlib.pyplot as plt

x = [1,2,3,4,5,6,7,8]
y = [5,2,4,2,1,4,5,2]

plt.scatter(x,y, label='skitscat', color='k', s=25, marker="o")

plt.xlabel('x')
plt.ylabel('y')
plt.title('Scatter Plot')
plt.legend()
plt.show()


# In[92]:


import matplotlib.pyplot as plt

x = [1, 1, 2, 3, 3, 5, 7, 8, 9, 10,
     10, 11, 11, 13, 13, 15, 16, 17, 18, 18,
     18, 19, 20, 21, 21, 23, 24, 24, 25, 25,
     25, 25, 26, 26, 26, 27, 27, 27, 27, 27,
     29, 30, 30, 31, 33, 34, 34, 34, 35, 36,
     36, 37, 37, 38, 38, 39, 40, 41, 41, 42,
     43, 44, 45, 45, 46, 47, 48, 48, 49, 50,
     51, 52, 53, 54, 55, 55, 56, 57, 58, 60,
     61, 63, 64, 65, 66, 68, 70, 71, 72, 74,
     75, 77, 81, 83, 84, 87, 89, 90, 90, 91
     ]

plt.hist(x, bins=10)
plt.show()


# In[99]:


import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
from matplotlib.ticker import PercentFormatter


# Creating dataset
np.random.seed(23685752)
N_points = 10000
n_bins = 20

# Creating distribution
x = np.random.randn(N_points)
y = .8 ** x + np.random.randn(10000) + 25
legend = ['distribution']

# Creating histogram
fig, axs = plt.subplots(1, 1,
						figsize =(10, 7),
						tight_layout = True)


# Remove axes splines
for s in ['top', 'bottom', 'left', 'right']:
	axs.spines[s].set_visible(False)

# Remove x, y ticks
axs.xaxis.set_ticks_position('none')
axs.yaxis.set_ticks_position('none')

# Add padding between axes and labels
axs.xaxis.set_tick_params(pad = 5)
axs.yaxis.set_tick_params(pad = 10)

# Add x, y gridlines
axs.grid(b = True, color ='grey',
		linestyle ='-.', linewidth = 0.5,
		alpha = 0.6)

# Add Text watermark
fig.text(0.9, 0.15, 'Jeeteshgavande30',
		fontsize = 12,
		color ='red',
		ha ='right',
		va ='bottom',
		alpha = 0.7)

# Creating histogram
N, bins, patches = axs.hist(x, bins = n_bins)

# Setting color
fracs = ((N**(1 / 5)) / N.max())
norm = colors.Normalize(fracs.min(), fracs.max())

for thisfrac, thispatch in zip(fracs, patches):
	color = plt.cm.viridis(norm(thisfrac))
	thispatch.set_facecolor(color)

# Adding extra features
plt.xlabel("X-axis")
plt.ylabel("y-axis")
plt.legend(legend)
plt.title('Customized histogram')

# Show plot
plt.show()


# In[109]:


# Stack Plot


# In[108]:


import matplotlib.pyplot as plt

days = [1,2,3,4,5]

sleeping = [7,8,6,11,7]
eating = [2,3,4,3,2]
working = [7,8,7,2,2]
playing = [8,5,7,8,13]

plt.plot([],[],color='m',label='sleeping',linewidth=5)
plt.plot([],[],color='c',label='eating',linewidth=5)
plt.plot([],[],color='r',label='working',linewidth=5)
plt.plot([],[],color='k',label='playing',linewidth=5)

plt.stackplot(days, sleeping, eating, working, playing, 
colors=['m','c','r','k'])

plt.xlabel('x')
plt.ylabel('y')
plt.title('Area Plot')
plt.legend()
plt.show()


# In[110]:


# Pie Chart


# In[114]:


import matplotlib.pyplot as plt

slices = [7,2,2,13]
activities = ['sleeping', 'eating', 'working', 'playing']
cols = ['c','m','r','b']

plt.pie(slices,
       labels=activities,
       colors=cols,
       shadow= True,
       explode=(0,0.1,0,0),
       autopct='%1.1f%%')

plt.title('Pie Chart')
plt.show()


# In[115]:


# How to handle multiple plots


# In[124]:


import numpy as np
import matplotlib.pyplot as plt

def f(t):
    return np.exp(-t) * np.cos(2*np.pi*t)
t1 = np.arrange(0.0, 5.0, 0.1)
t2 = np.arrange(0.0, 5.0, 0.02)

plt.subplot(211)
plt.plot(t1, f(t1), 'bo', t2, f(t2))

plt.subplot(212)
plt.plot(t2, np.cos(2*np.pi*t2))
plt.show()


# In[ ]:




