# importing libraries
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import seaborn as sns
import sys

# loading the csv file
data = pd.read_csv("training_set_VU_DM.csv")
pd.set_option('max_columns', None)
pd.set_option('max_rows', None)

# To view the first five rows of the dataset
print(data.head())
# The no. of rows and columns: 4958347, 54
print(data.shape)
# The names of all the columns
print(data.columns)
# The no. of unique values for each variable
print(data.nunique(axis=0))
# Summary of the dataset: the count, mean, sd, min, max for numeric variables
sys.stdout = open("Results_describe.txt", "w")
print(data.describe().apply(lambda s: s.apply(lambda x: format(x, 'f'))))
sys.stdout.close()
# To get data information
sys.stdout = open("Results_info.txt", "w")
print(data.info())
sys.stdout.close()

### ANALYSING THE DATA

# 1. CORRELATION MATRIX
corr = data.corr() # plotting the heatmap
sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns, annot=False,
            cmap=sns.diverging_palette(220, 20, as_cmap=True))
plt.show()  # needed to display the plot

# 2. Countries of the origin of Guests
sys.stdout = open("Country_guests.txt", "w")
print(data['visitor_location_country_id'].value_counts(normalize=True))
sys.stdout.close()

# Plotting the top 10 countries of the origin of guests
sns.countplot(x='visitor_location_country_id',
              data=data, order=pd.value_counts(data['visitor_location_country_id']).iloc[:10].index,
              palette='colorblind')
plt.title('Top 10 countries of the guests', weight='bold')
plt.xlabel('Country')
plt.ylabel('Reservation Count')
plt.show()

# 3. Countries of holiday destination
sys.stdout = open("Country_destination.txt", "w")
print(data['prop_country_id'].value_counts(normalize=True))
sys.stdout.close()

# Plotting the top 10 holiday destination countries
sns.countplot(x='prop_country_id',
              data=data, order=pd.value_counts(data['prop_country_id']).iloc[:10].index, palette='colorblind')
plt.title('Top 10 holiday destination countries', weight='bold')
plt.xlabel('Country')
plt.ylabel('Reservation Count')
plt.show()

