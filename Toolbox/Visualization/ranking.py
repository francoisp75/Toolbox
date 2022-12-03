import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('data/Measurement_summary.csv')
#print(df.head())
#print(df.info())

#Let's look at the total number of the variable 'Station code.'
df['Station code'].nunique()
## output
## 25 : There are 25 districts in total.
set(df['Station code'])

"""
Select and prepare data
For example, I will select Station codes 111-118.
If you want to plot other station numbers, feel free to modify the code below.
"""
list_stations = [111, 112, 113, 114, 115, 116, 117, 118]
df_select = df[df['Station code'].isin(list_stations)]
df_select.head()

"""
The retrieved dataset is not ready to be plotted.
Some columns are needed to be created or modified before use.
"""
## crete year_month, year and month columns
year_month = [i[0:7] for i in list(df_select['Measurement date'])]
df_select['year_month'] = year_month
df_select['year'] = [i[0:4] for i in year_month]
df_select['month'] = [i[-2:] for i in year_month]

## create district name column
district = [i.split(', ')[2] for i in df_select['Address']]
df_select['District'] = district

## change Station code column type
df_select = df_select.astype({'Station code': str})

## groupby with location and point of time
df_month = df_select.groupby(
    ['Station code', 'District', 'year_month', 'year', 'month']).mean()
df_month.reset_index(inplace=True)
df_month.head()

"""
Here comes an important step. The main idea of this article is to create
visualizations for ranking data. Next,
we will create a column for ranking the districts' CO number (ppm)
during each time point.
"""

keep = []
for i in list(set(df_month['year_month'])):
    df = df_month[df_month['year_month'] == i]
    order = df['CO'].rank(ascending=0)
    df['rank'] = [int(i) for i in order]
    keep.append(df)

df_month = pd.concat(keep)
df_month.sort_values(['year_month', 'Station code'],
                     ascending=True,
                     inplace=True,
                     ignore_index=True)
#print(df_month.head())

"""
Before continuing, we will define a dictionary of colors to facilitate
the plotting process.
"""
#extract color palette, the palette can be changed
list_dist = list(set(df_select['District']))
pal = list(
    sns.color_palette(palette='Spectral', n_colors=len(list_dist)).as_hex())
dict_color = dict(zip(list_dist, pal))

# Data visualization
# Comparing bar height with an Animated bar chart
import plotly.express as px

fig = px.bar(
    df_month,
    x='District',
    y='CO',
    color='District',
    text='rank',
    color_discrete_map=dict_color,
    animation_frame='year_month',
    animation_group='Station code',
    range_y=[0, 1.2],
    labels={'CO': 'CO (ppm)'},
)
fig.update_layout(width=1000,
                  height=600,
                  showlegend=False,
                  xaxis=dict(tickmode='linear', dtick=1))
fig.update_traces(textfont_size=16, textangle=0)
#print(fig.show())


#Racing with an Animated scatter plot
ym = list(set(year_month))
ym.sort()

df_month['posi'] = [ym.index(i) for i in df_month['year_month']]
df_month['CO_str'] = [str(round(i, 2)) for i in df_month['CO']]
df_month['CO_text'] = [str(round(i, 2)) + ' ppm' for i in df_month['CO']]
df_month.head()

import plotly.express as px

fig = px.scatter(df_month,
                 x='posi',
                 y='rank',
                 size='CO',
                 color='District',
                 text='CO_text',
                 color_discrete_map=dict_color,
                 animation_frame='year_month',
                 animation_group='District',
                 range_x=[-2, len(ym)],
                 range_y=[0.5, 6.5])
fig.update_xaxes(title='', visible=False)
fig.update_yaxes(autorange='reversed',
                 title='Rank',
                 visible=True,
                 showticklabels=True)
fig.update_layout(xaxis=dict(showgrid=False), yaxis=dict(showgrid=True))
fig.update_traces(textposition='middle left')
fig.show()
