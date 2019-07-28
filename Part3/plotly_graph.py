#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  8 22:34:51 2017

@author: Jiajia
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 15:23:44 2017

@author: Jiajia
"""

import pandas as pd
import plotly
plotly.tools.set_credentials_file(username='jiajialiu113', api_key='wFBG19KmTeLeoDwjlEPM')

import matplotlib.pyplot as plt
##When you set-up plotly, you will have a .credentials file on your computer with this
##info as well

import plotly.plotly as py
import plotly.graph_objs as go
from plotly import tools


foxCleandf=pd.read_csv('/Users/kaimouto/Desktop/501/project/project 2/foxnews_cleaned.csv')
cnnCleandf=pd.read_csv('/Users/kaimouto/Desktop/501/project/project 2/cnn_cleaned.csv')
ixicCleandf=pd.read_csv('/Users/kaimouto/Desktop/501/project/project 2/IXIC_cleaned.csv')#nasdaq
goldCleandf=pd.read_csv('/Users/kaimouto/Desktop/501/project/project 2/gold_cleaned.csv')
gspcCleandf=pd.read_csv('/Users/kaimouto/Desktop/501/project/project 2/gspc_cleaned.csv')#sp500
combinedf=pd.read_csv('/Users/kaimouto/Desktop/501/project/project 2/NewsCombined.csv')
df1=pd.read_csv('/Users/kaimouto/Desktop/501/project/project 2/News_vs_Gold.csv')
df2=pd.read_csv('/Users/kaimouto/Desktop/501/project/project 2/News_vs_Gspc.csv')
df3=pd.read_csv('/Users/kaimouto/Desktop/501/project/project 2/News_vs_Ixic.csv')

plt.plot(gspcCleandf['volume'])
plt.title ('the outlook of variable adjusted_close price with s&p 500')
plt.xlabel('counts of stock')
plt.ylabel('stock price')
plt.show ()


trace1=go.Box(y=ixicCleandf['adjusted_close'], name="Nasdaq's stock price", boxmean=True, jitter=.3)
trace2=go.Box(y=goldCleandf['adjusted_close'], name="Gold's stock price", boxmean=True)
trace3=go.Box(y=gspcCleandf['adjusted_close'], name="S&P 500's stock price", boxmean=True)
data=[trace1,trace2,trace3]


layout1 = go.Layout(title = "Box Plots for adjusted_close")

# Setup figure
fig = go.Figure(data=data, layout=layout1)

py.iplot(data,filename='stock datasets')

py.plot(data, filename='stock datasets')

#============================================================================================
trace1=go.Box(y=ixicCleandf['volume'], name="Nasdaq's volume", boxmean=True, jitter=.3)
trace2=go.Box(y=goldCleandf['volume'], name="Gold's volume", boxmean=True)
trace3=go.Box(y=gspcCleandf['volume'], name="S&P 500's volume", boxmean=True)

data_vol=[trace1,trace2,trace3]


layout2 = go.Layout(title = "Box Plots for volume")

# Setup figure
fig = go.Figure(data=data_vol, layout=layout2)

py.iplot(data_vol,filename='stock datasets_volume')

py.plot(data_vol, filename='stock dataset_volume')



#============================================================================================
trace1 = go.Scatter(
    x=combinedf['Date'],
    y=combinedf['CNNF&R'],
    
    name='CNN News Dataset'
)
trace2 = go.Scatter(
    x=combinedf['Date'],
    y=combinedf['FoxNewsF&R'],
    
    name='FoxNews Dataset',
    yaxis='y2'
)


data1 = [trace1, trace2]
layout = go.Layout(
    title='News Datasets',
    yaxis=dict(
        title='CNN News Dataset'
    ),
            
    yaxis2=dict(
        title='FoxNews Dataset',
        titlefont=dict(
            color='rgb(148, 103, 189)'
        ),
        tickfont=dict(
            color='rgb(148, 103, 189)'
        ),
        overlaying='y',
        side='right'
    ),
)
fig = go.Figure(data=data1, layout=layout)
plot_url = py.plot(fig, filename='News Datasets')

#============================================================================================

trace1 = go.Bar(x=df1['increase_rate'], y=df1['CNNPoints'], xaxis='x1', yaxis='y1',
                marker=dict(color='#404040'),
                name='CNN Points v.s Gold')
                
trace2 = go.Bar(x=df2['increase_rate'], y=df2['CNNPoints'], xaxis='x2', yaxis='y2',
                marker=dict(color='#404040'),
                name='CNN Points v.s S&P 500')
                
trace3 = go.Bar(x=df3['increase_rate'], y=df3['CNNPoints'], xaxis='x3', yaxis='y3',
                marker=dict(color='#404040'),
                name='CNN Points v.s Nasdaq')

fig = tools.make_subplots(rows=1, cols=3, shared_yaxes=True)

fig.append_trace(trace1, 1, 1)
fig.append_trace(trace2, 1, 2)
fig.append_trace(trace3, 1, 3)   

fig['layout'].update(height=500, width=850,
                     title='Multiple Bar Charts with stock datasets')
py.iplot(fig, filename='multiple-subplots')
plot_url = py.plot(fig, filename='multiple-subplots')   
#============================================================================================

trace4 = go.Bar(x=df1['increase_rate'], y=df1['FoxNewsPoints'], xaxis='x4', yaxis='y4',
                marker=dict(color='#0099ff'),
                name='Foxnews Points v.s Gold')
             
trace5 = go.Bar(x=df2['increase_rate'], y=df2['FoxNewsPoints'], xaxis='x5', yaxis='y5',
                marker=dict(color='#0099ff'),
                name='Foxnews Points v.s S&P 500')
                
trace6 = go.Bar(x=df3['increase_rate'], y=df3['FoxNewsPoints'], xaxis='x6', yaxis='y6',
                marker=dict(color='#0099ff'),
                name='Foxnews Points v.s Nasdaq')


fig = tools.make_subplots(rows=1, cols=3, shared_yaxes=True)

fig.append_trace(trace4, 1, 1)
fig.append_trace(trace5, 1, 2)
fig.append_trace(trace6, 1, 3)


fig['layout'].update(height=500, width=850,
                     title='Multiple Bar Charts with stock datasets')
py.iplot(fig, filename='multiple-subplots1')
plot_url = py.plot(fig, filename='multiple-subplots1')