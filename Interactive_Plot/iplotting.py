/*
This was inspired from https://towardsdatascience.com/building-rnn-lstm-and-gru-for-time-series-using-pytorch-a46e5b094e7b
*/

import plotly.graph_objs as go
from plotly.offline import iplot

def iplotting(df, columns_for_x, colors, title, xlabel, ylabel):
  data = []
  for i, column in enumerate (columns_for_x):
    value = go.Scatter(
         x = df.index,
         y = df[column],
         mode = 'lines',
         name = column,
         marker = dict(),
         text = df.index,
         line = dict(color=colors[i]),
    )
    data.append(value)

  layout = dict(
      title = {'text': title,
               'y':0.9,
               'x':0.45,
               'xanchor': 'center'},
      xaxis = dict(title=xlabel, ticklen=5, zeroline=False),
      yaxis = dict(title=ylabel, ticklen=5, zeroline=False)
  )
  
  fig = dict(data=data, layout=layout)
  iplot(fig)

  
### Usage Example ###
/* 
  df = pd.read_csv('data.csv')  # a csv file having 2 columns with each name of 'col_name_1' and 'col_name_2'
  columns_plot = ['col_name_1', 'col_name_1']
  line_colors = ["#ff7d00", "#8D8686"]
  title = "Interactive Plotting Example"
  xlabel = 'index'
  ylabel = 'value'
  iplotting(df, columns_plot, line_colors, title, xlabel, ylabel)
*/
