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
      title = title,
      xaxis = dict(title=xlabel, ticklen=5, zeroline=False),
      yaxis = dict(title=ylabel, ticklen=5, zeroline=False)
  )
  fig = dict(data=data, layout=layout)
  iplot(fig)
