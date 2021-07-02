from main import df
from bokeh.plotting import figure, show, output_file
from bokeh.models import HoverTool, ColumnDataSource

df["start_string"] = df["start"].dt.strftime("%Y-%m-%d %H:%M:%S")

df["end_string"] = df["end"].dt.strftime("%Y-%m-%d %H:%M:%S")
cds = ColumnDataSource(df)
p = figure(x_axis_type='datetime', height=300, width=700, sizing_mode='scale_width', title="Motion Graph")

hover = HoverTool(tooltips=[("start", "@start_string"), ("end", "@end_string")])
p.add_tools(hover)

q = p.quad(left="start", right="end", bottom=0, top=1, color="green", source=cds)

output_file("Graph.html")

show(p)
