from bokeh.themes import built_in_themes, DARK_MINIMAL
from bokeh.plotting import figure, Document
from bokeh.embed import components
from bokeh.models import ResetTool, CrosshairTool, HoverTool, PanTool, WheelZoomTool

def plot(data: dict):
    ...