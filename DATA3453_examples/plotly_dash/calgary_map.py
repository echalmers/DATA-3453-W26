import plotly.express as px
import pandas as pd
import json
from dash import Dash, html, dcc, callback, Output, Input
import dash_ag_grid as dag

# load the data
df = pd.read_csv('housing_holdout.csv', index_col=None)
df = df.groupby('comm_name')['re_assessed_value_2024'].mean().reset_index()
print(df.head())

with open('community_boundaries.geojson', 'r') as f:
    communities = json.load(f)

app = Dash()

app.layout = [
    html.Div(children='My Dash App'),
    html.Hr(),
    dcc.Graph(
        figure={},
        id='map'
    ),
    dcc.Slider(df['re_assessed_value_2024'].min() + 1, 1_000_000, value=250_000, id='slider')
]


# add the callback function to connect the slider to the map
@callback(
    Output(component_id='map', component_property='figure'),
    Input(component_id='slider', component_property='value')
)
def update_graph(slider_value):
    print('new slider value:', slider_value)
    df['out_of_budget'] = df['re_assessed_value_2024'] > slider_value
    fig = px.choropleth_map(
        df,
        geojson=communities,
        locations='comm_name',
        center={'lat': 51.053395, 'lon': -114.070909},
        color='out_of_budget',
        color_discrete_map={True: 'gray', False: 'blue'},
        opacity=0.4,
    )
    return fig


if __name__ == '__main__':
    app.run(debug=True)
