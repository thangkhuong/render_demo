#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import plotly.express as px
import plotly.graph_objs as go
from dash import Dash, dcc, html, Input, Output, dash_table
import dash_bootstrap_components as dbc

# Create a sample dataset
np.random.seed(42)
n_samples = 1000
hospitals = ['A', 'B', 'C', 'D']
hospital = np.random.choice(hospitals, size=n_samples)
age = np.random.randint(20, 80, size=n_samples)
severity = np.random.randint(1, 11, size=n_samples)
los = np.random.normal(loc=7 + 0.05*age + 0.5*severity + 
                       (hospital == 'A') * 2 + 
                       (hospital == 'B') * 1.5 + 
                       (hospital == 'C') * 1 + 
                       (hospital == 'D') * 0.5, 
                       scale=1.5, size=n_samples)
date_range = pd.date_range(start='2023-01-01', end='2023-12-31', periods=n_samples)
date = np.random.choice(date_range, size=n_samples)
data = pd.DataFrame({'Hospital': hospital, 'Age': age, 'Severity': severity, 'LOS': los, 'Date': date})

# Initialize Dash app
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Layout of the app
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.H1("Hospital LOS Analysis Dashboard", className='text-center mb-4'), width=12)
    ]),
    dbc.Row([
        dbc.Col([
            dcc.DatePickerRange(
                id='date-picker-range',
                min_date_allowed=data['Date'].min(),
                max_date_allowed=data['Date'].max(),
                start_date=data['Date'].min(),
                end_date=data['Date'].max()
            )
        ], width=6),
        dbc.Col([
            dbc.DropdownMenu(
                label="Select Hospitals",
                children=[
                    dcc.Checklist(
                        id='hospital-dropdown',
                        options=[{'label': h, 'value': h} for h in hospitals],
                        value=hospitals,
                        inline=False  # Set inline=False to arrange checkboxes vertically
                    )
                ],
                className='mb-4'
            ),
        ], width=6)
    ]),
    dbc.Row([
        dbc.Col([
            dbc.Button("Apply Filter", id='apply-filter-btn', color='primary', className='mb-4'),
        ], width=12, className='d-flex justify-content-end')
    ]),
    dbc.Row([
        dbc.Col([
            html.H5("Adjusted Means, Raw Means, and Relative Adjusted Means Table"),
            dash_table.DataTable(id='table-container', style_table={'overflowX': 'auto'})
        ], width=12)
    ]),
    dbc.Row([
        dbc.Col(dcc.Graph(id='raw-mean-plot'), width=6),
        dbc.Col(dcc.Graph(id='adjusted-mean-plot'), width=6),
    ]),
    dbc.Row([
        dbc.Col(dcc.Graph(id='relative-adjusted-mean-plot'), width=12),
    ]),
    dbc.Row([
        dbc.Col(dcc.Graph(id='los-distribution-plot'), width=12),
    ]),
], fluid=True, style={'padding': '20px'})

@app.callback(
    [Output('table-container', 'data'),
     Output('table-container', 'columns'),
     Output('raw-mean-plot', 'figure'),
     Output('adjusted-mean-plot', 'figure'),
     Output('relative-adjusted-mean-plot', 'figure'),
     Output('los-distribution-plot', 'figure')],
    [Input('apply-filter-btn', 'n_clicks')],
    [Input('date-picker-range', 'start_date'), Input('date-picker-range', 'end_date'), Input('hospital-dropdown', 'value')]
)
def update_dashboard(n_clicks, start_date, end_date, selected_hospitals):
    if start_date is None or end_date is None or selected_hospitals is None:
        return [], [], {}, {}, {}, {}

    # Filter data based on date range and selected hospitals
    filtered_data = data[(data['Date'] >= start_date) & (data['Date'] <= end_date) & (data['Hospital'].isin(selected_hospitals))]

    # Fit linear regression model
    formula = 'LOS ~ C(Hospital) + Age + Severity'
    model = smf.ols(formula, data=filtered_data).fit()

    # Calculate raw mean LOS for each hospital
    raw_means = filtered_data.groupby('Hospital')['LOS'].mean().to_dict()

    # Predict adjusted LOS for each hospital by setting covariates to their mean values
    adjusted_means = {}
    for h in selected_hospitals:
        pred_data = pd.DataFrame({
            'Hospital': [h],
            'Age': [filtered_data['Age'].mean()],
            'Severity': [filtered_data['Severity'].mean()]
        })
        pred = model.get_prediction(pred_data).predicted_mean[0]
        adjusted_means[h] = pred

    # Calculate relative adjusted means
    relative_means = {}
    for h in selected_hospitals:
        other_means = [mean for hosp, mean in adjusted_means.items() if hosp != h]
        rest_mean = np.mean(other_means)
        relative_means[h] = adjusted_means[h] / rest_mean

    # Prepare data for the table
    table_data = {
        'Hospital': selected_hospitals,
        'Raw Mean LOS': [raw_means[h] for h in selected_hospitals],
        'Adjusted Mean LOS': [adjusted_means[h] for h in selected_hospitals],
        'Relative Adjusted Mean LOS': [relative_means[h] for h in selected_hospitals]
    }
    table_df = pd.DataFrame(table_data)
    columns = [{"name": i, "id": i} for i in table_df.columns]

    # Plot raw mean LOS
    raw_mean_fig = px.bar(table_df, x='Hospital', y='Raw Mean LOS', title='Raw Mean LOS by Hospital')

    # Plot adjusted mean LOS
    adjusted_mean_fig = px.bar(table_df, x='Hospital', y='Adjusted Mean LOS', title='Adjusted Mean LOS by Hospital')

    # Plot relative adjusted mean LOS
    relative_adjusted_mean_fig = px.bar(table_df, x='Hospital', y='Relative Adjusted Mean LOS', title='Relative Adjusted Mean LOS by Hospital')
    relative_adjusted_mean_fig.add_hline(y=1, line_dash="dash", line_color="red")

    # Plot LOS distribution
    los_dist_fig = go.Figure()
    for h in selected_hospitals:
        subset = filtered_data[filtered_data['Hospital'] == h]
        los_dist_fig.add_trace(go.Histogram(x=subset['LOS'], name=f'Hospital {h}', opacity=0.5))
    los_dist_fig.update_layout(title='LOS Distribution by Hospital', barmode='overlay')
    los_dist_fig.update_traces(opacity=0.75)

    return table_df.to_dict('records'), columns, raw_mean_fig, adjusted_mean_fig, relative_adjusted_mean_fig, los_dist_fig

if __name__ == '__main__':
    app.run_server(debug=True)


# In[2]:


import os
print(os.getcwd())


# In[ ]:




