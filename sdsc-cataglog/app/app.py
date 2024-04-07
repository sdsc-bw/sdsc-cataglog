#import dash
import dash
from jupyter_dash import JupyterDash
from dash import html, dcc
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import pandas as pd

from dataset_request.dataset_request import search_top_related_local_datasets_with_cs, search_top_related_local_repositories_with_cs

#app = dash.Dash(__name__)
app = JupyterDash(__name__)

# Define the layout of the app
app.layout = html.Div([
    html.H1("Search for Repositories and Datasets"),
    html.P(f"Input 3 keywords and the tool will return the top 5 related repositories and datasets for each keyword."),
    dcc.Input(id='keyword1', type='text', placeholder='Keyword 1'),
    dcc.Input(id='keyword2', type='text', placeholder='Keyword 2'),
    dcc.Input(id='keyword3', type='text', placeholder='Keyword 3'),
    html.Button('Search', id='search-button'),
    html.H2("Top 5 Repositories"), 
    dbc.Row([
        dbc.Col(
            dcc.Loading(
                id="loading-repository1",
                type="default",
                children=[html.Div()]
            ),
            style={'display': 'inline-block', 'width': '31%', 'verticalAlign': 'top'}
        ),
        dbc.Col(
            dcc.Loading(
                id="loading-repository2",
                type="default",
                children=[html.Div()]
            ),
            style={'display': 'inline-block', 'width': '31%', 'margin-left': '2rem', 'verticalAlign': 'top'}
        ),
        dbc.Col(
            dcc.Loading(
                id="loading-repository3",
                type="default",
                children=[html.Div()]
            ),
            style={'display': 'inline-block', 'width': '31%', 'margin-left': '2rem', 'verticalAlign': 'top'}
        )
    ],),

    html.Hr(),

    html.H2("Top 5 Datasets"), 
    dbc.Row([
        dbc.Col(
            dcc.Loading(
                id="loading-datasets1",
                type="default",
                children=[html.Div()]
            ),
            style={'display': 'inline-block', 'width': '31%', 'verticalAlign': 'top'}
        ),
        dbc.Col(
            dcc.Loading(
                id="loading-datasets2",
                type="default",
                children=[html.Div()]
            ),
            style={'display': 'inline-block', 'width': '31%', 'margin-left': '2rem', 'verticalAlign': 'top'}
        ),
        dbc.Col(
            dcc.Loading(
                id="loading-datasets3",
                type="default",
                children=[html.Div()]
            ),
            style={'display': 'inline-block', 'width': '31%', 'margin-left': '2rem', 'verticalAlign': 'top'}
        )
    ],)
])

# Define the callback function to update the results area
@app.callback(
    Output('loading-repository1', 'children'),
    Input('search-button', 'n_clicks'),
    State('keyword1', 'value'),
)
def update_repositories(n_clicks, keyword):
    if n_clicks is None:
        return dash.no_update

    if keyword is None or keyword == '':
        return html.Div()

    # Call your search function and get the results
    repo_names, repo_descriptions, repo_urls = search_top_related_local_repositories_with_cs(keyword)

    
    result_html = [html.H3(f"Keyword: {keyword}")]
    if len(repo_names) == 0:
        result_html.append(html.Div([
            html.P(f"No related repositories found"),
        ]))
    else:
        for i in range(len(repo_names)):  # Display the top 5 results
            result_html.append(html.Div([
                html.H4(f"{i + 1}. {repo_names[i]}"),
                html.P(f"Description: {repo_descriptions[i]}"),
                dcc.Link("Repository URL", href=repo_urls[i], target='_blank'),
                #html.Hr()
            ]))
    
    return result_html

# Define the callback function to update the results area
@app.callback(
    Output('loading-repository2', 'children'),
    Input('search-button', 'n_clicks'),
    State('keyword2', 'value'),
)
def update_repositories(n_clicks, keyword):
    if n_clicks is None:
        return dash.no_update

    if keyword is None or keyword == '':
        return html.Div()

    # Call your search function and get the results
    repo_names, repo_descriptions, repo_urls = search_top_related_local_repositories_with_cs(keyword)

    
    result_html = [html.H3(f"Keyword: {keyword}")]
    if len(repo_names) == 0:
        result_html.append(html.Div([
            html.P(f"No related repositories found"),
        ]))
    else:
        for i in range(len(repo_names)):  # Display the top 5 results
            result_html.append(html.Div([
                html.H4(f"{i + 1}. {repo_names[i]}"),
                html.P(f"Description: {repo_descriptions[i]}"),
                dcc.Link("Repository URL", href=repo_urls[i], target='_blank'),
                #html.Hr()
            ]))
    
    return result_html

# Define the callback function to update the results area
@app.callback(
    Output('loading-repository3', 'children'),
    Input('search-button', 'n_clicks'),
    State('keyword3', 'value'),
)
def update_repositories(n_clicks, keyword):
    if n_clicks is None:
        return dash.no_update

    if keyword is None or keyword == '':
        return html.Div()

    # Call your search function and get the results
    repo_names, repo_descriptions, repo_urls = search_top_related_local_repositories_with_cs(keyword)

    
    result_html = [html.H3(f"Keyword: {keyword}")]
    if len(repo_names) == 0:
        result_html.append(html.Div([
            html.P(f"No related repositories found"),
        ]))
    else:
        for i in range(len(repo_names)):  # Display the top 5 results
            result_html.append(html.Div([
                html.H4(f"{i + 1}. {repo_names[i]}"),
                html.P(f"Description: {repo_descriptions[i]}"),
                dcc.Link("Repository URL", href=repo_urls[i], target='_blank'),
                #html.Hr()
            ]))
    
    return result_html

# Define the callback function to update the results area
@app.callback(
    Output('loading-datasets1', 'children'),
    Input('search-button', 'n_clicks'),
    State('keyword1', 'value'),
)
def update_datasets(n_clicks, keyword):
    if n_clicks is None:
        return dash.no_update

    if keyword is None or keyword == '':
        return html.Div()

    # Call your search function and get the results
    dataset_names, dataset_ids, dataset_description, dataset_urls = search_top_related_local_datasets_with_cs(keyword)

    
    result_html = [html.H3(f"Keyword: {keyword}")]
    if len(dataset_names) == 0:
        result_html.append(html.Div([
            html.P(f"No related datasets found"),
        ]))
    else:
        for i in range(len(dataset_names)):  # Display the top 5 results
            result_html.append(html.Div([
                html.H4(f"{i + 1}. {dataset_names[i]}"),
                html.P(f"ID: {dataset_ids[i]}"),
                html.P(f"Description: {dataset_description[i]}"),
                dcc.Link("Dataset URL", href=dataset_urls[i], target='_blank'),
                #html.Hr()
            ]))
    
    return result_html

# Define the callback function to update the results area
@app.callback(
    Output('loading-datasets2', 'children'),
    Input('search-button', 'n_clicks'),
    State('keyword2', 'value'),
)
def update_datasets(n_clicks, keyword):
    if n_clicks is None:
        return dash.no_update

    if keyword is None or keyword == '':
        return html.Div()

    # Call your search function and get the results
    dataset_names, dataset_ids, dataset_description, dataset_urls = search_top_related_local_datasets_with_cs(keyword)

    result_html = [html.H3(f"Keyword: {keyword}")]
    if len(dataset_names) == 0:
        result_html.append(html.Div([
            html.P(f"No related datasets found"),
        ]))
    else:
        for i in range(len(dataset_names)):  # Display the top 5 results
            result_html.append(html.Div([
                html.H4(f"{i + 1}. {dataset_names[i]}"),
                html.P(f"ID: {dataset_ids[i]}"),
                html.P(f"Description: {dataset_description[i]}"),
                dcc.Link("Dataset URL", href=dataset_urls[i], target='_blank'),
                #html.Hr()
            ]))
    
    return result_html

# Define the callback function to update the results area
@app.callback(
    Output('loading-datasets3', 'children'),
    Input('search-button', 'n_clicks'),
    State('keyword3', 'value'),
)
def update_datasets(n_clicks, keyword):
    if n_clicks is None:
        return dash.no_update

    if keyword is None or keyword == '':
        return html.Div()

    # Call your search function and get the results
    dataset_names, dataset_ids, dataset_description, dataset_urls = search_top_related_local_datasets_with_cs(keyword)

    
    result_html = [html.H3(f"Keyword: {keyword}")]
    if len(dataset_names) == 0:
        result_html.append(html.Div([
            html.P(f"No related datasets found"),
        ]))
    else:
        for i in range(len(dataset_names)):  # Display the top 5 results
            result_html.append(html.Div([
                html.H4(f"{i + 1}. {dataset_names[i]}"),
                html.P(f"ID: {dataset_ids[i]}"),
                html.P(f"Description: {dataset_description[i]}"),
                dcc.Link("Dataset URL", href=dataset_urls[i], target='_blank'),
                #html.Hr()
            ]))
    
    return result_html
