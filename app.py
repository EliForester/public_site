from flask import Flask, render_template, url_for, request, redirect, session
from flask_bootstrap import Bootstrap
from flask_nav import Nav
from flask_nav.elements import Navbar, View, Subgroup, Link, Text, Separator
import pandas as pd
from bokeh.plotting import figure,vplot
from bokeh.charts.utils import cycle_colors
from bokeh.models import LinearAxis,Range1d
from bokeh.embed import components
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
import Quandl
import datetime

df_merged_all = pd.read_csv('https://github.com/EliForester/kcmo_311_data_viewer/raw/master/data_311.csv')
df_merged_all.reset_index()
df_merged_all['CREATION DATE'] = pd.to_datetime(df_merged_all['CREATION DATE'],format='%Y-%m-%d')

tools = "pan,wheel_zoom,box_zoom,reset,resize"
authtoken = 'Yw2y-7UJQzKyK5sL4kZH'

app = Flask(__name__)
Bootstrap(app)
app.secret_key = '2we2rsdkfja@ER@lk2@*&09*)(DS*F'
app.config['BOOTSTRAP_USE_MINIFIED'] = False

nav = Nav()

@nav.navigation()
def navbar():
    return Navbar(
        'Eli Forester',
        View('Home','index'),
        Subgroup('Projects',
            View('311 Data Viewer','kcmo_311_data_viewer'),
            View('Stock Chart','stock_chart')
    ))

nav.init_app(app)

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/kcmo_311_data_viewer',methods=['GET','POST'])
def kcmo_311_data_viewer():
    if request.method == 'POST':
        # Get the user's chart and set options, put into single session
        session['chart_name'] = request.form['chart_name']
        checkbox_list = request.form.getlist('checkbox')[:]
        start_date = request.form['start']
        end_date = request.form['end']

        if checkbox_list:
            for option in checkbox_list:
                session[option] = True
        if start_date or end_date:
            session['start'] = start_date
            session['end'] = end_date
        return redirect(url_for('graph'))
    else:
        request_types = df_merged_all.keys().unique()[1:-1]
        return render_template('kcmo_311_data_viewer.html',request_types=request_types)

@app.route('/kcmo_311_data_viewer/graph')
def graph():
    # Only display graph if variables are set in a cookie
    if session:
        chart_name = session['chart_name']
        chart_colors = {}

        if 'start' in session:
            start_date = session['start']
        else:
            start_date = '01/01/2014'
        if 'end' in session:
            end_date = session['end']
        else:
            end_date = '12/31/2014'

        df_merged = df_merged_all[(df_merged_all['CREATION DATE'] > start_date) & (df_merged_all['CREATION DATE'] < end_date)]

        # Set random chart line colors
        for i in range(0,10):
            chart_colors[str(i)] = np.random.normal(0,1,100)
            palette = cycle_colors(chart_colors)

        def random_color(palette):
            color = palette[np.random.randint(0,palette.__len__())]
            return color

        # Draw the basic graph
        graph = figure(width=1800, height=600,x_axis_type='datetime',title=chart_name,x_axis_label='Date',y_axis_label='Number of Reports',tools=tools)
        graph.line(df_merged['CREATION DATE'],df_merged[chart_name],color=random_color(palette),legend='311 Reports')

        # Add the options from the checkboxes
        if 'include_temp' in session:
            graph.extra_y_ranges = {'Avg Daily Temperature': Range1d(start=-20,end=33)}
            graph.add_layout(LinearAxis(y_range_name='Avg Daily Temperature',axis_label='Temp (C)'), 'right')
            graph.line(df_merged['CREATION DATE'],df_merged['Avg Temp'],color='green',y_range_name='Avg Daily Temperature',legend='Temp')

        if 'include_mean' in session:
            graph.line(df_merged['CREATION DATE'],df_merged[chart_name].mean(),color='navy',legend='311 Report Avg',line_dash=[6,3])

        if 'show_lr' in session or 'show_dt' in session:
            scatter = figure(height=800,width=800,title=chart_name,tools=tools)
            scatter.circle(df_merged['Avg Temp'],df_merged[chart_name])
            scatter.yaxis.axis_label = 'Reports'
            scatter.xaxis.axis_label = 'Temp'
            xx = np.arange(df_merged['Avg Temp'].min(),df_merged['Avg Temp'].max()).T
            if 'show_dt' in session:
                dt_regressor = DecisionTreeRegressor(max_depth=2)
                dt_regressor.fit(np.array([df_merged['Avg Temp']]).T, df_merged[chart_name])
                dt_yy = []
                for x_cord in xx:
                    dt_yy.append(float(dt_regressor.predict(x_cord)))
                scatter.line(xx,dt_yy,line_width=4,color='red',line_alpha=0.4,legend='Prediction')
            if 'show_lr' in session:
                lr_regressor = LinearRegression()
                lr_regressor.fit(np.array([df_merged['Avg Temp']]).T, df_merged[chart_name])
                lr_score = lr_regressor.score(np.array([df_merged['Avg Temp']]).T, df_merged[chart_name])
                lr_yy = []
                for x_cord in xx:
                    lr_yy.append(float(lr_regressor.predict(x_cord)))
                scatter.line(xx,lr_yy,line_width=4,color='green',line_alpha=0.4,legend='R^2={}'.format(str(lr_score)))
            # Vertical plot - output data to bokeh template
            vert = vplot(graph,scatter)
            script, div = components(vert)
        else:
            script, div = components(graph)
        session.clear() # Clear cookie
        return render_template('graph.html', script=script, div=div, chart_name=chart_name)
    # If no cookies set, return to the index page
    else:
        return redirect('/')

@app.route('/stock_chart',methods=['GET','POST'])
def stock_chart():
    if request.method == 'POST':
        # Get form data and/or set session defaults
        if request.form['symbol']:
            session['symbol'] = request.form['symbol']
        else:
            session['symbol'] = 'AAPL'

        checkbox_list = request.form.getlist('checkbox')[:]

        if request.form['start']:
            session['start'] = request.form['start']
        else:
            session['start'] = '01/01/2015'

        if request.form['end']:
            session['end'] = request.form['end']
        else:
            session['end'] =  '12/31/2015'

        if checkbox_list:
            for option in checkbox_list:
                session[option] = True

        return redirect(url_for('stock_chart_graph'))
    return render_template('stock_chart.html')

@app.route('/stock_chart/graph')
def stock_chart_graph():
    #Gather up the options from the session
    symbol = session['symbol']
    start_date = datetime.datetime.strftime(datetime.datetime.strptime(session['start'],'%m/%d/%Y'),'%Y-%m-%d')
    end_date = datetime.datetime.strftime(datetime.datetime.strptime(session['end'],'%m/%d/%Y'),'%Y-%m-%d')

    #Get market data and do some error handling
    try:
        symbol_data = Quandl.get('WIKI/{}'.format(symbol),trim_start=start_date,trim_end=end_date,authtoken=authtoken)
    except Quandl.Quandl.DatasetNotFound:
        error_text = 'Dataset not found'
        return redirect(url_for('error',error_text=error_text))
    except:
        error_text = 'Unknown error'
        return redirect(url_for('error',error_text=error_text))

    if len(symbol_data) == 0:
        error_text = 'No data with these criteria: try using a different symbol or date range'
        return redirect(url_for('error',error_text=error_text))

    #Build graph
    graph = figure(width=1800, height=600,title=symbol,x_axis_type='datetime',tools=tools)
    graph.line(symbol_data.index,symbol_data['Close'],color='blue',line_width=2)

    #Add graph options
    #if 'include_vol' in session: save this for later
    #    max_vol = symbol_data.Volume.max() / 100000
    #    min_vol = symbol_data.Volume.min() / 100000
    #    graph.extra_y_ranges = {'Vol': Range1d(start=min_vol,end=max_vol)}
    #    graph.add_layout(LinearAxis(y_range_name='Vol',axis_label='Vol'), 'right')
    #    graph.line(symbol_data.index,symbol_data['Volume']/100000,color='green',y_range_name='Vol',legend='Vol')

    if 'include_bb' in session:
        symbol_data['MEAN'] = pd.stats.moments.rolling_mean(symbol_data['Close'],20)
        symbol_data['STDDEV'] = pd.stats.moments.rolling_std(symbol_data['Close'],20)
        symbol_data['UPPER_BB'] = symbol_data['MEAN']+2*symbol_data['STDDEV']
        symbol_data['LOWER_BB'] = symbol_data['MEAN']-2*symbol_data['STDDEV']

        graph.line(symbol_data.index,symbol_data['MEAN'],color='green',line_alpha=0.4)
        graph.line(symbol_data.index,symbol_data['UPPER_BB'],color='orange',line_alpha=0.4)
        graph.line(symbol_data.index,symbol_data['LOWER_BB'],color='red',line_alpha=0.4)

    #Pipe to template
    script, div = components(graph)
    session.clear()
    return render_template('graph.html', script=script, div=div)

@app.route('/error/<error_text>')
def error(error_text):
    return render_template('error.html',error_text=error_text)

if __name__ == "__main__":
    app.run()
