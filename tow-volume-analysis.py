# %%
# use source .venv/bin/activate 
#Import packages
import streamlit as st
import pandas as pd
import numpy as np
import boto3
import plotly.graph_objects as go
from PIL import Image
from io import BytesIO
# %% USER INPUTS
# S3 info for retrieving static data and writing outputs
region_name = 'us-east-2'
bucket_name = 'wait-time-analysis'
bfpm_bucket_name = 'barge-freight-predictive-modeling'
static_data_folder = 'static_data/'
event_data_folder = 'event_testing/'
ais_event_folder = 'ais-dashboard-events/'
usda_data_folder = 'usda-barge-data/'
aws_access_key_id = st.secrets['access_key']
aws_secret_access_key = st.secrets['secret_key']
input_filename = 'RawAISDashboardData_current.csv'
output_filename = 'tow_key_locs_data'+pd.Timestamp.today().now().strftime('%Y-%m-%d')+'.csv'
river_stages_file = 'RiverStages_AllLowerMS.csv'
tow_benchmark_file = 'trip_data_Vicksburg_2023-08-22_1326.csv'
logo_file = 'Original Logo.png'
cache_time_to_live = 3600 # caching time to live in seconds for tow data call 
# parameters
start_date = '2023-05-08'
owner_type_dict = {'main line grain tows':['ACBL','ARTCO','INGRAM','MARQUETTE','SCF','WESTERN RIVERS'],\
                   'liquids tows':['BLESSEY','ENTERPRISE MARINE','GENESIS MARINE','GOLDING','KIRBY','LEBEOUF BROS.'\
                              'MAGNOLIA MARINE','MARATHON PETROLEUM']}
tow_min_hp_dict = {'main line grain tows': 6500, 'liquids tows': 1000, 'all tows': 1000}
collapsable_note_dict = {'main line grain tows': ('Includes tows over '+str(tow_min_hp_dict['main line grain tows']) +'hp from these companies as a proxy:  \n\n'+\
                                                    (', '.join(owner_type_dict['main line grain tows']))),
                        'liquids tows': ('Includes tows over '+str(tow_min_hp_dict['liquids tows']) +'hp from these companies as a proxy: \n\n'+\
                                                    (', '.join(owner_type_dict['liquids tows']))),
                        'all tows': ('Includes all tows over '+str(tow_min_hp_dict['liquids tows'])+'hp')
                        }
river_location_list = ['St. Louis','Wickliffe','Memphis','Vicksburg','Baton Rouge','New Orleans']
river_plot_aliases = {'St. Louis':' St. Louis (Mel Price Lower)', 
                      'Wickliffe': ' Cairo Area (Wickliffe, Ky)',
                      'Memphis': ' Memphis',
                      'Vicksburg': ' Vicksburg',
                      'Baton Rouge': ' Baton Rouge',
                      'New Orleans': ' New Orleans (Carrollton)'}
usda_data_types = ['unloads-nola', 'barge-counts']
usda_locks_fields = {'lock':['Ark Lock 1','Ohio Olmsted','Miss Locks 27'], #filtering criteria
                    'type':['Grain, Loaded'],
                    'direction':['Down']}
river_stage_bench_years = 10 # past years to consider for river stage benchmarking
tow_bench_years = 10 # past years to consider for freight rate benchmarking
river_benchmark_years = [2022, 2019] # years to add as low water and high water benchmarks, repspectively
usda_bench_years = 10 # past years to consider for usda data benchmarking
normal_stddev = 1.645 # standard deviations to consider "normal" (90%)
# %% Define functions
@st.cache_data(ttl=cache_time_to_live)
def dataframe_from_s3_csv(bucket_name,object_key):
    response = s3.get_object(Bucket = bucket_name,Key = object_key)
    df = pd.read_csv(response['Body'])
    return df
def weekly_benchmark_calc(weekly_data,end_time,years):
    # Create a week of year freight benchmark as the trailing average
    bench_calc = pd.DataFrame()
    bench_calc = weekly_data[((weekly_data['date']>=end_time-pd.Timedelta(years*365,"d"))
                                &(weekly_data['date']<=end_time))].copy()
    bench_calc.loc[:,'week_of_year'] = bench_calc['date'].apply(lambda x: x.weekofyear).values
    bench_calc = bench_calc.fillna(method = 'bfill')
    weekly_benchmark_mean = bench_calc.groupby(by=['week_of_year']).mean(numeric_only = True)
    weekly_benchmark_std = bench_calc.groupby(by=['week_of_year']).std(numeric_only = True)
    return weekly_benchmark_mean, weekly_benchmark_std
def apply_owner_hp_filter(data,owner_type,owner_type_dict,minimum_HP):
    if owner_type != 'all tows':
        data = data[(data['owner'].isin(owner_type_dict[owner_type]))]
    else:
        data = data
    data = data[(data['hp']>minimum_HP)]
    return data
# %% Establish s3 client
st.set_page_config(page_title=None, page_icon=None, layout="wide", initial_sidebar_state="auto", menu_items=None)
with st.spinner('Loading ...'):
    s3 = boto3.client('s3', aws_access_key_id=aws_access_key_id, aws_secret_access_key=aws_secret_access_key, region_name = region_name)
    # %% Initial Streamlit UI setup
    # logo for UI display
    response = s3.get_object(Bucket=bucket_name, Key=static_data_folder+logo_file)
    image_data = response['Body'].read()
    logo_image = Image.open(BytesIO(image_data))
    resize_logo = logo_image.resize([200,75])
    river_tab, transit_tab, eta_tab, unloads_tab, exports_tab = st.tabs(["MS river levels", "Lower MS tow transit", "Lower MS transit ETAs", "Grain barge lockings and unloads", "NOLA grain exports"])
    with river_tab:
        river_left_col, river_right_col = st.columns([0.3, 0.7],gap = 'medium')
        with river_left_col:
            st.image(resize_logo)
            # st.subheader('User selections')
            # river_location = st.selectbox('Select river location',river_location_list)
    with transit_tab:
        transit_left_col, transit_right_col = st.columns([0.3, 0.7],gap = 'medium')
        with transit_left_col:
            st.image(resize_logo)
            st.subheader('User selections')
            owner_type = st.selectbox('Select tow type category',['main line grain tows','liquids tows'])
            st.write(collapsable_note_dict[owner_type])
            tow_hp_lower_lim = tow_min_hp_dict[owner_type]
    #       tow_hp_lower_lim = st.slider('Select minimum tow horsepower',value = 6500,min_value = 2000, max_value = 10000, step = 100)
    with unloads_tab:
        unloads_left_col, unloads_right_col = st.columns([0.3, 0.7],gap = 'medium')
        with unloads_left_col:
            st.image(resize_logo)
    # %% ETL data from daily AIS dashboard monitoring
    tow_data = dataframe_from_s3_csv(bucket_name, ais_event_folder+input_filename)
    tow_data.columns = [col.lower() for col in tow_data.columns]
    # apply owner type and HP filter to data
    tow_data = apply_owner_hp_filter(tow_data,owner_type,owner_type_dict,tow_hp_lower_lim)
    tow_data['date'] = pd.to_datetime(tow_data['date'],errors='coerce')
    tow_data['week_of_year'] = tow_data['date'].apply(lambda x: x.weekofyear)
    tow_data['u/d'] = tow_data['u/d'].str.strip()
    start_date = tow_data['date'].min()
    end_date = tow_data['date'].max()
    # %% ETL river stage data
    river_data = dataframe_from_s3_csv(bucket_name,static_data_folder+river_stages_file)
    river_data = river_data.dropna(how = 'all')
    river_data = river_data.rename(columns = {'Date':'date'})
    for col in river_data.columns:
        if not(col == 'date'):
            river_data[col] = pd.to_numeric(river_data[col],errors='coerce')
    river_data['date'] = pd.to_datetime(river_data['date'])
    river_data = river_data.sort_values(by=['date'])
    river_data['week_of_year'] = river_data['date'].apply(lambda x: x.weekofyear)
    river_data_bench_mean, river_data_bench_std = weekly_benchmark_calc(river_data,pd.to_datetime(end_date),river_stage_bench_years)
    river_data = river_data.merge(river_data_bench_mean,how = 'left',on = 'week_of_year',suffixes = ('','_mean'))
    river_data = river_data.merge(river_data_bench_std,how = 'left',on = 'week_of_year',suffixes = ('','_std'))
    # %% ETL USDA data
    # load usda data from s3
    usda_data = dict()
    for data_type in usda_data_types:
        usda_data[data_type] = dataframe_from_s3_csv(bfpm_bucket_name,usda_data_folder+('usda-data-'+data_type+'.csv'))
    # usda locks data - formatting and filtering
    data = pd.DataFrame()
    data = usda_data['barge-counts'].copy()
    for col in usda_locks_fields.keys(): #filtering
        data = data[(data[col].isin(usda_locks_fields[col]))]
    data['date'] = pd.to_datetime(data['date'])
    usda_weekly_barge_counts = data[['date','barge_count']].groupby(['date']).sum().dropna()
    usda_weekly_barge_counts = usda_weekly_barge_counts.sort_values(by=['date']).reset_index()
    usda_weekly_barge_counts['week_of_year'] = usda_weekly_barge_counts['date'].apply(lambda x: x.weekofyear)
    usda_counts_mean, usda_counts_std = weekly_benchmark_calc(usda_weekly_barge_counts,pd.to_datetime(end_date),usda_bench_years)
    usda_weekly_barge_counts = usda_weekly_barge_counts.merge(usda_counts_mean,how = 'left',on = 'week_of_year',suffixes = ('','_mean'))
    usda_weekly_barge_counts = usda_weekly_barge_counts.merge(usda_counts_std,how = 'left',on = 'week_of_year',suffixes = ('','_std'))
    # usda barge unloads data - formatting and filtering
    data = pd.DataFrame()
    data = usda_data['unloads-nola'].copy()
    data['date'] = pd.to_datetime(data['date'])
    usda_weekly_unloads = data[['date','unloads']].dropna()
    usda_weekly_unloads['week_of_year'] = usda_weekly_unloads['date'].apply(lambda x: x.weekofyear)
    usda_unloads_mean, usda_unloads_std = weekly_benchmark_calc(usda_weekly_unloads,pd.to_datetime(end_date),usda_bench_years)
    usda_weekly_unloads = usda_weekly_unloads.merge(usda_unloads_mean,how = 'left',on = 'week_of_year',suffixes = ('','_mean'))
    usda_weekly_unloads = usda_weekly_unloads.merge(usda_unloads_std,how = 'left',on = 'week_of_year',suffixes = ('','_std')) 
    # %% ETL tow count benchmark data
    tow_bench_data = pd.DataFrame()
    tow_bench_data = dataframe_from_s3_csv(bucket_name,event_data_folder+tow_benchmark_file)
    tow_bench_data = tow_bench_data[['date','vessel_name','mmsi','location','direction','owner','HP']]
    tow_bench_data = tow_bench_data.drop_duplicates()
    tow_bench_data.columns = [col.lower() for col in tow_bench_data.columns]
    tow_bench_data = apply_owner_hp_filter(tow_bench_data,owner_type,owner_type_dict,tow_hp_lower_lim)
    tow_bench_data['date'] = pd.to_datetime(tow_bench_data['date'])
    tow_benchmark = pd.DataFrame()
    tow_benchmark = tow_bench_data[((tow_bench_data['date']>=pd.to_datetime(end_date)-pd.Timedelta(tow_bench_years*365,"d"))
                                &(tow_bench_data['date']<=pd.to_datetime(end_date)))].copy()
    tow_benchmark = tow_benchmark.fillna(0)
    tow_benchmark = tow_benchmark[['date','direction','location']].groupby(by=['date','direction']\
                                                                        ,as_index=False).count()
    tow_benchmark = tow_benchmark.rename(columns = {'location': 'event_count'})
    tow_benchmark = tow_benchmark.groupby(by = ['date','direction']).sum().reset_index()
    tow_benchmark['direction'] = tow_benchmark['direction'].replace(to_replace={25:1,15:2,20:np.nan,10:np.nan,0:np.nan})
    tow_benchmark = tow_benchmark.dropna()
    tow_benchmark['direction'] = tow_benchmark['direction'].astype('int')
    tow_benchmark = pd.get_dummies(tow_benchmark,columns=['direction']) #creates one-hot encoding for direction 1 or 2
    tow_benchmark['downstream_bench'] = tow_benchmark['event_count']*tow_benchmark['direction_1'] 
    tow_benchmark['upstream_bench'] = tow_benchmark['event_count']*tow_benchmark['direction_2']
    tow_benchmark = tow_benchmark.drop(columns = ['event_count','direction_1','direction_2']).groupby('date').sum()
    tow_benchmark = tow_benchmark.asfreq('D').fillna(0) #important to make zeroes where there are no counts
    tow_benchmark = tow_benchmark.rolling(7).sum().reset_index()
    tow_benchmark['week_of_year'] = tow_benchmark['date'].apply(lambda x: x.weekofyear)
    tow_benchmark = tow_benchmark.drop(columns=['date']).groupby(by=['week_of_year']).agg(['mean','std'])
    tow_benchmark.columns = ['_'.join(col) for col in tow_benchmark.columns.values] #flattens indexing to one level
    tow_benchmark = tow_benchmark.rename(columns = {'week_of_year_':'week_of_year'})
    # %% ETL barge unloads prediction results
    predict_date = usda_weekly_unloads['date'].max()+pd.Timedelta('7D')
    unloads_predictions = {'Week ending': str(predict_date.date()),
                           'Predicted value': 519,
                           'Last week ending': str(usda_weekly_unloads['date'].max().date()),
                           'Last week predicted': 425,
                           'Last week reported': 528,
                           'R2: BargeAI model': 0.82,
                           'R2: locks linreg': 0.13}
    # %% Daily tow count analysis
    daily_series = pd.date_range(start=start_date, end=end_date, freq='d')
    tows_daily = pd.DataFrame(index=daily_series)
    tows_daily['downstream'] = tow_data[(tow_data['u/d']=='D')][['date','u/d']]\
        .groupby(by='date').count().rename(columns={'u/d':'downstream_count'})
    tows_daily['upstream'] = tow_data[(tow_data['u/d']=='U')][['date','u/d']]\
        .groupby(by='date').count().rename(columns={'u/d':'upstream_count'})
    tows_daily = tows_daily.fillna(0)
    tows_daily = tows_daily.reset_index(names = 'date')
    tows_daily['week_of_year'] = tows_daily['date'].apply(lambda x: x.weekofyear)
    # %% Weekly tow analysis
    # seven day rolling, trailing sum
    tows_weekly = tows_daily.set_index('date').rolling(7).sum().reset_index()
    tows_weekly['week_of_year'] = tows_weekly['date'].apply(lambda x: x.weekofyear)
    tows_weekly['date'] = tows_weekly['date'].dt.date
    tows_weekly = tows_weekly.merge(tow_benchmark[['downstream_bench_mean','downstream_bench_std']],
                                how = 'left',on = 'week_of_year',) #add weekly benchmark for downstream
    tows_weekly = tows_weekly.merge(tow_benchmark[['upstream_bench_mean','upstream_bench_std']],\
                                how = 'left',on = 'week_of_year') #add weekly benchmark for downstream
    # weekly absolute values
    start_date = pd.to_datetime(start_date)
    frequency = 'W'  # 'W' stands for weekly
    num_weeks = np.floor(((pd.to_datetime(end_date) - start_date).days+7)/ 7) #add seven days to make sure first week is a full week
    date_series = pd.date_range(start=start_date+pd.Timedelta(7,'D'), periods=num_weeks, freq=frequency) #add seven days to make sure first week is a full week
    tows_weekly_abs = tows_daily.set_index('date').resample('W',convention='start').sum()
    tows_weekly_abs = tows_weekly_abs[tows_weekly_abs.index.isin(date_series)].reset_index()
    tows_weekly_abs = tows_weekly_abs.rename(columns={'date':'week_ending'})
    tows_weekly_abs['week_ending'] = tows_weekly_abs['week_ending'].dt.date
    tows_weekly_abs = tows_weekly_abs.set_index('week_ending').drop(columns = 'week_of_year')
    # %% Create plot for weekly results
    #downstream tow transit
    fig_d = go.Figure()
    dates_rev = tows_weekly.loc[::-1,'date']
    downstream_norm_upper = tows_weekly['downstream_bench_mean']+normal_stddev*tows_weekly['downstream_bench_std']
    downstream_norm_lower = tows_weekly['downstream_bench_mean']-normal_stddev*tows_weekly['downstream_bench_std']
    downstream_norm_lower = downstream_norm_lower[::-1]

    fig_d.add_trace(go.Scatter(
        x=pd.concat([tows_weekly['date'],dates_rev]),
        y= pd.concat([downstream_norm_upper,downstream_norm_lower]), 
        fill='toself',
        fillcolor='rgba(70, 130, 180,0.3)',
        line_color='rgba(255,255,255,0)',
        showlegend=True,
        name='Normal range',
    ))
    fig_d.add_trace(go.Scatter(
        x=tows_weekly['date'],
        y= tows_weekly['downstream'], 
        line_color='goldenrod',
        showlegend=True,
        name='Measured tow count',
    ))
    fig_d.update_layout(
        plot_bgcolor='black', 
        paper_bgcolor='black',
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=False),
        font=dict(color='#c0c0c0'),
        title = ('Downstream MS River Tow Traffic (Vicksburg Proxy)')
    )
    fig_d.update_yaxes(
        zeroline=False,
        linecolor = 'silver',
        title_text = 'Tows moving toward NOLA <br> (trailing seven days)'  
    )
    fig_d.update_xaxes(
        zerolinecolor='#4682b4',
        zerolinewidth = 1,
        linecolor = 'silver',
        range = [start_date, end_date]  
    )

    #upstream tow transit
    fig_u = go.Figure()
    upstream_norm_upper = tows_weekly['upstream_bench_mean']+normal_stddev*tows_weekly['upstream_bench_std']
    upstream_norm_lower = tows_weekly['upstream_bench_mean']-normal_stddev*tows_weekly['upstream_bench_std']
    upstream_norm_lower = upstream_norm_lower[::-1] #this is a facet to create the filled line plot

    fig_u.add_trace(go.Scatter(
        x=pd.concat([tows_weekly['date'],dates_rev]),
        y= pd.concat([upstream_norm_upper,upstream_norm_lower]), 
        fill='toself',
        fillcolor='rgba(70, 130, 180,0.3)',
        line_color='rgba(255,255,255,0)',
        showlegend=True,
        name='Normal range',
    ))
    fig_u.add_trace(go.Scatter(
        x=tows_weekly['date'],
        y= tows_weekly['upstream'], 
        line_color='salmon',
        showlegend=True,
        name='Measured tow count',
    ))
    fig_u.update_layout(
        plot_bgcolor='black', 
        paper_bgcolor='black',
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=False),
        font=dict(color='#c0c0c0'),
        title = ('Upstream MS River Tow Traffic (Vicksburg Proxy)')
    )
    fig_u.update_yaxes(
        zeroline=False,
        linecolor = 'silver',
        title_text = 'Tows moving North <br> (trailing seven days)'  
    )
    fig_u.update_xaxes(
        zerolinecolor='#4682b4',
        zerolinewidth = 1,
        linecolor = 'silver',
        range = [start_date, end_date]  
    )
    #usda locks data
    locks_data_plot = usda_weekly_barge_counts[(usda_weekly_barge_counts['date']>start_date)]
    lock_dates_rev = locks_data_plot.loc[::-1,'date']
    locks_norm_upper = locks_data_plot['barge_count_mean']+normal_stddev*locks_data_plot['barge_count_std']
    locks_norm_lower = locks_data_plot['barge_count_mean']-normal_stddev*locks_data_plot['barge_count_std']
    locks_norm_lower = locks_norm_lower.clip(0)
    locks_norm_lower = locks_norm_lower[::-1] #this is a facet to create the filled line plot
    fig_locks = go.Figure()
    fig_locks.add_trace(go.Scatter(
        x= pd.concat([locks_data_plot['date'],lock_dates_rev ]),
        y= pd.concat([locks_norm_upper,locks_norm_lower]), 
        fill='toself',
        fillcolor='rgba(70, 130, 180,0.3)',
        line_color='rgba(255,255,255,0)',
        showlegend=True,
        name='Normal range',
    ))
    fig_locks.add_trace(go.Scatter(
        x= locks_data_plot['date'],
        y= locks_data_plot['barge_count'], 
        mode='lines+markers',  
        line_color='goldenrod',
        marker=dict(  
            symbol='circle',  
            size=8,  
            color='goldenrod',  
            line=dict(  
                color='black',  
                width=1, 
            ),
        ),
        showlegend=True,
        name='Locking grain barges',
    ))
    fig_locks.update_layout(
        plot_bgcolor='black', 
        paper_bgcolor='black',
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=False),
        font=dict(color='#c0c0c0'),
        title = ('USDA Downstream Locking Grain Barges')
    )
    fig_locks.update_yaxes(
        zeroline=False,
        linecolor = 'silver',
        title_text = 'Barge count'  
    )
    fig_locks.update_xaxes(
        zerolinecolor='#4682b4',
        zerolinewidth = 1,
        linecolor = 'silver',
        range = [start_date, end_date]  
    )
    #usda unloads data
    unloads_data_plot = usda_weekly_unloads[(usda_weekly_unloads['date']>start_date)]
    unloads_dates_rev = unloads_data_plot.loc[::-1,'date']
    unloads_norm_upper = unloads_data_plot['unloads_mean']+normal_stddev*unloads_data_plot['unloads_std']
    unloads_norm_lower = unloads_data_plot['unloads_mean']-normal_stddev*unloads_data_plot['unloads_std']
    unloads_norm_lower = unloads_norm_lower.clip(0)
    unloads_norm_lower = unloads_norm_lower[::-1] #this is a facet to create the filled line plot
    fig_unloads = go.Figure()
    fig_unloads.add_trace(go.Scatter(
        x=pd.concat([unloads_data_plot['date'],unloads_dates_rev ]),
        y= pd.concat([unloads_norm_upper,unloads_norm_lower]), 
        fill='toself',
        fillcolor='rgba(70, 130, 180,0.3)',
        line_color='rgba(255,255,255,0)',
        showlegend=True,
        name='Normal range',
    ))
    fig_unloads.add_trace(go.Scatter(
        x=unloads_data_plot['date'],
        y= unloads_data_plot['unloads'], 
        mode='lines+markers',  
        line_color='salmon',
        marker=dict(  # Define marker properties
            symbol='circle',  # Set the marker symbol (you can choose other shapes)
            size=8,  # Set the marker size
            color='salmon',  # Set the marker color
            line=dict(  # Define marker border properties
                color='black',  # Set the border color
                width=1,  # Set the border width
            ),
        ),
        showlegend=True,
        name='NOLA barge unloads',
    ))
    fig_unloads.update_layout(
        plot_bgcolor='black', 
        paper_bgcolor='black',
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=False),
        font=dict(color='#c0c0c0'),
        title = ('USDA NOLA Grain Barge Unloads')
    )
    fig_unloads.update_yaxes(
        zeroline=False,
        linecolor = 'silver',
        title_text = 'Barge count'  
    )
    fig_unloads.update_xaxes(
        zerolinecolor='#4682b4',
        zerolinewidth = 1,
        linecolor = 'silver',
        range = [start_date, end_date]  
    )
    # %% Create weekly table for UI display 
    tows_weekly_display = tows_weekly.set_index('date')\
        [['week_of_year','downstream','downstream_bench_mean','upstream','upstream_bench_mean']]
    tows_weekly_display = tows_weekly_display.rename(columns = {'downstream_bench_mean':'downstream benchmark',\
                                                                'upstream_bench_mean':'upstream benchmark'})
# %% Streamlit UI
with transit_tab:
    with transit_right_col:
        st.title('BargeAI:tm: Lower MS River Intel')
        st.subheader('Tow Traffic Charts')
        st.plotly_chart(fig_d)
        st.plotly_chart(fig_u)
        st.divider()
        st.subheader('Weekly Tow Count Data')
        st.write('The tow counts in the table below correspond to the absolute totals for the week.')
        st.write(tows_weekly_abs.round(0))
        st.download_button('Download data', tows_weekly_abs.to_csv(index = False),file_name = 'tow_counts.csv') 
with river_tab:
    with river_right_col:
        st.title('BargeAI:tm: Lower MS River Intel')
        st.subheader('River Levels - Interactive Plots')
        river_data_plot = river_data[(river_data['date']>start_date)]
        date_today = pd.Timestamp.today()
        start_date_low = pd.Timestamp(year = river_benchmark_years[0],month=start_date.month,day = start_date.day)
        end_date_low = pd.Timestamp(year = river_benchmark_years[0],month=date_today.month,day = date_today.day)
        start_date_hi = pd.Timestamp(year = river_benchmark_years[1],month=start_date.month,day = start_date.day)
        end_date_hi = pd.Timestamp(year = river_benchmark_years[1],month=date_today.month,day = date_today.day)
        river_data_low = river_data[((river_data['date']>start_date_low)&(river_data['date']<=end_date_low))]
        river_data_hi = river_data[((river_data['date']>start_date_hi)&(river_data['date']<=end_date_hi))]
        offset_years_low = start_date.year - start_date_low.year 
        offset_years_hi = start_date.year - start_date_hi.year 
        plot_date_low = river_data_low['date']+pd.DateOffset(years=offset_years_low)
        plot_date_hi = river_data_hi['date']+pd.DateOffset(years=offset_years_hi)

        for river_location in river_location_list:
            fig_r = go.Figure()
            river_dates_rev = river_data_plot.loc[::-1,'date']
            river_norm_upper = river_data_plot[river_location+'_mean']+normal_stddev*river_data_plot[river_location+'_std']
            river_norm_lower = river_data_plot[river_location+'_mean']-normal_stddev*river_data_plot[river_location+'_std']
            river_norm_lower = river_norm_lower[::-1] #this is a facet to create the filled line plot
            
            fig_r.add_trace(go.Scatter(
                x = pd.concat([river_data_plot['date'],river_dates_rev]),
                y = pd.concat([river_norm_upper,river_norm_lower]), 
                fill='toself',
                fillcolor='rgba(70, 130, 180,0.4)',
                line_color='rgba(255,255,255,0)',
                showlegend=True,
                name='Normal range',
            ))
            fig_r.add_trace(go.Scatter(
                x = river_data_plot['date'],
                y = river_data_plot[river_location], 
                line_color='goldenrod',
                showlegend=True,
                name='Current river stage',
            ))
            fig_r.add_trace(go.Scatter(
                x = plot_date_low,
                y = river_data_low[river_location], 
                line=dict(
                    color='cyan',
                    width = 1,  # Set the width to your desired value
                ),
                line_dash='dash',
                showlegend=True,
                name='Low water ref '+str(river_benchmark_years[0]),
            ))

            fig_r.add_trace(go.Scatter(
                x = plot_date_hi,
                y = river_data_hi[river_location], 
                line=dict(
                    color='paleturquoise',
                    width = 1,  # Set the width to your desired value
                ),
                line_dash='dash',
                showlegend=True,
                name='High water ref '+str(river_benchmark_years[1]),
            ))
            fig_r.update_layout(
                plot_bgcolor='black', 
                paper_bgcolor='black',
                xaxis=dict(showgrid=False),
                yaxis=dict(showgrid=False),
                font=dict(color='#c0c0c0'),
                title = (river_plot_aliases[river_location]+' River Stage')
            )
            fig_r.update_yaxes(
                zeroline=False,
                linecolor = 'silver',
                title_text = 'River stage (ft)'  
            )
            fig_r.update_xaxes(
                zerolinecolor='#4682b4',
                zerolinewidth = 1,
                linecolor = 'silver'  
            )
            st.plotly_chart(fig_r)
with unloads_tab:
    with unloads_right_col:
        st.title('BargeAI:tm: Lower MS River Intel')
        st.subheader('USDA Barge Unloads Predictions (Beta)')
        st.dataframe(pd.DataFrame(unloads_predictions,index = [0]),hide_index=True)
        st.divider()
        st.subheader('USDA Locks Data and NOLA Unloads - Downstream Grain Barges')
        st.plotly_chart(fig_locks)
        st.plotly_chart(fig_unloads)
with eta_tab:
    st.header('Coming Soon!')
with exports_tab:
    st.header('Coming Soon!')