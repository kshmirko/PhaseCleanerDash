#-*- encoding: utf-8 -*-
import dash
from dash.dependencies import Input, Output, State
import dash_html_components as html
import dash_core_components as dcc
from datetime import datetime as dt
from datetime import date as dtt
import dash_auth
#import dash_bootstrap_components as dbc
#import dash_table_experiments as dte
import dash_table as dte

from calc_module.distrtype import *
from calc_module.mieutils import CalcScatteringProps1, MAX_DEG, MAX_LEG, AotRayleigh, PrepareScatFile
import numpy as np
from calc_module.rt3code import *

import os
import base64
import io
import numpy as np
from model import db_proxy, initialize, Measurements, PhaseFunction, DirectParameters
import plotly.graph_objs as go


 
VALID_USERNAME_PASSWORD_PAIRS = [
  ['kshmirko','2332361'],
  ['anpavlov', '123123'],
]

UPLOAD_TO = 'measurements/'
UPLOAD_PHASES = 'phases/'
DIRECT_PARAMS = 'direct_params/'
APP_NAME = 'Phase Clener'


external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

from flask import Flask
flaskapp = Flask(__name__)

wsgi_app = flaskapp.wsgi_app

app = dash.Dash(__name__, server=flaskapp, url_base_pathname='/', external_stylesheets=external_stylesheets)
auth = dash_auth.BasicAuth(
    app,
    VALID_USERNAME_PASSWORD_PAIRS
)

app.title = APP_NAME

app.layout = html.Div([
    dcc.Tabs(id="tabs", value='tab-meas', children=[
        dcc.Tab(label='1. Загрузка данных измерений на сервер', value='tab-meas', children=[
            html.Div([
                html.H3('Загрузка данных измерений'),
                html.Div([
                    html.Div([
                        html.Label('Дата измерений: '),
                    ], style={'width':'200pt', 'display': 'inline-block'}),
                    html.Div([
                        dcc.DatePickerSingle(
                            id='measurement-date',
                            min_date_allowed=dtt(2017, 1, 1),
                            max_date_allowed=dtt(2020, 12, 31),
                            initial_visible_month=dtt(2018, 11, 5).month,
                            date=dtt(2018, 11, 29),
                        )],style={'display': 'inline-block'})
                ]),
                html.Div([
                    html.Div([
                        html.Label('Время измерений: '),
                    ], style={'width':'200pt','display': 'inline-block'}),
                    html.Div([
                        dcc.Input(
                            id='measurement-time',
                            type='text',
                            placeholder='HH:MM:SS',
                        )],style={'display': 'inline-block'})
                ]),
                dcc.Upload(
                        id='upload-data',
                        children=html.Div([
                            'Перенесите файл или ',
                            html.A('Выберите его')
                        ]),
                        style={
                            'width': '100%',
                            'height': '60px',
                            'lineHeight': '60px',
                            'borderWidth': '1px',
                            'borderStyle': 'dashed',
                            'borderRadius': '5px',
                            'textAlign': 'center',
                            'margin': '10px'
                        },
                        # Allow multiple files to be uploaded
                        multiple=False
                    ),
                    html.Div(id='output-data-upload'),
            ]),
            html.Button('Загрузить', id='button'),
            html.Div(id='do-upload-button'),
        ]),
        dcc.Tab(label='2. Создание фазовой функции', value='tab-phase', children=[
          html.Div([
            html.H3("Выберите параметры фазовой функции"),
            html.Table([
              html.Tr([
                html.Td([
                  html.Label("R0:", style={'display':'inline'})
                ]),
                html.Td([
                  dcc.Input(id="phase-r0", type="number", step=0.05, debounce=True, 
                          style={'display':'inline'}, min=0.1, max=0.5, value=0.1),
                ]),
                html.Td([
                  html.Label("R1:", style={'display':'inline'}),
                ]),
                html.Td([
                  dcc.Input(id="phase-r1", type="number", step=0.05, debounce=True, 
                          style={'display':'inline'}, min=0.55, max=1.0, value=1.0),
                ]),
              ]),
              html.Tr([
                html.Td([
                  html.Label("Real(midx):", style={'display':'inline'})
                ]),
                html.Td([
                  dcc.Input(id="phase-mre", type="number", step=0.005, debounce=True, 
                            style={'display':'inline'}, min=1.35, max=1.85, value=1.5),
                ]),
                html.Td([
                  html.Label("Imag(midx):", style={'display':'inline'}),
                ]),
                html.Td([
                  dcc.Input(id="phase-mim", type="number", step=0.001, debounce=True, 
                            style={'display':'inline'}, min=0.0, max=0.5, value=0.0),
                ]),
              ]),
              html.Tr([
                html.Td([
                  html.Label("Модель частиц:", style={'display':'inline'})
                ]),
                html.Td([
                  dcc.Dropdown(id="phase-modeltype", options=[
                      {'label':'Сферы', 'value':'1'},
                      {'label':'Агломераты', 'value':'2'},
                    ], value='1')
                ]),
                html.Td([
                  html.Label("Длина волны:", style={'display':'inline'})
                ]),
                html.Td([
                  dcc.Input(id="phase-wavelen", type="number", step=0.001, debounce=True, 
                              style={'display':'inline'}, min=0.355, max=1.064, value=0.87)
                  ]),
                ]),
              html.Tr([
                html.Td([
                  html.Label("Тип распределения:", style={'display':'inline'})
                ]),
                html.Td([
                  dcc.Dropdown(id="phase-distrtype", options=[
                    {'label':'Степенное', 'value':'1'},
                    {'label':'Логнормальное', 'value':'2'},
                    {'label':'Двумодальное логнормальное', 'value':'5'},
                  ], value='1'),
                ], colSpan='3'),
              ]),
            ], style={'width':'100%'}),
            html.Br(),
            html.Div(id='params', children=[
              html.Div(children=[
                html.Label(["P",html.Sub("1"),":"], style={'display':'inline-block', 'width':'100pt'}),
                dcc.Input(id="phase-p1", type="number", debounce=True, style={'display':'inline'}, 
                  value=-4, min=-10, max=10, step=0.1),
                ]),
                html.Div(children=[
                  html.Label(["P",html.Sub("2"),":"], style={'display':'inline-block', 'width':'100pt'}),
                  dcc.Input(id="phase-p2", type="number", debounce=True, style={'display':'inline'}, 
                    value=2.0, min=1.1, max=2.8, step=0.1),
                ]),
                html.Div(children=[
                  html.Label(["P",html.Sub("3"),":"], style={'display':'inline-block', 'width':'100pt'}),
                  dcc.Input(id="phase-p3", type="number", debounce=True, style={'display':'inline'}, 
                    value=0.8, min=0.31, max=2.0, step=0.1),
                ]),
                html.Div(children=[
                  html.Label(["P",html.Sub("4"),":"], style={'display':'inline-block', 'width':'100pt'}),
                  dcc.Input(id="phase-p4", type="number", debounce=True, style={'display':'inline'}, 
                    value=2.1, min=1.1, max=2.8, step=0.1),
                ]),
                html.Div(children=[
                  html.Label(["P",html.Sub("5"),":"], style={'display':'inline-block', 'width':'100pt'}),
                  dcc.Input(id="phase-p5", type="number", debounce=True, style={'display':'inline'}, 
                    value=0.99, min=0.0001, max=0.9999, step=0.0001),
                ]),
            ]),
            html.Hr(),
            html.Div(id='phase-result'),
            html.Button('Рассчитать', id='button-calc-phase', style={'class':'btn btn-primary'}),
          ]),
        ]),
        dcc.Tab(label='3. Выполнение расчетов прямого моделирования', value='tab-direct-calc', children=[
          html.Div([
            html.H3('Настройка прямого моделирования'),
            html.Table([
              html.Tr([
                html.Td([html.Label("Обновить поля", style={'display':'inline-block'})], style={'width':'30%'}),
                html.Td([html.Button("Обновить", id='update-fields')], style={'width':'70%'})
              ], ),
              html.Tr([
                html.Td([html.Label("Измерение", style={'display':'inline-block'})], style={'width':'30%'}),
                html.Td([dcc.Dropdown(id='select-meas', options=[], style={'display':'inline-block', 'width':'100%'})], style={'width':'70%'})
              ]),
              html.Tr([
                html.Td([html.Label('Фазовая функция', style={'display':'inline-block'})], style={'width':'30%'}),
                html.Td([dcc.Dropdown(id='select-phase', options=[], style={'display':'inline-block', 'width':'100%'})], style={'width':'70%'})
              ]),
              html.Tr([
                html.Td([html.Label("Зенитный угол:", style={'display':'inline-block'})], style={'width':'30%'}),
                html.Td([dcc.Input(id='zenithAngle', type='number', style={'display':'inline-block', 'width':'100%'}, value=65.0, min=0, max=75, step=0.5)], style={'width':'70%'}),
              ]),
              html.Tr([
                html.Td([html.Label("Аэрозольная отпическая толща:", style={'display':'inline-block'})], style={'width':'30%'}),
                html.Td([dcc.Input(id='aerosolOpticalDepth', type='number', style={'display':'inline-block', 'width':'100%'}, value=0.1, min=0.0, max=2.0, step=0.005)], style={'width':'70%'}),
              ]),
              html.Tr([
                html.Td([html.Label("Альбедо подстилающей поверхности:", style={'display':'inline-block'})], style={'width':'30%'}),
                html.Td([dcc.Input(id='groundAlbedo', type='number', style={'display':'inline-block', 'width':'100%'}, value=0.06, min=0.0, max=1.0, step=0.01)], style={'width':'70%'}),
              ]),
            ], style={'width':'100%'})
          ]),
          html.Hr(),
          html.Div(id='direct-result'),
          html.Button('Рассчитать', id='button-calc-direct'),
        ]),
        dcc.Tab(label="4. Просмотр", value='tab-view-calc', children=[
          html.Div([
            html.H3('Просмотр результатов моделирования для выбранного натурного измерения'),
            html.Table([
              html.Tr([
                html.Td([
                  html.Button("Обновить", id='refresh-result-list')
                ], colSpan=2, style={'textAlign':'right'})
              ]),
              html.Tr([
                html.Td([html.Label("Выберите измерение:", style={'display':'inline-block', 'width':'100%'})], style={'width':'30%'}),
                html.Td([dcc.Dropdown(id='select-disp-meas', options=[], style={'display':'inline-block', 'width':'100%'}, multi=False, clearable=False)], style={'width':'69%'}),
              ], style={'width':'100%'}),
              html.Tr([
                html.Td([html.Label("Выберите результат моделирования:", style={'display':'inline-block', 'width':'100%'})], style={'width':'30%'}),
                html.Td([dcc.Dropdown(id='select-disp-direct', options=[], style={'display':'inline-block', 'width':'100%', 'height': '30px',}, multi=True, clearable=False)], style={'width':'69%', }),
              ]),
            ], style={'width':'100%'}),
            dte.DataTable(
              columns=[{'name':'ID', 'id':0, 'deletable':False},
                    {'name':'PhaseFunction_descr', 'id':1, 'deletable':False},
                    {'name':'Зенитный угол солнца', 'id':2, 'deletable':False},
                    {'name':'Аэрозольная оптическая толща', 'id':3, 'deletable':False},
                    {'name':'Альбедо поверхности', 'id':4, 'deletable':False},
                  ],
              data=[{}], # initialise the rows
              row_selectable='multi',
              filtering=False,
              sorting=False,
              editable=False,
              id='result-table'
            ),
            html.Hr(),
            html.Div(id='view-result'),
            html.Button('Посмотреть', id='button-view-direct-calc'),
          ]),
        ]),
    ]),
    html.Div(id='tabs-content'),
])


def parse_contents(contents, filename, date):
  content_type, content_string = contents.split(',')
  decoded = base64.b64decode(content_string)
  
  try:
    df = np.loadtxt(io.StringIO(decoded.decode('utf-8')))
    
  except Exception as e:
    print(e)
    return None
  return df
  

@app.callback(Output('output-data-upload', 'children'),
              [Input('button', 'n_clicks')],
              [State('upload-data', 'contents'),
               State('measurement-date', 'date'),
               State('measurement-time', 'value'),
               State('upload-data', 'filename'),
               State('upload-data', 'last_modified')])
def add_measurements_to_db(n_clicks, list_of_contents, meas_date, meas_time, list_of_names, list_of_dates):
  if n_clicks!=None:
    if (not meas_date  is None) and (not meas_time  is None):
      date = dt.strptime(meas_date, '%Y-%m-%d').date()
      time = dt.strptime(meas_time, "%H:%M:%S").time()
      datetime = dt.combine(date, time)#dt(date.year, date.month, date.day, time.hour, time.minute, time.second)
      print(dt.combine(date, time))
      if list_of_contents is not None:
        data = parse_contents(list_of_contents, list_of_names, list_of_dates)
        
        query = (Measurements.select().where(
          Measurements.datetime==datetime,
        ))
        
        if len(query)==0:
          timepath="%04d-%02d-%02d/%02d_%02d"%(datetime.year, datetime.month, datetime.day, datetime.hour, datetime.minute)
          filepath = os.path.join(UPLOAD_TO, timepath)
          os.makedirs(filepath, exist_ok=True)
          filepath = os.path.join(filepath, 'meas.npz')
          np.savez(filepath, data=data)
          
          if data is not None:
            ret = html.Div([
              html.Hr(),
              html.Div('Таблица измерений'),
              html.Div(style={'align':'center'},children=[
                html.Table(id='meas-table', children=[
                  html.Tr([html.Th('Angle'), html.Th('Q-Intensity'), html.Th('I-Intensity')])]+
      
                  [html.Tr([html.Td(f"{data[i,j]:.3f}") for j in range(data.shape[1])]) for i in range(5)],
                ),
              ])  
            ])
          item = Measurements(datetime=datetime, filepath=filepath)
          item.save()
        else:
          item = query.get()
          ret = html.P(f"Такие данные уже есть в таблице {item.datetime}")
          
      else:
        ret = html.Div("Файл не выбран")
    else: 
      ret = html.Div("Поля 'Дата' или 'Время' содержат пустые значения")
    return ret


  
@app.callback(
  Output('phase-result','children'),
  [Input('button-calc-phase', 'n_clicks')],
  [State('phase-r0', 'value'),
  State('phase-r1', 'value'),
  State('phase-mre', 'value'),
  State('phase-mim', 'value'),
  State('phase-modeltype', 'value'),
  State('phase-wavelen', 'value'),
  State('phase-distrtype', 'value'),
  State('phase-p1', 'value'),
  State('phase-p2', 'value'),
  State('phase-p3', 'value'),
  State('phase-p4', 'value'),
  State('phase-p5', 'value')],
  )
def submit_phase_function(n_clicks, r0, r1, mre, mim, modeltype, wavelen, distrtype, p1, p2, p3, p4, p5):
  # преобразуем строки в цклые числа
  distrtype=int(distrtype)
  modeltype=int(modeltype)
  if n_clicks!=None:
    query = (PhaseFunction.select().where((PhaseFunction.R0==r0)&
                                          (PhaseFunction.R1==r1)&
                                          (PhaseFunction.Mre==mre)&
                                          (PhaseFunction.Mim==mim)&
                                          (PhaseFunction.particlesType==modeltype)&
                                          (PhaseFunction.Wl==wavelen)&
                                          (PhaseFunction.distrType==distrtype)&
                                          (PhaseFunction.p1==p1)&
                                          (PhaseFunction.p2==p2)&
                                          (PhaseFunction.p3==p3)&
                                          (PhaseFunction.p4==p4)&
                                          (PhaseFunction.p5==p5)))
    if len(query)!=0:
      return html.P("В базе данный уже есть запись с такими параметрами")
    else:
      
      F=None
      
      if distrtype==1:
        F = PowerLaw(r0, r1, p1)
        p2=p3=p4=p5=0.0
      elif distrtype==2:
        F = LogNormal(r0, r1, p1, p2)
        p3=p4=p5=0.0
      elif distrtype==5:
        F = LogNormal2(r0, r1, p1, p2, p3, p4, p5)
      
      if modeltype==0:
        #чферические частицы
        midx = complex(mre, -mim)
        xi, _ = np.polynomial.legendre.leggauss(MAX_DEG)
        EvA, _, _, _, ssa = CalcScatteringProps1(F, wavelen, midx, xi, MAX_LEG)
        EvA = EvA / EvA[0,0]
      elif modeltype==1:
        # агломераты
        pass
      dtnow = dt.now()
      timepath="%04d-%02d-%02d/%02d_%02d"%(dtnow.year, dtnow.month, dtnow.day, dtnow.hour, dtnow.minute)
      filepath = os.path.join(UPLOAD_PHASES, timepath)
      os.makedirs(filepath, exist_ok=True)
      filepath = os.path.join(filepath, 'phase_legendre.npz')
      np.savez(filepath, EvA=EvA, ssa=ssa)
      
      item = PhaseFunction.create(R0=r0, R1=r1, Mre=mre, Mim=mim, particlesType=modeltype, 
        Wl=wavelen, distrType=distrtype, p1=p1, p2=p2, p3=p3, p4=p4, p5=p5, matrix=filepath)
      item.save()
      I = np.polynomial.legendre.legval(xi, EvA[:,0])
      Q = np.polynomial.legendre.legval(xi, EvA[:,1])
      return [
        html.P("Запись о фазофой функции успешно добавлена в базу данных"),
        html.Div([
          dcc.Graph(id='phase-function-i',
            figure={
              'data': [
                go.Scatter(
                  x=xi,
                  y=I,
                )
              ],
              'layout': go.Layout(
                xaxis={'title':'μ'},
                yaxis={'type':'log', 'title':'I-Интенсивность (Вт/(м2*ср*мкм))'},
                margin={'l': 40, 'b': 40, 't': 10, 'r': 10},
                legend={'x': 0, 'y': 1},
                hovermode='closest'
              )
            }
          )
        ], style={'width':'32%', 'display':'inline-block'}),
        html.Div([
          dcc.Graph(id='phase-function-q',
            figure={
              'data': [
                go.Scatter(
                  x=xi,
                  y=Q,
                )
              ],
              'layout': go.Layout(
                xaxis={'title':'μ'},
                yaxis={'title':'Q-Интенсивность (Вт/(м2*ср*мкм))'},
                margin={'l': 40, 'b': 40, 't': 10, 'r': 10},
                legend={'x': 0, 'y': 1},
                hovermode='closest'
              )
            }
          )
        ], style={'width':'32%', 'display':'inline-block'}),
        html.Div([
          dcc.Graph(id='phase-function-Pol',
            figure={
              'data': [
                go.Scatter(
                  x=xi,
                  y=-Q/I*100.0,
                )
              ],
              'layout': go.Layout(
                xaxis={'title':'μ'},
                yaxis={'title':'Степень линейной поляризации, %'},
                margin={'l': 40, 'b': 40, 't': 10, 'r': 10},
                legend={'x': 0, 'y': 1},
                hovermode='closest'
              )
            }
          )
        ], style={'width':'32%', 'display':'inline-block'}),
      ]
  pass
    

@app.callback(Output('select-meas','options'),
              [Input('update-fields', 'n_clicks')],
              )
def update_meas_fields(n_clicks):
  ret = []
  if not (n_clicks  is None):
    query=(Measurements.select())
    for item in query:
      ret.append({'label':item.datetime, 'value':item.id})
  return ret
  
@app.callback(Output('select-phase','options'),
              [Input('update-fields', 'n_clicks')],
              )
def update_phase_fields(n_clicks):
  ret = []
  if not (n_clicks  is None):
    query=(PhaseFunction.select())
    for item in query:
      ret.append({'label':str(item), 'value':item.id})
  return ret

@app.callback(Output('direct-result', 'children'),
              [Input('button-calc-direct','n_clicks')],
              [State('select-meas', 'value'),
              State('select-phase', 'value'),
              State('zenithAngle', 'value'),
              State('aerosolOpticalDepth', 'value'),
              State('groundAlbedo', 'value')])
def calc_direct(n_clicks, meas_id, phase_id, zenAng, aerDepth, grdAlb):
  if n_clicks is None:
    return
  ret = []
  if meas_id is None:
    ret.append(html.P('Не выбраны измерения'))
  if phase_id is None:
    ret.append(html.P('Не выбрана фазовая функция'))
  if len(ret) > 0:
    return ret
  meas_id = int(meas_id)
  phase_id = int(phase_id)
  
  meas_item = Measurements.get(Measurements.id==meas_id)
  phase_item = PhaseFunction.get(PhaseFunction.id==phase_id)
  
  #проверяем, были ли такие расчеты
  query = (DirectParameters.select().where((DirectParameters.measType==meas_item)&
                                            (DirectParameters.phaseFunction==phase_item)&
                                            (DirectParameters.zenithAngle==zenAng)&
                                            (DirectParameters.aerosolOpticalDepth==aerDepth)&
                                            (DirectParameters.groundAlbedo==grdAlb)))
  if len(query)>0:
    return html.P('Расчеты для данных параметров были выполнены ранее')
  
  F = np.load(phase_item.matrix)
  EvA = F['EvA'][...]
  ssa = F['ssa'][...]
  F.close()
  taum = AotRayleigh(phase_item.Wl)
  mu = np.cos(np.deg2rad(zenAng))
  transmittance = np.exp(-(taum+aerDepth)/mu)
      
  meas_data = np.load(meas_item.filepath)
  Ameas = meas_data["data"][:,0]
  Qmeas = meas_data["data"][:,1]
  Imeas = meas_data["data"][:,2]
  
  meas_data.close()
  EvA = EvA/EvA[0,0]
  PrepareScatFile(EvA, aerDepth, ssa, phase_item.Wl)
      
  flux = 1.0/mu
  rt3app = RtCode(solar_zenith=zenAng, direct_flux=flux,\
          ground_albedo=grdAlb)
  rt3app.run()
  rt3app.I/=transmittance
  rt3app.Q/=transmittance
  
  datetime = dt.now()
  timepath="%04d-%02d-%02d/%02d_%02d_%02d"%(datetime.year, datetime.month, 
    datetime.day, datetime.hour, datetime.minute, datetime.second)
  filepath = os.path.join(DIRECT_PARAMS, timepath)
  os.makedirs(filepath, exist_ok=True)
  filepath = os.path.join(filepath, 'result.npz')
  np.savez(filepath, Ang=rt3app.mu, I=rt3app.I, Q=rt3app.Q)
    
  direct_item = DirectParameters(measType=meas_item, phaseFunction=phase_id, zenithAngle=zenAng,
    aerosolOpticalDepth=aerDepth, groundAlbedo=grdAlb, filepath=filepath)
  direct_item.save()
  #print(meas_item, phase_id, zenAng, aerDepth, grdAlb)
  ret = [
    html.Div([
      dcc.Graph(id='ph-fun-i',
        figure={
          'data': 
            [go.Scatter(
              x=np.rad2deg(np.arccos(rt3app.mu)),
              y=rt3app.I,
              name=f'ID=Model',
            ),
            go.Scatter(
              x=Ameas,
              y=Imeas,
              mode = 'lines+markers',
              name='measurements'
            )]
          ,
          'layout': go.Layout(
            xaxis={'title':'Угол рассеяния (град.)'},
            yaxis={'type':'log', 'title':'I-Интенсивность (Вт/(м2*ср*мкм))'},
            margin={'l': 40, 'b': 40, 't': 10, 'r': 10},
            legend={'x': 0, 'y': 1},
            hovermode='closest'
          )
        }
      )
    ], style={'width':'32%', 'display':'inline-block'}),
    html.Div([
      dcc.Graph(id='ph-fun-q',
        figure={
          'data': 
            [go.Scatter(
              x=np.rad2deg(np.arccos(rt3app.mu)),
              y=rt3app.Q,
              name=f'ID=Model',
            ),
            go.Scatter(
              x=Ameas,
              y=Qmeas,
              mode = 'lines+markers',
              name='measurements'
            )]
          ,
          'layout': go.Layout(
            xaxis={'title':'Угол рассеяния (град.)'},
            yaxis={'title':'Q-Интенсивность (Вт/(м2*ср*мкм))'},
            margin={'l': 40, 'b': 40, 't': 10, 'r': 10},
            legend={'x': 0, 'y': 1},
            hovermode='closest'
          )
        }
      )
    ], style={'width':'32%', 'display':'inline-block'}),
    html.Div([
      dcc.Graph(id='ph-fun-Pol',
        figure={
          'data': 
            [go.Scatter(
              x=np.rad2deg(np.arccos(rt3app.mu)),
              y=-rt3app.Q/rt3app.I*100,
              name=f'ID=Model',
            ),
            go.Scatter(
              x=Ameas,
              y=-Qmeas/Imeas*100,
              mode = 'lines+markers',
              name='measurements'
            )]
          ,
          'layout': go.Layout(
            xaxis={'title':'Угол рассеяния (град.)'},
            yaxis={'title':'Степень линейной поляризации, %'},
            margin={'l': 40, 'b': 40, 't': 10, 'r': 10},
            legend={'x': 0, 'y': 1},
            hovermode='closest'
          )
        }
      )
    ], style={'width':'32%', 'display':'inline-block'}),
  ]
  return ret


@app.callback(Output('select-disp-meas','options'),
              [Input('refresh-result-list', 'n_clicks')],
              )
def update_select_disp_meas(n_clicks):
  ret = []
  if not (n_clicks  is None):
    query=(Measurements.select())
    for item in query:
      ret.append({'label':item.datetime, 'value':item.id})
  return ret
  
@app.callback(Output('result-table', 'data'),
              [Input('select-disp-meas', 'value')]
              )
def update_select_table_direct(meas_pk):
  ret = []
  if not meas_pk is None:
    meas_pk = int(meas_pk)
    query=DirectParameters.select().join(PhaseFunction).where(DirectParameters.measType==meas_pk)
    
    if len(query)!=0:
      for item in query:
        ret.append({'0':str(item.id), '1':str(item.phaseFunction), 
                    '2':item.zenithAngle,
                    '3':item.aerosolOpticalDepth,
                    '4':item.groundAlbedo})
    
  return ret


@app.callback(Output('view-result', 'children'),
              [Input('button-view-direct-calc', 'n_clicks')],
              [State('result-table','selected_rows'),
               State('select-disp-meas', 'value'),
               State('result-table', 'data')])
def polt_results(n_clicks, selected_rows, meas_id, data):
  ret=[]
  if (not (n_clicks is None)) and (len(selected_rows)>0):
    meas_id = int(meas_id)
    
    q_meas = Measurements.get(Measurements.id==meas_id)
    tmpret=[]
    idxs=[]
    for id in selected_rows:
      idx = int(data[id]['0'])
      #print(idx)
      item = DirectParameters.get(DirectParameters.id==idx)
      F = np.load(item.filepath)
      Ang = F['Ang'][:]
      I = F['I'][:]
      Q = F['Q'][:]
      #print(Ang)
      F.close()
      tmpret.append([Ang, I, Q, idx])
      
    meas_data = np.load(q_meas.filepath)['data'][:,:]
    ret = [
      html.Div([
        dcc.Graph(id='ph-function-i',
          figure={
            'data': 
              [go.Scatter(
                x=np.rad2deg(np.arccos(it[0])),
                y=it[1],
                name=f'ID={it[3]}',
              ) for it in tmpret]+
              [go.Scatter(
                x=meas_data[:,0],
                y=meas_data[:,2],
                mode = 'lines+markers',
                name='measurements'
              )]
            ,
            'layout': go.Layout(
              xaxis={'title':'Угол рассеяния (град.)'},
              yaxis={'type':'log', 'title':'I-Интенсивность (Вт/(м2*ср*мкм))'},
              margin={'l': 40, 'b': 40, 't': 40, 'r': 40},
              legend={'x': 0, 'y': 1},
              hovermode='closest'
            )
          }
        )
      ], style={'width':'32%', 'display':'inline-block'}),
      html.Div([
        dcc.Graph(id='ph-function-q',
          figure={
            'data': [
              go.Scatter(
                x=np.rad2deg(np.arccos(it[0])),
                y=it[2],
                name=f'ID={it[3]}',
              ) for it in tmpret]+
              [go.Scatter(
                x=meas_data[:,0],
                y=meas_data[:,1],
                mode = 'lines+markers',
                name='measurements'
              )]
            ,
            'layout': go.Layout(
              xaxis={'title':'Угол рассеяния (град.)'},
              yaxis={'title':'Q-Интенсивность (Вт/(м2*ср*мкм))'+("&#10;&#13;&nbsp;1")*4},
              margin={'l': 40, 'b': 40, 't': 40, 'r': 40},
              legend={'x': 0, 'y': 1},
              hovermode='closest'
            )
          }
        )
      ], style={'width':'32%', 'display':'inline-block'}),
      html.Div([
        dcc.Graph(id='ph-function-Pol',
          figure={
            'data': 
              [go.Scatter(
                x=np.rad2deg(np.arccos(it[0])),
                y=-it[2]/it[1]*100,
                name=f'ID={it[3]}',
              ) for it in tmpret]+
              [go.Scatter(
                x=meas_data[:,0],
                y=-meas_data[:,1]/meas_data[:,2]*100,
                mode = 'lines+markers',
                name='measurements'
              )]
            ,
            'layout': go.Layout(
              xaxis={'title':'Угол рассеяния (град.)'},
              yaxis={'title':'Степень линейной поляризации, %'},
              margin={'l': 40, 'b': 40, 't': 40, 'r': 40},
              legend={'x': 0, 'y': 1},
              hovermode='closest'
            )
          }
        )
      ], style={'width':'32%', 'display':'inline-block'}),
    ]
 
  return ret

server=app.run_server
initialize()
if __name__ == '__main__':
  #initialize()
  server(debug=True)
