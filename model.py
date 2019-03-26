from peewee import *
from os import unlink

db_proxy = Proxy()

class Measurements(Model):
  id = AutoField()
  datetime = DateTimeField()
  filepath = TextField()
  class Meta:
    database = db_proxy
    indexes =(
      (('datetime',), True),
    )
    
class PhaseFunction(Model):
  DISTRS = ((1, "Степенное"),
            (2, "Логнормальное"),
            (5, "Бимодальное логнормальное"),
          )
  MODEL_TYPE = ((1, "Сферы"),
                (2, "Агломераты"),
          )
  id = AutoField()
  R0 = DoubleField(default=0.1)
  R1 = DoubleField(default=1.0)
  Mre = DoubleField(default=1.5)
  Mim = DoubleField(default=0.0)
  particlesType=IntegerField(choices=MODEL_TYPE, default=1)
  Wl = DoubleField(default=0.87)
  distrType = IntegerField(choices=DISTRS, default=1)
  p1 = DoubleField(default=-4)
  p2 = DoubleField(default=0)
  p3 = DoubleField(default=0)
  p4 = DoubleField(default=0)
  p5 = DoubleField(default=0)
  matrix = TextField()
  
  class Meta:
    database = db_proxy
    indexes = (
      (('R0','R1','Mre','Mim','particlesType', 'Wl', 'distrType', 'p1', 'p2', 'p3', 'p4', 'p5'), True),
    )
  
  def __str__(self):
    name=None
    for it in self.DISTRS:
      if it[0]==self.distrType:
        name=it[1]
        
    modelname=None
    for it in self.MODEL_TYPE:
      if it[0]==self.particlesType:
        modelname=it[1]
    
    return f"<Распред:{name}, Модель:{modelname}, \n"+\
           f"\tparams=({self.p1},{self.p2},{self.p3},{self.p4},{self.p5}),\n"+\
           f"\tmidx=({self.Mre}, {self.Mim}), R=[{self.R0},{self.R1}], Wl={self.Wl}]>"
    
class DirectParameters(Model):
  id = AutoField()
  measType = ForeignKeyField(Measurements, backref="direct_parameters")
  phaseFunction = ForeignKeyField(PhaseFunction, backref="direct_parameters")
  zenithAngle = DoubleField(default=65.0)
  aerosolOpticalDepth = DoubleField(default=0.01)
  groundAlbedo = DoubleField(default=0.06)
  filepath = TextField()
  class Meta:
    database = db_proxy
    indexes = (
      (('measType', 'phaseFunction', 'zenithAngle', 'aerosolOpticalDepth', 'groundAlbedo',), True),
    )

def recreate():
  unlink("datastore.db")
  db = SqliteDatabase("datastore.db")
  db_proxy.initialize(db)
  with db_proxy:
    db_proxy.create_tables([Measurements, PhaseFunction, DirectParameters])
    
def initialize():
  db = SqliteDatabase("datastore.db")
  db_proxy.initialize(db)
  
if __name__ == '__main__':
  recreate()