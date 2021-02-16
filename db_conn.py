import pymysql
import numpy as np
import time
import datetime

data = pymysql.connect(host='203.250.32.83', user='band_root', password='wlgns12', db='band_db')
curs = data.cursor()
curs.execute("SELECT object_temp,ambient_temp,gas FROM `detection_table` ORDER BY `detection_table`.`datetime` DESC Limit 1")
result = curs.fetchall()

Obj_tp = np.array([(x[0]) for x in result])
Amb_tp = np.array([(x[1]) for x in result])
Gas = np.array([(x[2]) for x in result])

if(0 <= Gas or 0 <= Obj_tp or 0 <= Amb_tp):
    print(1)
elif(Gas >= 300 or Obj_tp >= 40 or Amb_tp >= 50):
    print(2)
elif(Gas >= 500 or Obj_tp >= 60 or Amb_tp >= 70):
    print(3)

#print(list(result))
print(float(Obj_tp))
