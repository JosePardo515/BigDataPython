import codecs
import csv
import os
import time

from contextlib import closing

import matplotlib.pyplot as plt
import requests

from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.edge.options import Options

edgedriver = "C:/edgedriver_win64/msedgedriver.exe"

def ejemplo1():
    url = "http://www.mambiente.munimadrid.es/opendata/horario.txt"
    resp = requests.get(url)
    path = f"BigDataPython\src\Cap4\Test\horario.txt"
    with open(path, 'wb') as output:
        output.write(resp.content)

    with open(path) as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        for row in readCSV:
            if (row[0]+row[1]+row[2]=='28079004' and row[3]=='12'):
                plt.title("Óxido de nitrógeno: "+row[8]+"/"+row[7]+"/"+row[6])
                hora = 0
                desp = 9
                vs = []
                while (hora<=23 and row[desp+2*hora+1]=='V'):
                    vs.append(row[desp+2*hora])
                    hora +=1
                plt.plot(range(hora), vs)
                plt.show()

def ejemplo2():
    url = "http://www.mambiente.munimadrid.es/opendata/horario.txt"

    with closing(requests.get(url, stream=True)) as r:
        reader = csv.reader(codecs.iterdecode(r.iter_lines(), 'utf-8'), 
                            delimiter=',', quotechar='"')
        for row in reader:
            if (row[0]+row[1]+row[2]=='28079004' and row[3]=='12'):
                plt.title(row[8]+"/"+row[7]+"/"+row[6])
                hora = 0
                desp = 9
                vs = []
                while (hora<=23 and row[desp+2*hora+1]=='V'):
                    vs.append(row[desp+2*hora])
                    hora +=1
                plt.plot(range(hora), vs) 
                plt.show()

def ejemplo3():
    URI = 'http://maps.googleapis.com/maps/api/directions/json'

    params = dict(
        origin='Madrid,Spain',
        destination='Barcelona,Spain',
        mode='driving'
    )

    resp = requests.get(url=URI, params=params)
    data = resp.json()
    print(data['routes'][0]['legs'][0]['duration']['text'])

    '''
    You must use an API key to authenticate each request 
    to Google Maps Platform APIs. For additional information, 
    please refer to http://g.co/dev/maps-no-account
    '''

def ejemplo4():
    url = f"BigDataPython\src\Cap4\Test\mini.html"
    #url = r"c:\rafa\docencia\1718\libro\pythonspark\mini.html"
    with open(url, "r") as f:
        page = f.read()

    soup = BeautifulSoup(page, "html.parser")
    print(soup.prettify())

    hijosDoc = list(soup.children)
    print([type(item) for item in hijosDoc])
    #[<class 'bs4.element.Doctype'>, <class 'bs4.element.NavigableString'>, <class 'bs4.element.Tag'>]
    print(hijosDoc)

    html = hijosDoc[2]
    print(list(html.children))

    body = list(html.children)[3]
    print(list(body.children))
    divDate = list(body.children)[1]
    print([type(item) for item in list(divDate.children)])

    print(divDate.get_text())
    #Fecha 25/03/2035
    divs = soup.find_all("div")
    print(divs[0].get_text())
    #Fecha 25/03/2035
    print(soup.find("div").get_text())
    #Fecha 25/03/2035
    print(soup.find("div", id="date").get_text())
    #Fecha 25/03/2035
    print(soup.select("body div")[0].get_text())

def ejemplo5():
    url = "https://www.timeanddate.com/worldclock/timezone/utc"

    r  = requests.get(url)
    print(r.content)

    soup = BeautifulSoup(r.content, "html.parser")
    print(soup.prettify())

def ejemplo6():
    '''
    Ya no existe este servicio
    '''
    url = "https://www.boe.es/sede_electronica/informacion/hora_oficial.php"
    r  = requests.get(url)
    print(r)

    soup  = BeautifulSoup(r.content, "html.parser")
    cajaHora = soup.find("p",class_="cajaHora")
    print(list(cajaHora.children)[1].get_text())

def InitEdge(url):
    os.environ["webdriver.edge.driver"] = edgedriver

    edge_options = Options()
    edge_options.add_argument('--no-sandbox')
    edge_options.add_experimental_option('excludeSwitches', ['enable-logging'])
    edge_options.use_chromium = True
    edge_options.add_argument("start-maximized")
    edge_options.add_argument("--headless")
    edge_options.add_argument("--window-size=1920x1080")

    driver = webdriver.Edge(executable_path=edgedriver,options=edge_options)
    driver.get(url)
    return driver

if __name__ == '__main__':
    Test = 'ej6'
    os.system('cls')
    if Test == 'ej1':
        ejemplo1()
    
    if Test == 'ej2':
        ejemplo2()

    if Test == 'ej3':
        ejemplo3()

    if Test == 'ej4':
        ejemplo4()

    if Test == 'ej5':
        ejemplo5()

    if Test == 'ej6':
        ejemplo6()
