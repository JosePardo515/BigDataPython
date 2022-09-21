chromedriver = "c:/hlocal/tdm/chromedriver.exe" # cambiar esta variable con el path a nuestro chromedriver
edgedriver = "C:/edgedriver_win64/msedgedriver.exe"
import os
from selenium import webdriver  # si da error, desde anaconda prompt hacer pip install --user  selenium
from selenium.webdriver.common.by import By
from selenium.webdriver.edge.options import Options
import time

def chrome_examp():
    os.environ["webdriver.chrome.driver"] = chromedriver
    chrome_options = webdriver.ChromeOptions()
    chrome_options.add_argument('--no-sandbox')
    driver = webdriver.Chrome(executable_path=chromedriver,options=chrome_options)

def InitEdge(url):
    os.environ["webdriver.edge.driver"] = edgedriver
    edge_options = Options()
    edge_options.add_argument('--no-sandbox')
    edge_options.add_experimental_option('excludeSwitches', ['enable-logging'])
    edge_options.use_chromium = True
    edge_options.add_argument("start-maximized")
    driver = webdriver.Edge(executable_path=edgedriver,options=edge_options)
    driver.get(url)
    return driver

def edge_examp1():
    url = 'https://www1.sedecatastro.gob.es/CYCBienInmueble/OVCBusqueda.aspx'

    driver=InitEdge(url)
    coord   = driver.find_element(By.LINK_TEXT, 'COORDENADAS')
    coord.click()
    time.sleep(1)

    lat = driver.find_element(By.ID,"ctl00_Contenido_txtLatitud")
    lon = driver.find_element(By.ID,"ctl00_Contenido_txtLongitud")
    latitud  = "41.545639 "
    longitud = "1.893817"
    lat.send_keys(latitud)
    lon.send_keys(longitud)

    datos = driver.find_element(By.ID,"ctl00_Contenido_btnDatos")
    datos.click()

    xpath = "//*[./span/text()='Referencia catastral']//label"

    etiqs = driver.find_element(By.XPATH,xpath)
    print(etiqs.text)
    xpath = "//*[./span/text()='Uso principal']//label"

    etiqs = driver.find_element(By.XPATH,xpath)
    print(etiqs.text)

    html = driver.find_element(By.XPATH,"/html")
    print(html.text)

    head = driver.find_element(By.XPATH,"/html/head")
    body = driver.find_element(By.XPATH,"/html/body")
    html2 = body.find_element(By.XPATH,"/html")

    hijos = driver.find_elements(By.XPATH,"/html/body/*")
    for element in hijos:
        print(element.tag_name)

    divs = driver.find_elements(By.XPATH,"/html/body/*/div")
    print(len(divs))

    divs = body.find_elements(By.XPATH,"./*/div")
    print(len(divs))

    divs = driver.find_elements(By.XPATH,"/html/body//div")
    print(len(divs))

    labels = driver.find_elements(By.XPATH,"//label")
    print(len(labels))

    id = "ctl00_Contenido_tblInmueble"
    div = driver.find_element(By.ID,id)
    label = div.find_element(By.XPATH,"//label")
    print(label.text)

    xpath = "//*[./span/text()='Referencia catastral']//label"
    etiqs = driver.find_element(By.XPATH,xpath)
    print(etiqs.text)

    clase = driver.find_elements(By.XPATH,"(//label)[position()=3]")
    print(clase[0].text)

    etiqs = driver.find_elements(By.XPATH,"//label")
    print(etiqs[2].text)

    ulti = driver.find_elements(By.XPATH,"(//label)[last()]")
    print(ulti[0].text)

    driver.close()

def edge_examp():
    os.environ["webdriver.edge.driver"] = edgedriver
    edge_options = Options()
    edge_options.add_argument('--no-sandbox')

    driver = webdriver.Edge(executable_path=edgedriver,options=edge_options)
    driver.get('https://bing.com')

    element = driver.find_element(By.ID, 'sb_form_q')
    element.send_keys('WebDriver')
    element.submit()

    time.sleep(5)
    driver.quit()

def edge_examp3():
    url = "https://www.eurovent-certification.com/es/advancedsearch/result?program=LCP-HP"
    driver=InitEdge(url)

if __name__ == '__main__':
    Test = 'eurovent'

    if Test == 'edge':
        edge_examp1()

    if Test == 'eurovent':
        edge_examp3()