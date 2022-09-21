import csv
import json
from operator import truediv
import xml.etree.ElementTree as ET

ruta_csv = 'BigDataPython\data\Cap1\subvenciones.csv'
ruta_json = 'BigDataPython\data\Cap1\subvenciones.json'
ruta_json_agrup = 'BigDataPython\data\Cap1\subvenciones_agrupadas.json'
ruta_json_agrup_error = 'BigDataPython\data\Cap1\subvenciones_agrupadas_error.json'
ruta_json_agrup_gasto = 'BigDataPython\data\Cap1\subvenciones_agrupadas_con_gasto.json'
ruta_xml = 'BigDataPython\data\Cap1\subvenciones.xml'
ruta_xml_lista_total = 'BigDataPython\data\Cap1\subvenciones_lista_total.xml'

def leercsv1():
    with open(ruta_csv, encoding='latin1') as fichero_csv:
        lector = csv.reader(fichero_csv)
        next(lector, None)
        asociaciones = {}
        for linea in lector:
            centro = linea[0]
            subvencion = float(linea[2])
            if centro in asociaciones:
                asociaciones[centro] = asociaciones[centro] + subvencion
            else:
                asociaciones[centro] = subvencion
        print(asociaciones)

def leercsv2():
    with open(ruta_csv, encoding='latin1') as fichero_csv:
        dict_lector = csv.DictReader(fichero_csv)
        asociaciones = {}
        for linea in dict_lector:
            centro = linea['Asociación']
            subvencion = float(linea['Importe'])
            if centro in asociaciones:
                asociaciones[centro] = asociaciones[centro] + subvencion
            else:
                asociaciones[centro] = subvencion
        print(asociaciones)

def leerjson1():
    with open(ruta_json, encoding='utf-8') as fich_lect,open(ruta_json_agrup, 'w', encoding='utf-8') as fich_escr:
        data = json.load(fich_lect)
        asoc_str = "Asociación"
        act_str = "Actividad Subvencionada"
        imp_str = "Importe en euros"
        lista = []
        lista_act = []
        asoc_actual = ""
        dicc = {}
        for elem in data:
            asoc = elem[asoc_str]
            act = elem[act_str]
            imp = elem[imp_str]
            if asoc_actual != asoc:
                dicc["Actividades"] = lista_act
                dicc = {"Asociación": asoc}
                lista.append(dicc)
                lista_act = []
            lista_act.append({act_str : act, imp_str : imp})
            asoc_actual = asoc
        print(lista)
        json.dump(lista, fich_escr, ensure_ascii=False, indent=4) # , sort_keys=False

def leerxml1():
    arbol = ET.parse(ruta_xml)
    asociaciones = {}
    for fila in arbol.findall('Row'): # raiz.iter('Row'):
        centro = fila.find('Asociaci_n').text
        subvencion = float(fila.find('Importe').text)
        if centro in asociaciones:
            asociaciones[centro] = asociaciones[centro] + subvencion
        else:
            asociaciones[centro] = subvencion
    print(asociaciones)

def leerxml2():
    arbol = ET.parse(ruta_xml)
    raiz = arbol.getroot()
    nuevo = ET.ElementTree()
    raiz_nueva = ET.Element("Raiz")
    nuevo._setroot(raiz_nueva)
    elem_actual = ET.Element("Asociacion")
    asoc_actual = ""
    actividades = ET.SubElement(elem_actual, "Actividades")
    gasto = 0
    for fila in raiz.findall('Row'):
        asoc = fila.find('Asociaci_n').text
        act = fila.find('Actividad_Subvencionada').text
        imp = float(fila.find('Importe').text)
        if asoc_actual != asoc:
            gas_total = ET.SubElement(elem_actual, "Total")
            gas_total.text = str(gasto)
            elem_actual = ET.SubElement(raiz_nueva, "Asociacion")
            elem_actual.set('nombre', asoc)
            actividades = ET.SubElement(elem_actual, "Actividades")
            gasto = 0
        act_elem = ET.SubElement(actividades, "Actividad")
        nom_elem = ET.SubElement(act_elem, "Nombre")
        nom_elem.text = act
        imp_elem = ET.SubElement(act_elem, "Subvencion")
        imp_elem.text = str(imp)
        gasto = gasto + imp
        asoc_actual = asoc
    nuevo.write(ruta_xml_lista_total)    

if __name__ == '__main__':
    Test = 'xml'

    if Test == 'csv':
        leercsv1()
        leercsv2()

    if Test == 'json':
        leerjson1()

    if Test == 'xml':
        leerxml1()
        leerxml2()