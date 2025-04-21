import pandas as pd

MAX_C = 20.0

class Item:
    def __init__(self, weight, value, quantity):
        self.weight = weight
        self.value = value
        self.quantity = quantity

    def __repr__(self):
        return f"Item(weight={self.weight}, value={self.value}, quantity={self.quantity})"

def readItems(filename):
    df = pd.read_excel(filename)
    items = []
    for _, row in df.iterrows():
        weight = row["Peso_kg"]
        value = row["Valor"]
        quantity = int(row["Cantidad"])
        items.append(Item(weight, value, quantity))
    return items


#path = 'data/Mochila_capacidad_maxima_20kg.xlsx'
