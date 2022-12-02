class Node(object):
    def __init__(self, value=None, next_node=None):
        self.value = value
        self.next = next_node

    def __str__(self):
        return f"[Node with value {self.value}]"


def print_linked_list(head):
    cur = head
    while cur is not None:
        print(cur)
        cur = cur.next


def reverse_linked_list(head):
    prev = None
    cur = head
    while cur is not None:
        next = cur.next
        cur.next = prev
        prev = cur
        cur = next
    return prev


def sort_nodes(head):
    cur = head
    while cur is not None:
        next = cur.next
        while next is not None:
            if cur.value > next.value:
                cur.value, next.value = next.value, cur.value
            next = next.next
        cur = cur.next
    return head

h, a, b, c, d = Node(1), Node(0), Node(3), Node(4), Node(5)

h.next = a
a.next = b
b.next = c
c.next = d

print(sort_nodes(h))


'''Выясните, сколько в среднем выходи полный счёт по пятницам на ланч у курящих мужчин (датасет tips)'''

import pandas as pd

df = pd.read_csv('data/tips.csv')
le_masque = df["sex"] == "Male"
le_masque &= (df["day"] == "Fri")
le_masque &= df['time'] == 'Lunch'
le_masque &= df["smoker"] == "Yes"
print(df[le_masque]["total_bill"].mean())

'''Создайте новый столбец с плотностью населения. Переименуйте названия первого и третьего столбца на русский язык'''

import pandas as pd

df = pd.DataFrame({
     'country': ['Kazakhstan', 'Russia', 'Belarus', 'Ukraine'],
     'population': [17.04, 143.5, 9.5, 45.5],
     'square': [2724902, 17125191, 207600, 603628]
 })

df['density'] = df['population'] / df['square'] * 1000000
df.rename(columns={'country': 'страна', 'square': 'площадь'}, inplace=True)
print(df)
