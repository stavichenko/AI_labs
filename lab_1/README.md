# Лабороторна работа 1. Розпізнавання об'єктів з якісними характеристиками
С.Г. Ставиченко

В данній работі реалізовнна алгоритм розпізновання об'єктів за бінарним вектором
їх якісних характеристик, з використанням різних функцій подібності, таких як:
 - функція подібності Рассела та Рао
 - функція подібності Жокара та Нідмена
 - функція подібності Дайса
 - функція подібності Сокаля та Сніфу
 - функція подібності Сокаля та Мішнера
 - функція подібності Кульжинського
 - функція подібності Юла
 - відстань Хеммінгу
 - та своя реалізація метрики IOU (Intersection Over Union)

IOU - показує відношення об'єднання множин до їх перетину, та є типовую метрікою в машнному навчанні.
В значеннях a, b, g, h які використовуються в інших функціяї подібності, може бути виражена як:
 
IOU = a / (b + g + h) 

Для обчислень використовувалась бібліотека numpy.

Наступні вектори ознак дають значення відастані Хеммінга яке дорівнює 7
````
{1, 0, 1, 0, 1, 0, 1, 0}
{0, 1, 0, 1, 0, 1, 0, 0}
````
