import sys, os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sources = []
ratio_dic = {}

file = open('averages_lstm_nosw_2.csv', 'r')
file.readline()
line = file.readline()
while(line):
    arr = line.split(',')
    sources.append(arr[0])
    ratio = float(arr[1].rstrip('\n'))
    ratio_dic[arr[0]] = ratio
    line = file.readline()
file.close()

file = open('averages_lstm_nosw.csv', 'r')
file.readline()
line = file.readline()
while(line):
    arr = line.split(',')
    sources.append(arr[0])
    ratio = float(arr[1].rstrip('\n'))
    ratio_dic[arr[0]] = ratio
    line = file.readline()

#skew averages based on opeds
#greater number => less ops
ops_dic = {
    'NewYorkTimes': 52,
    'Breitbart': 8,
    'CNN': 32,
    'BusinessInsider': 39,
    'Atlantic': 46,
    'FoxNews': 20,
    'TalkingPointsMemo': 41,
    'BuzzfeedNews': 51,
    'NationalReview': 51,
    'NewYorkPost': 20,
    'Guardian': 48,
    'NPR': 56,
    'Reuters': 62,
    'Vox': 43,
    'WashingtonPost': 51,
    'abc-news': 57,
    'associated-press': 62,
    'bbc-news': 54,
    'bloomberg': 58,
    'cnbc': 57,
    'daily-mail': 19,
    'financial-times': 48,
    'fortune': 46,
    'msnbc': 34,
    'nbc-news': 57,
    'politico': 55,
    'the-wall-street-journal': 53,
    'time': 43,
    'vice-news': 42,
    'usa-today': 52
}

#-42 - 0: skews left, 0 neutral, >0: skews right
bias_dic = {
    'NewYorkTimes': -5,
    'Breitbart': 34,
    'CNN': -6,
    'BusinessInsider': 0,
    'Atlantic': -15,
    'FoxNews': 27,
    'TalkingPointsMemo': -13,
    'BuzzfeedNews': -15,
    'NationalReview': 20,
    'NewYorkPost': 18,
    'Guardian': -6,
    'NPR': -5,
    'Reuters': 0,
    'Vox': -16,
    'WashingtonPost': -10,
    'abc-news': 0,
    'associated-press': 0,
    'bbc-news': -3,
    'bloomberg': 4,
    'cnbc': -3,
    'daily-mail': 13,
    'financial-times': 3,
    'fortune': 5,
    'msnbc': -19,
    'nbc-news': -3,
    'politico': -3,
    'the-wall-street-journal': 11,
    'time': -1,
    'vice-news': -10,
    'usa-today': 0
}

LCR_dic = {}
LCR_dic['NewYorkTimes'] = ['L1', 13694, 23141, 'D']
LCR_dic['Breitbart'] = ['R2', 13482, 7154, 'A']
LCR_dic['CNN'] = ['L1', 24816, 25685, 'SD']
LCR_dic['Atlantic'] = ['L1', 3301, 2075, 'A']
LCR_dic['FoxNews'] = ['R1', 19445, 29070, 'D']
LCR_dic['TalkingPointsMemo'] = ['X', 'X', 'X', 'X']
LCR_dic['BuzzfeedNews'] = ['L1', 3689, 5111, 'SD']
LCR_dic['NationalReview'] = ['R2', 7528, 3714, 'SA']
LCR_dic['NewYorkPost'] = ['R2', 2461, 1835, 'SA']
LCR_dic['Guardian'] = ['L1', 5018, 3122, 'A']
LCR_dic['NPR'] = ['C', 17492, 12515, 'SA']
LCR_dic['Reuters'] = ['C', 5295, 3654, 'SA']
LCR_dic['Vox'] = ['L2', 9233, 11450, 'SD']
LCR_dic['WashingtonPost'] = ['L1', 22564, 13775, 'A']
LCR_dic['abc-news'] = ['L1', 10589, 7818, 'SA']
LCR_dic['associated-press'] = ['C', 5751, 4048, 'SA']
LCR_dic['bbc-news'] = ['C', 8795, 8239, 'SA']
LCR_dic['bloomberg'] = ['C', 4861, 5571, 'SD']
LCR_dic['cnbc'] = ['C', 1410, 3680, 'D']
LCR_dic['daily-mail'] = ['R2', 1351, 1128, 'SA']
LCR_dic['fortune'] = ['X', 'X', 'X', 'X']
LCR_dic['msnbc'] = ['L2', 3943, 1534, 'A']
LCR_dic['nbc-news'] = ['L1', 1153, 628, 'A']
LCR_dic['politico'] = ['L1', 12314, 19877, 'D']
LCR_dic['the-wall-street-journal'] = ['C', 11039, 16765, 'D']


#out of opinion %, multiply by ops # (news arts), add % to fact
new_ratio_dic = {}
for source in sources:
    non_op = (ops_dic[source] / 64) * 100
    subj_percent = 100 - (ratio_dic[source] * 100)
    scale = (non_op / 100) * (ratio_dic[source])
    new_ratio = 100 - (subj_percent - (subj_percent * scale))
    new_ratio_dic[source] = new_ratio
print(new_ratio_dic)

########plot
x = [] #bias
y = [] #ratios
p = []
for source in sources:
    x.append(bias_dic[source])
    y.append(new_ratio_dic[source])
    LCR = LCR_dic[source][0]
    if LCR == 'L2':
        color = '#0d76e0'
    elif LCR == 'L1':
        color = '#76cbf2'
    elif LCR == 'R1':
        color = '#f7b4b4'
    elif LCR == 'R2':
        color = '#ff0000'
    elif LCR == 'X':
        color = '#e1f274'
    else:
        color = '#a274f2'
    p.append(color)


fig = plt.figure(figsize=(10,5))
fig.add_axes()
a1 = fig.add_subplot(111)
a1.set(title='Reported Bias vs. Predicted Objectivity',
        xlabel='Bias [AdFontesMedia]',
        ylabel='Percentage of Sentences Labeled Objective')
a1.set_xlim([-40, 40])
a1.set_ylim([50,100])

show_sources = ['Breitbart', 'Reuters', 'msnbc', 'NewYorkTimes', 'FoxNews']
for i, source in enumerate(sources):
    if source in show_sources:
      a1.annotate(source, (x[i], y[i]))


plt.scatter(x,y, c=p)

p0 = mpatches.Patch(color='#ffffff', label='[AllSides]')
p1 = mpatches.Patch(color='#0d76e0', label='Far Left')
p2 = mpatches.Patch(color='#76cbf2', label='Left')
p3 = mpatches.Patch(color='#a274f2', label='Center')
p4 = mpatches.Patch(color='#f7b4b4', label='Right')
p5 = mpatches.Patch(color='#ff0000', label='Far Right')
p6 = mpatches.Patch(color='#e1f274', label='Not Ranked')
plt.legend(handles=[p1,p2,p3,p4,p5,p6,p0])


plt.show()
